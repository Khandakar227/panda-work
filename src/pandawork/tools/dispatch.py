"""Running tool calls: the fixed pipeline every call goes through.

    log call -> tier gate -> guards (deny-only) -> budget -> validate args -> run (deadline)
             -> render -> log result

Calls from one model turn run in the order given. If an ACT call fails or only partly succeeds,
the turn's remaining ACT calls are skipped, so the robot never acts on a plan that no longer
fits. Every failure becomes an error result the model can read; nothing here raises.
"""
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from pydantic import ValidationError

from .base import Group, Tool, ToolContext, ToolError, ToolResult
from .registry import ToolRegistry
from .rendering import render_result, text_block, to_json

# A guard may only deny: it returns a reason to refuse the call, or None.
Guard = Callable[[ToolContext, Tool, "ToolCall"], str | None]

WALL_GUARD_FACTOR = 20.0  # wall-clock hang guard = 20 x the simulated-time budget
CALL_TOPIC = "/agent/tool_call"
RESULT_TOPIC = "/agent/tool_result"
TURN_END_TOPIC = "/agent/turn_end"


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


def episode_over_guard(ctx: ToolContext, tool: Tool, call: ToolCall) -> str | None:
    """After done() the episode is over; refuse anything else."""
    return "the episode has already ended" if ctx.state.get("done") else None


class Dispatcher:
    def __init__(self, registry: ToolRegistry, ctx: ToolContext,
                 guards: Iterable[Guard] = (episode_over_guard,),
                 reminder_at: Iterable[int] = (3, 5, 8)):
        self.registry = registry
        self.ctx = ctx
        self.guards = list(guards)
        self.reminder_at = frozenset(reminder_at)
        self._last_call_key: str | None = None
        self._repeats = 0
        self.turn = 0  # number of completed turns
        self._halted = False
        ctx.world.bus.set_retention("/agent/*", float("inf"))  # keep the whole episode's events

    def run_batch(self, calls: list[ToolCall]) -> list[dict]:
        """Run one turn's tool calls; returns Anthropic tool_result blocks in the same order.
        The budget footer is added once, at the end of the last result."""
        self.begin_turn()
        blocks = []
        for call in calls:
            result, content = self.call_in_turn(call)
            block = {"type": "tool_result", "tool_use_id": call.id, "content": content}
            if not result.ok:
                block["is_error"] = True
            blocks.append(block)
        footer = self.end_turn(len(calls))
        if blocks:
            blocks[-1]["content"].append(text_block(footer))
        return blocks

    # A turn can also be driven one call at a time (an agent loop that delivers tool calls
    # individually, like Claude Code): begin_turn(), call_in_turn() per call, end_turn().

    def begin_turn(self) -> None:
        self._halted = False  # set when an action fails: later actions this turn are skipped

    def call_in_turn(self, call: ToolCall) -> tuple[ToolResult, list[dict]]:
        tool = self.registry.get(call.name)
        is_act = tool is not None and tool.group is Group.ACT
        result, content = self.run(call, skip_act=self._halted and is_act)
        if is_act and result.halts_batch:
            self._halted = True
        return result, content

    def end_turn(self, n_calls: int) -> str:
        """Close the turn (logged as /agent/turn_end); returns the budget footer line."""
        footer = self.ctx.budget.footer()
        self.ctx.world.bus.publish(TURN_END_TOPIC, {"turn": self.turn, "n_calls": n_calls,
                                                    "footer": footer}, frame_id="agent")
        self.turn += 1
        return footer

    def run(self, call: ToolCall, skip_act: bool = False) -> tuple[ToolResult, list[dict]]:
        """Run one call through the pipeline; returns the result and the model-facing content."""
        bus = self.ctx.world.bus
        bus.publish(CALL_TOPIC, {"call_id": call.id, "tool": call.name, "args": to_json(call.args),
                                 "turn": self.turn}, frame_id="agent")
        tool = self.registry.get(call.name)
        result, trace = self._decide(tool, call, skip_act)
        content = tool.render(result) if tool is not None else render_result(result)
        reminder = self._repeat_reminder(call)
        if reminder:
            content.insert(1, text_block(reminder))
        bus.publish(RESULT_TOPIC, self._result_event(call, result, content, trace), frame_id="agent")
        return result, content

    def _decide(self, tool: Tool | None, call: ToolCall, skip_act: bool) -> tuple[ToolResult, str]:
        ctx = self.ctx
        if tool is None:
            names = ", ".join(t.name for t in self.registry.available(ctx.tier))
            return ToolResult.failure("unknown_tool", f"there is no tool named {call.name!r}",
                                      hint=f"available tools: {names}"), ""
        if ctx.tier not in tool.tiers:
            return ToolResult.failure("tool_not_available",
                                      f"{tool.name} is not available in the {ctx.tier.value} tier"), ""
        for guard in self.guards:
            reason = guard(ctx, tool, call)
            if reason:
                return ToolResult.failure("denied", reason), ""
        if skip_act:
            return ToolResult.failure(
                "skipped", "not run because an earlier action in this turn did not succeed",
                hint="read the earlier result, then decide whether to try this action again"), ""
        reason = ctx.budget.exhausted()
        if reason:
            return ToolResult.failure("budget_exhausted", reason,
                                      hint="the episode is ending; call done() now"), ""
        ctx.budget.charge_call()
        try:
            args = tool.Args.model_validate(call.args)
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}"
                                 for e in exc.errors())
            return ToolResult.failure("invalid_arguments", problems,
                                      hint="check the tool's input schema"), ""
        return self._execute(tool, args)

    def _execute(self, tool: Tool, args) -> tuple[ToolResult, str]:
        self.ctx.deadline = self.ctx.world.clock.sim_time + tool.timeout_s
        self.ctx.wall_deadline = time.monotonic() + WALL_GUARD_FACTOR * tool.timeout_s
        try:
            result = tool.run(self.ctx, args)
            if not isinstance(result, ToolResult):
                raise TypeError(f"{tool.name}.run returned {type(result).__name__}, not ToolResult")
            return result, ""
        except ToolError as exc:
            return ToolResult(error=exc.info), ""
        except Exception as exc:  # never let a tool bug end the episode; log the traceback
            return ToolResult.failure(
                "internal_error", f"{type(exc).__name__}: {exc}",
                hint="this is a problem in the harness, not in your request; try another approach",
            ), traceback.format_exc()
        finally:
            self.ctx.deadline = None

    def _repeat_reminder(self, call: ToolCall) -> str | None:
        key = call.name + to_json(call.args)
        self._repeats = self._repeats + 1 if key == self._last_call_key else 1
        self._last_call_key = key
        if self._repeats in self.reminder_at:
            return (f"note: this is call {self._repeats} in a row to {call.name} with identical "
                    "arguments. If it isn't making progress, try a different approach.")
        return None

    @staticmethod
    def _result_event(call: ToolCall, result: ToolResult, content: list[dict], trace: str) -> dict:
        event = {
            "call_id": call.id, "tool": call.name, "ok": result.ok,
            "error_code": result.error.code if result.error else "",
            "data": to_json(result.data),
            "text": "\n".join(b["text"] for b in content if b["type"] == "text"),
            "n_images": len(result.images),
        }
        for i, image in enumerate(result.images):
            event[f"image_{i}"] = image
        if trace:
            event["traceback"] = trace
        return event


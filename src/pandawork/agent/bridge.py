"""Bridge between Claude Code (via the Claude Agent SDK) and the harness's tool dispatcher.

The episode's tools are exposed as an in-process MCP server. Claude Code runs tool calls one
at a time and does not tell a handler which call it is serving, so the bridge pairs things up:
  - a PreToolUse hook sees each call's tool_use_id (queued until its handler runs);
  - the message stream shows which assistant message issued each tool_use_id.
One assistant message is one turn: actions after a failed action in the same message are
skipped, and turns are logged so replay can reproduce them.

All simulation work and every bus publication run on one worker thread: the world is not
thread-safe, and OpenGL rendering contexts belong to the thread that created them.
"""
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from claude_agent_sdk import AssistantMessage, ToolUseBlock, create_sdk_mcp_server
from claude_agent_sdk import tool as sdk_tool

from ..tools import ToolCall, ToolResult
from ..tools.rendering import text_block
from ..tasks import Episode

SERVER = "panda"
PREFIX = f"mcp__{SERVER}__"
TURN_WAIT_S = 5.0  # how long a handler waits to learn which message issued its call


def to_mcp(content: list[dict]) -> list[dict]:
    """Anthropic content blocks -> MCP tool-result content."""
    out = []
    for block in content:
        if block["type"] == "text":
            out.append({"type": "text", "text": block["text"]})
        elif block["type"] == "image":
            out.append({"type": "image", "data": block["source"]["data"],
                        "mimeType": block["source"]["media_type"]})
    return out


class ToolBridge:
    def __init__(self, episode: Episode):
        self.episode = episode
        self.dispatcher = episode.dispatcher
        self.tools = self.dispatcher.registry.available(episode.world.tier)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sim")
        self._pending: list[tuple[str, dict, str]] = []  # (tool, input, tool_use_id) from hooks
        self._turn_of: dict[str, str] = {}               # tool_use_id -> assistant message id
        self._seen = asyncio.Event()
        self._current_turn: str | None = None
        self._calls_in_turn = 0
        self._n = 0
        self.unmatched_calls = 0  # handler calls that could not be paired with a tool_use_id

    # ---- configuration for ClaudeAgentOptions ------------------------------------------------

    def mcp_server(self):
        return create_sdk_mcp_server(SERVER, "1.0.0", [
            sdk_tool(t.name, t.description, t.schema()["input_schema"])(self._handler(t.name))
            for t in self.tools])

    @property
    def allowed_tools(self) -> list[str]:
        return [PREFIX + t.name for t in self.tools]

    async def pre_tool_use(self, input_data: dict, tool_use_id: str | None, context: Any) -> dict:
        name = input_data.get("tool_name", "")
        if name.startswith(PREFIX):
            self._pending.append((name[len(PREFIX):], input_data.get("tool_input") or {},
                                  input_data.get("tool_use_id") or tool_use_id or ""))
        return {}

    # ---- message stream ---------------------------------------------------------------------

    def observe(self, message: Any) -> None:
        """Feed every message from the SDK stream (learns which message issued which call)."""
        if isinstance(message, AssistantMessage):
            turn = message.message_id or message.uuid or ""
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    self._turn_of[block.id] = turn
            self._seen.set()

    async def on_sim(self, fn: Callable, *args):
        """Run `fn` on the simulation thread."""
        return await asyncio.get_running_loop().run_in_executor(self.executor, fn, *args)

    def end_turn(self) -> None:
        """Simulation thread: close the open turn (logged) once the agent has stopped."""
        self._end_turn()

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)

    # ---- tool calls -------------------------------------------------------------------------

    def _handler(self, name: str):
        async def handler(args: dict) -> dict:
            tool_use_id = self._claim(name, args)
            turn = await self._turn_for(tool_use_id)
            self._n += 1
            call = ToolCall(tool_use_id or f"unpaired_{self._n}", name, args)
            result, content = await self.on_sim(self._run, turn, call)
            return {"content": to_mcp(content), "is_error": not result.ok}
        return handler

    def _claim(self, name: str, args: dict) -> str | None:
        for i, (pending_name, pending_args, tool_use_id) in enumerate(self._pending):
            if pending_name == name and pending_args == args:
                del self._pending[i]
                return tool_use_id
        self.unmatched_calls += 1
        return None

    async def _turn_for(self, tool_use_id: str | None) -> str | None:
        if tool_use_id is None:
            return self._current_turn
        deadline = asyncio.get_running_loop().time() + TURN_WAIT_S
        while tool_use_id not in self._turn_of:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return self._current_turn  # unknown: treat as part of the current turn
            self._seen.clear()
            try:
                await asyncio.wait_for(self._seen.wait(), remaining)
            except asyncio.TimeoutError:
                pass
        return self._turn_of[tool_use_id]

    def _run(self, turn: str | None, call: ToolCall) -> tuple[ToolResult, list[dict]]:
        """Simulation thread: open a new turn when the issuing message changes, run the call,
        and append the budget footer (each result carries it: calls arrive one at a time)."""
        if turn != self._current_turn or self._current_turn is None:
            self._end_turn()
            self.dispatcher.begin_turn()
            self._current_turn, self._calls_in_turn = turn or f"turn{self.dispatcher.turn}", 0
        result, content = self.dispatcher.call_in_turn(call)
        self._calls_in_turn += 1
        return result, content + [text_block(self.dispatcher.ctx.budget.footer())]

    def _end_turn(self) -> None:
        if self._current_turn is not None:
            self.dispatcher.end_turn(self._calls_in_turn)
            self._current_turn = None


def assistant_event(message: AssistantMessage) -> dict[str, Any]:
    """What the model said in one message, for the episode log (/agent/assistant)."""
    text, thinking, tool_uses = [], [], []
    for block in message.content:
        kind = type(block).__name__
        if kind == "TextBlock":
            text.append(block.text)
        elif kind == "ThinkingBlock":
            thinking.append(block.thinking)
        elif kind == "ToolUseBlock":
            tool_uses.append({"id": block.id, "name": block.name, "input": block.input})
    return {"message_id": message.message_id or "", "model": message.model,
            "text": "\n".join(text), "thinking": "\n".join(thinking),
            "tool_uses": json.dumps(tool_uses, sort_keys=True),
            "stop_reason": message.stop_reason or "",
            # usage as streamed: output counts can be partial; the episode total comes from
            # the final ResultMessage (see runner)
            "stream_usage": json.dumps(message.usage or {}, sort_keys=True)}

"""Interactive session: type instructions, watch Claude carry them out in the MuJoCo viewer.

One Claude conversation runs across all instructions, so follow-ups ("now put it on the pad")
work. Calling done ends only the current instruction. Motions play at real speed; between
turns the world pauses while Claude thinks. The session is recorded like an episode, so it can
be shown, replayed and rendered to video afterwards.
"""
import asyncio
from pathlib import Path
from typing import Any, Callable

from claude_agent_sdk import AssistantMessage, ClaudeSDKClient, ResultMessage

from ..recording import EpisodeRecorder
from ..watch import ViewerOverlay, latest_words, text_rows  # noqa: F401 (text_rows: re-exported)
from ..tasks import Task, make_episode
from ..tiers import Tier
from ..tools import BudgetLimits
from .bridge import ToolBridge
from .prompt import system_prompt
from .session import AgentConfig, MessageLog, claude_options

LIVE_NOTE = ("\n\nThis is an interactive session. A person gives you instructions one at a time. "
             "Carry out each one, then call done to report back: done ends only that instruction, "
             "and the scene stays as it is for the next one. Keep your replies short.")
SESSION_LIMITS = BudgetLimits(max_calls=None, max_sim_seconds=None)


class LiveSession:
    def __init__(self, task: str | Task, tier: Tier, seed: int, out_dir: Path,
                 config: AgentConfig = AgentConfig(),
                 on_message: Callable[[Any], None] | None = None):
        self.episode = make_episode(task, Tier(tier), seed, SESSION_LIMITS)
        self.bridge = ToolBridge(self.episode)
        self.config = config
        self.prompt = system_prompt(self.episode.world.tier) + LIVE_NOTE
        self.recorder = EpisodeRecorder(self.episode, out_dir, agent={
            "kind": "claude-agent-sdk-live", "model": config.model, "effort": config.effort,
            "max_usd": config.max_usd, "cli": config.cli,
            "system_prompt": self.prompt})
        self.log = MessageLog(self.episode, self.bridge, config.model)
        self.overlay: ViewerOverlay | None = None
        self.on_message = on_message
        self.stderr: list[str] = []

    def attach_viewer(self, viewer, realtime: float = 1.0) -> None:
        """Show the session in a passive viewer opened on self.episode.world.plant / .data."""
        self.overlay = ViewerOverlay(viewer, self.episode, realtime)
        self.overlay.draw()

    async def check(self) -> str:
        return (await self._sim(self.episode.evaluate)).summary

    def usage_line(self) -> str:
        budget = self.episode.dispatcher.ctx.budget
        line = f"{budget.calls} tool calls, {budget.sim_seconds:.1f} s sim, ${budget.cost_usd:.4f}"
        if self.log.ledger.unpriced:
            line += f" (not counted, no price known: {', '.join(sorted(self.log.ledger.unpriced))})"
        return line

    async def run(self, next_instruction: Callable[[], Any]) -> Path:
        """`next_instruction()` returns the next instruction (or a coroutine giving one), or None
        to stop. Console commands like ":check" are handled by the caller."""
        error = None
        try:
            options = claude_options(self.bridge, self.config, self.prompt, None,
                                     None, self.stderr)
            async with ClaudeSDKClient(options=options) as client:
                while not (self.overlay and self.overlay.closed):
                    instruction = next_instruction()
                    if asyncio.iscoroutine(instruction):
                        instruction = await instruction
                    if not instruction:
                        break
                    await self.instruct(client, instruction)
        except Exception as exc:  # still write the log, then re-raise
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            await self.bridge.on_sim(self.bridge.end_turn)
            path = await self.bridge.on_sim(self.recorder.finish, {
                "requests": self.log.results, "unmatched_tool_calls": self.bridge.unmatched_calls,
                "error": error, "cli_stderr_tail": self.stderr[-20:]})
            self.bridge.shutdown()
        return path

    async def instruct(self, client: ClaudeSDKClient, instruction: str) -> None:
        """One instruction: send it, follow Claude until it finishes, then reopen for the next."""
        await self._sim(self.episode.world.bus.publish, "/agent/user_message",
                        {"text": instruction}, "agent")
        await self._show(you=instruction, claude="thinking...", doing="", target=None)
        await client.query(instruction)
        async for message in client.receive_response():
            await self.log.handle(message)
            if self.on_message:
                self.on_message(message)
            if isinstance(message, AssistantMessage):
                said = latest_words(message)
                if said:
                    await self._show(claude=said)
            elif isinstance(message, ResultMessage):
                await self._show(claude="done - waiting for your next instruction", doing="")
        # done() ends this instruction only: let the next one use the tools again.
        await self._sim(self.episode.reopen)

    async def _show(self, **fields) -> None:
        if self.overlay:
            await self._sim(self.overlay.set, **fields)

    async def _sim(self, fn, *args, **kwargs):
        return await self.bridge.on_sim(lambda: fn(*args, **kwargs))

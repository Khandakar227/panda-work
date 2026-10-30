"""Run one episode with Claude driving the robot, through the Claude Agent SDK and the local
Claude Code CLI (its own login, no API key needed).

Claude Code gets only the robot tools: its built-in tools, settings, plugins, skills and other
MCP servers are all switched off, and the system prompt replaces Claude Code's own. Everything
is recorded with EpisodeRecorder, so an episode can be inspected, replayed and rendered later.
"""
from pathlib import Path
from typing import Callable

from claude_agent_sdk import ClaudeSDKClient

from ..recording import EpisodeRecorder
from ..tasks import make_episode
from ..tiers import Tier
from ..tools import BudgetLimits
from .bridge import ToolBridge
from .prompt import system_prompt, task_message
from .session import AgentConfig, MessageLog, RateLimited, claude_options


async def run_agent_episode(task: str, tier: Tier, seed: int, out_dir: Path,
                            config: AgentConfig = AgentConfig(), limits: BudgetLimits | None = None,
                            on_message: Callable | None = None,
                            on_episode: Callable | None = None) -> Path:
    """Play one episode; returns the episode folder. `on_message(msg)` sees every SDK message.
    `on_episode(episode)` is called before the first tool call (e.g. to open a viewer); if it
    returns a callable, that runs once the episode is over. Raises RateLimited (after writing
    the log) if the subscription's usage limit cut it short."""
    episode = make_episode(task, Tier(tier), seed, limits)
    after = on_episode(episode) if on_episode else None
    bridge = ToolBridge(episode)
    prompt = system_prompt(episode.world.tier)
    first = task_message(episode.task.instruction)
    recorder = EpisodeRecorder(episode, out_dir, agent={
        "kind": "claude-agent-sdk", "model": config.model, "effort": config.effort,
        "max_usd": config.max_usd, "cli": config.cli,
        "system_prompt": prompt, "first_message": first})
    max_calls = episode.dispatcher.ctx.budget.limits.max_calls or 40
    stderr: list[str] = []
    log = MessageLog(episode, bridge, config.model)
    error = None
    try:
        options = claude_options(bridge, config, prompt, None, max_calls + 10, stderr)
        async with ClaudeSDKClient(options=options) as client:
            await client.query(first)
            async for message in client.receive_response():
                await log.handle(message)
                if on_message:
                    on_message(message)
        if log.rate_limited:
            raise RateLimited(log.rate_limited)
    except Exception as exc:  # still write the log, then re-raise
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        await bridge.on_sim(bridge.end_turn)
        path = await bridge.on_sim(recorder.finish, {
            "sdk_result": log.results[-1] if log.results else {},
            "unmatched_tool_calls": bridge.unmatched_calls, "error": error,
            "cli_stderr_tail": stderr[-20:]})
        bridge.shutdown()
        if after:
            after()
    return path

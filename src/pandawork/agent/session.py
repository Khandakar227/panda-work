"""Pieces shared by single episodes (runner.py) and interactive sessions (live.py): how Claude
Code is configured, and how each message from it is logged and paid for."""
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage, ClaudeAgentOptions, HookMatcher, RateLimitEvent, ResultMessage, SystemMessage,
)

from ..recording import CostLedger, Usage
from ..tasks import Episode
from ..tools.rendering import to_json
from .bridge import SERVER, ToolBridge, assistant_event

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_MESSAGE_BYTES = 32 * 1024 * 1024  # SDK stdout line buffer; images travel inside one line


@dataclass(frozen=True)
class AgentConfig:
    model: str = DEFAULT_MODEL
    effort: str = "medium"       # low | medium | high | xhigh | max
    max_usd: float = 1.0         # hard stop, enforced by Claude Code
    cli_path: str | None = None  # default: the `claude` on PATH

    @property
    def cli(self) -> str | None:
        return self.cli_path or shutil.which("claude")


def neutral_cwd() -> Path:
    """An empty directory with a bland name for the agent's working directory.

    Claude Code tells the model its working directory, so the episode folder must never be used:
    its name carries the task, the tier and the seed, and agents have been seen quoting their
    working directory; an experiment's condition must not reach the agent that way."""
    cwd = Path(tempfile.gettempdir()) / "robot-session"
    cwd.mkdir(exist_ok=True)
    return cwd


def claude_options(bridge: ToolBridge, config: AgentConfig, prompt: str, cwd: Path | None,
                   max_turns: int | None, stderr: list[str]) -> ClaudeAgentOptions:
    """Claude Code with nothing but the robot tools and our system prompt. `cwd` is ignored
    unless it is already neutral; see neutral_cwd()."""
    cwd = neutral_cwd()
    return ClaudeAgentOptions(
        model=config.model, effort=config.effort,
        thinking={"type": "adaptive", "display": "summarized"},
        system_prompt=prompt,
        tools=[],                                   # no Bash, Read, Edit, ...: robot tools only
        mcp_servers={SERVER: bridge.mcp_server()}, strict_mcp_config=True,
        allowed_tools=bridge.allowed_tools, permission_mode="dontAsk",
        hooks={"PreToolUse": [HookMatcher(hooks=[bridge.pre_tool_use])]},
        setting_sources=[],                         # no user/project settings, CLAUDE.md, plugins
        cwd=str(cwd), max_turns=max_turns, max_budget_usd=config.max_usd,
        cli_path=config.cli, stderr=stderr.append,
        # One 640x480 image from a noisy (realistic) camera is a ~500 KB PNG; the SDK's default
        # 1 MB line buffer then fails the episode ("JSON message exceeded maximum buffer size").
        max_buffer_size=MAX_MESSAGE_BYTES,
    )


class RateLimited(Exception):
    """The Claude subscription's usage limit was hit during the episode. The episode is not a
    measurement of the agent: callers set it aside and re-run it after the reset."""

    def __init__(self, info: dict[str, Any]):
        self.info = info
        self.resets_at: float | None = info.get("resetsAt")
        when = time.strftime("%H:%M", time.localtime(self.resets_at)) if self.resets_at else "?"
        super().__init__(f"usage limit hit ({info.get('rateLimitType') or 'unknown window'}); "
                         f"resets at {when}")


class MessageLog:
    """Logs every SDK message on the episode's bus and records each request's usage."""

    def __init__(self, episode: Episode, bridge: ToolBridge, model: str):
        self.episode = episode
        self.bridge = bridge
        self.model = model
        self.ledger = CostLedger(episode.world.bus, episode.dispatcher.ctx.budget)
        self.results: list[dict[str, Any]] = []  # one per completed request
        self.rate_limited: dict[str, Any] | None = None  # set when the limit rejected a request
        self._usage_so_far: dict[str, tuple[Usage, float]] = {}  # model_usage is cumulative per session

    async def handle(self, message: Any) -> None:
        self.bridge.observe(message)
        publish = self.episode.world.bus.publish
        if isinstance(message, SystemMessage) and message.subtype == "init":
            await self.bridge.on_sim(publish, "/agent/session_init", {
                "session_id": str(message.data.get("session_id", "")),
                "model": str(message.data.get("model", "")),
                "tools": to_json(message.data.get("tools", [])),
                "mcp_servers": to_json(message.data.get("mcp_servers", []))}, "agent")
        elif isinstance(message, AssistantMessage):
            await self.bridge.on_sim(publish, "/agent/assistant", assistant_event(message), "agent")
        elif isinstance(message, RateLimitEvent):
            info = message.rate_limit_info
            await self.bridge.on_sim(publish, "/agent/rate_limit", {
                "status": info.status, "window": info.rate_limit_type or "",
                "resets_at": float(info.resets_at or 0), "utilization": float(info.utilization or 0.0),
            }, "agent")
            if info.status == "rejected":
                self.rate_limited = dict(info.raw)
        elif isinstance(message, ResultMessage):
            self.results.append({
                "subtype": message.subtype, "num_turns": message.num_turns,
                "stop_reason": message.stop_reason, "is_error": message.is_error,
                "total_cost_usd": message.total_cost_usd, "usage": message.usage,
                "duration_ms": message.duration_ms, "duration_api_ms": message.duration_api_ms,
                "session_id": message.session_id, "errors": message.errors,
                "model_usage": message.model_usage})
            if message.usage:
                await self.bridge.on_sim(record_usage, self.ledger, self.model, message,
                                         self._usage_so_far)


def usage_from(u: dict) -> Usage:
    """Claude Code's usage totals -> Usage, keeping the 1-hour cache writes apart (priced 2x)."""
    return Usage(input_tokens=int(u.get("input_tokens") or 0),
                 output_tokens=int(u.get("output_tokens") or 0),
                 cache_creation_input_tokens=int(u.get("cache_creation_input_tokens") or 0),
                 cache_read_input_tokens=int(u.get("cache_read_input_tokens") or 0),
                 cache_creation_1h_input_tokens=int(
                     (u.get("cache_creation") or {}).get("ephemeral_1h_input_tokens") or 0))


def record_usage(ledger: CostLedger, model: str, message: ResultMessage,
                 seen: dict[str, tuple[Usage, float]] | None = None) -> None:
    """One request's model calls. model_usage lists every model Claude Code used (the agent
    model and helpers it calls on its own, e.g. a small Haiku call) with tokens and costUSD,
    cumulative over the session, so each model's increase since the previous request is what
    this request used. The agent model's tokens come from message.usage instead (per request,
    with the 1-hour cache writes broken out); its cost from model_usage where reported."""
    seen = {} if seen is None else seen
    used: dict[str, tuple[Usage, float | None]] = {}
    for name, mu in (message.model_usage or {}).items():
        canonical = mu.get("canonicalModel") or name
        total = Usage(input_tokens=int(mu.get("inputTokens") or 0),
                      output_tokens=int(mu.get("outputTokens") or 0),
                      cache_creation_input_tokens=int(mu.get("cacheCreationInputTokens") or 0),
                      cache_read_input_tokens=int(mu.get("cacheReadInputTokens") or 0))
        reported = mu.get("costUSD")
        before, cost_before = seen.get(canonical, (Usage(), 0.0))
        delta = Usage(**{f: getattr(total, f) - getattr(before, f) for f in (
            "input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")})
        seen[canonical] = (total, float(reported) if reported is not None else cost_before)
        used[canonical] = (delta, None if reported is None else float(reported) - cost_before)
    _, agent_cost = used.pop(model, (None, None))
    ledger.record(model, usage_from(message.usage), request_id=message.session_id,
                  latency_s=message.duration_api_ms / 1000, reported_usd=agent_cost)
    for helper, (delta, cost) in used.items():
        if delta.total > 0 or (cost or 0.0) > 0:
            ledger.record(helper, delta, request_id=f"{message.session_id}:{helper}",
                          reported_usd=cost)

"""Load episode logs, summarise them, and replay them to check they reproduce exactly.

Replay rebuilds the episode from (task, tier, seed, budget limits) and re-issues the logged tool
calls turn by turn. No LLM is involved: the simulation is deterministic, so the same calls must
lead to the same states. Each turn's state and each tool's model-visible text is compared with
the log.
"""
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from ..tasks import make_episode
from ..tiers import Tier
from ..tools import BudgetLimits, ToolCall
from .recorder import world_state


@dataclass
class EpisodeLog:
    dir: Path
    meta: dict
    events: list[dict]
    snapshots: dict[str, np.ndarray]
    result: dict
    streams: Any  # numpy NpzFile, loaded lazily

    def events_on(self, topic: str) -> list[dict]:
        return [e for e in self.events if e["topic"] == topic]

    def blob(self, ref: dict) -> np.ndarray:
        path = self.dir / ref["$blob"]
        return np.asarray(Image.open(path)) if path.suffix == ".png" else np.load(path)

    def turns(self) -> list[list[ToolCall]]:
        """The logged tool calls, grouped into the turns they were issued in."""
        by_turn: dict[int, list[ToolCall]] = defaultdict(list)
        for e in self.events_on("/agent/tool_call"):
            d = e["data"]
            by_turn[d["turn"]].append(ToolCall(d["call_id"], d["tool"], json.loads(d["args"])))
        return [by_turn[t] for t in sorted(by_turn)]

    def instruction_starts(self) -> set[int]:
        """Turns that began right after a new user instruction (interactive sessions)."""
        starts, pending = set(), False
        for e in self.events:
            if e["topic"] == "/agent/user_message":
                pending = True
            elif e["topic"] == "/agent/tool_call" and pending:
                starts.add(e["data"]["turn"])
                pending = False
        return starts


def load(directory: Path) -> EpisodeLog:
    directory = Path(directory)
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    snapshots = dict(np.load(directory / "snapshots.npz"))
    return EpisodeLog(directory, json.loads((directory / "meta.json").read_text()), events,
                      snapshots, json.loads((directory / "result.json").read_text()),
                      np.load(directory / "streams.npz"))


def summarise(log: EpisodeLog) -> str:
    """A readable timeline: instruction, every turn's calls and results, verdict, usage."""
    m, r = log.meta, log.result
    lines = [f"{m['task']} | tier {m['tier']} | seed {m['seed']} | harness {m['harness_version']}",
             f"instruction: {m['instruction']}", ""]
    results = {e["data"]["call_id"]: e for e in log.events_on("/agent/tool_result")}
    calls = log.events_on("/agent/tool_call")
    for e in calls:
        res = results.get(e["data"]["call_id"])
        mark = "ok" if res and res["data"]["ok"] else "x"
        first = res["data"]["text"].splitlines()[0] if res else "(no result)"
        lines.append(f"  t={e['sim_time']:6.2f}s  turn {e['data']['turn']:>2}  "
                     f"{e['data']['tool']}({_short(e['data']['args'])}) [{mark}] -> {first}")
    usage = r["usage"]
    lines += ["", f"verdict: {'SUCCESS' if r['success'] else 'FAILURE'} - {r['summary']}",
              f"agent claimed success: {r['agent_claimed_success']}",
              f"usage: {usage['turns']} turns, {usage['tool_calls']} tool calls, "
              f"{usage['sim_seconds']:.1f} s sim, {usage['tokens']} tokens, ${usage['cost_usd']:.4f}"]
    return "\n".join(lines)


def _short(args_json: str, limit: int = 60) -> str:
    s = args_json.strip("{}")
    return s if len(s) <= limit else s[:limit - 3] + "..."


@dataclass
class ReplayReport:
    identical: bool
    turns: int
    divergence: str | None = None  # first difference found
    max_state_diff: float = 0.0
    final_state_identical: bool = False
    text_mismatches: list[str] = field(default_factory=list)


def replay(log: EpisodeLog, on_world: Callable | None = None,
           on_episode: Callable | None = None) -> ReplayReport:
    """Re-run the logged tool calls and compare against the log. `on_world(world)` is called
    once before replaying, e.g. to attach a video recorder hook; `on_episode(episode)` likewise,
    and may return a callable that is run on the finished episode before its world is closed."""
    meta = log.meta
    ep = make_episode(meta["task"], Tier(meta["tier"]), meta["seed"],
                      BudgetLimits(**meta["budget_limits"]))
    if on_world:
        on_world(ep.world)
    at_end = on_episode(ep) if on_episode else None
    report = ReplayReport(identical=True, turns=0)
    if not np.array_equal(world_state(ep.world), log.snapshots["start"]):
        report.identical, report.divergence = False, "start state differs (code or seed changed?)"
    # Results are compared in order: call ids need not be unique across turns.
    logged_texts = iter(e["data"]["text"] for e in log.events_on("/agent/tool_result"))
    starts = log.instruction_starts()
    for turn, calls in enumerate(log.turns()):
        if turn in starts:
            ep.reopen()
        blocks = ep.dispatcher.run_batch(calls)
        report.turns += 1
        for call, block in zip(calls, blocks):
            text = "\n".join(c["text"] for c in block["content"] if c["type"] == "text"
                             and not c["text"].startswith("[t="))
            if text != next(logged_texts, None):
                report.text_mismatches.append(f"turn {turn} {call.id} ({call.name})")
        logged = log.snapshots.get(f"turn{turn}")
        if logged is not None:
            diff = float(np.abs(world_state(ep.world) - logged).max())
            report.max_state_diff = max(report.max_state_diff, diff)
            if diff > 0 and report.divergence is None:
                report.identical, report.divergence = False, f"state differs after turn {turn} (max {diff:.3g})"
    final = world_state(ep.world)
    report.final_state_identical = bool(np.array_equal(final, log.snapshots["end"]))
    if report.text_mismatches and report.divergence is None:
        report.divergence = f"tool output differs for calls {report.text_mismatches[:5]}"
    report.identical = report.identical and report.final_state_identical and not report.text_mismatches
    if callable(at_end):
        at_end(ep)
    ep.world.close()
    return report

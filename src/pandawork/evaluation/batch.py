"""Run a grid of episodes (tasks x tiers x seeds), resumably and under a total spending cap.

Each episode gets its own folder <run>/<task>-<tier>[-f_<fault>][-warn|-warnmild]-s<seed>/. A finished
episode (with result.json) is never re-run, so an interrupted batch continues where it stopped.
Episodes run grouped by tier and prompt variant: the system prompt and tool list stay in Claude
Code's 1-hour prompt cache between consecutive episodes, which lowers cost. When the Claude
subscription's usage limit is hit mid-episode, that episode is set aside, the batch sleeps until
the limit resets, and the episode is re-run.
"""
import asyncio
import json
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterator

from ..agent.prompt import warn_level
from ..agent import AgentConfig, run_agent_episode
from ..agent.session import RateLimited
from ..core.faults import FAULTS
from ..recording import EpisodeRecorder
from ..tasks import make_episode
from ..tiers import Tier
from ..tools import BudgetLimits

AGENTS = ("claude", "oracle")
# The keypad start posture (elbow at -145 deg) lies outside the stiff elbow's range, and the
# drawer handle and two of the three stack cubes lie closer to the base than the stiff elbow can
# reach (grid run 1: 0/9 each in every tier, oracle too), so those cells would measure reach,
# not self-knowledge. stiff_j4 is run on pick_place and press_button only.
EXCLUDED_CELLS = {("press_key", "stiff_j4"), ("type_code", "stiff_j4"),
                  ("open_drawer", "stiff_j4"), ("stack", "stiff_j4")}
RESET_MARGIN_S = 60.0  # wait this much past the reported reset before resuming


@dataclass(frozen=True)
class Job:
    task: str
    tier: Tier
    seed: int
    fault: str = "none"   # a core.faults preset name
    warn: str = "none"    # prompt warning arm: "none" | "mild" | "full" (see agent.prompt.WARNINGS)

    def __post_init__(self):
        object.__setattr__(self, "warn", warn_level(self.warn))

    @property
    def name(self) -> str:
        return (f"{self.task}-{self.tier.value}" + (f"-f_{self.fault}" if self.fault != "none" else "")
                + WARN_SUFFIX[self.warn] + f"-s{self.seed}")


WARN_SUFFIX = {"none": "", "full": "-warn", "mild": "-warnmild"}


def grid(tasks: list[str], tiers: list[Tier], seeds: list[int], faults=("none",),
         warn=("none",)) -> list[Job]:
    """Tier-major order (cache stays warm), then prompt variant, fault, task, seed."""
    return [Job(t, tier, s, f, w) for tier in tiers for w in warn for f in faults for t in tasks
            for s in seeds if (t, f) not in EXCLUDED_CELLS]


def set_aside(folder: Path) -> bool:
    """Folders with a dot in the name were set aside (`.incomplete-<time>`, or renamed by hand,
    e.g. `.bad`): resume, cost, reports and analysis all ignore them. Job names have no dots."""
    return "." in folder.name


def spent_usd(run_dir: Path) -> float:
    total = 0.0
    for result in run_dir.glob("*/result.json"):
        if not set_aside(result.parent):
            total += json.loads(result.read_text())["usage"]["cost_usd"]
    return total


def pending(run_dir: Path, jobs: list[Job]) -> Iterator[Job]:
    for job in jobs:
        folder = run_dir / job.name
        if (folder / "result.json").exists():
            continue
        if folder.exists():  # an episode that crashed or was interrupted: keep it aside, re-run
            folder.rename(folder.with_name(f"{folder.name}.incomplete-{time.strftime('%H%M%S')}"))
        yield job


def run_oracle(job: Job, folder: Path, limits: BudgetLimits | None = None,
               on_episode: Callable | None = None) -> Path:
    episode = make_episode(job.task, job.tier, job.seed, limits, fault=FAULTS[job.fault])
    after = on_episode(episode) if on_episode else None
    recorder = EpisodeRecorder(episode, folder, agent={"kind": "oracle"})
    try:
        episode.task.oracle(episode.dispatcher)
    finally:
        path = recorder.finish()
        if after:
            after()
    episode.world.close()
    return path


def run_grid(run_dir: Path, jobs: list[Job], agent: str = "claude",
             config: AgentConfig = AgentConfig(), max_total_usd: float = 8.0,
             report: Callable[[str], None] = print,
             sleep: Callable[[float], None] = time.sleep,
             limits: BudgetLimits | None = None, on_episode: Callable | None = None) -> list[Path]:
    """Run every unfinished job; stops early once the run's total cost reaches max_total_usd.
    `limits`: the per-episode budget (default BudgetLimits(): 40 calls). `on_episode(episode)`
    runs as each episode starts (a `watch.Watch` opens the viewer); its return value, if
    callable, runs when that episode ends."""
    if agent not in AGENTS:
        raise ValueError(f"agent must be one of {AGENTS}")
    limits = limits or BudgetLimits()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "batch.json").write_text(json.dumps({
        "agent": agent, "model": config.model, "effort": config.effort,
        "max_usd_per_episode": config.max_usd, "max_total_usd": max_total_usd,
        "budget_limits": {k: getattr(limits, k) for k in limits.__dataclass_fields__},
        "faults": sorted({j.fault for j in jobs}, key=[j.fault for j in jobs].index),
        "warn": sorted({j.warn for j in jobs}, key=["none", "mild", "full"].index),
        "jobs": [j.name for j in jobs]}, indent=2))
    done = []
    todo = list(pending(run_dir, jobs))
    report(f"{len(jobs) - len(todo)} of {len(jobs)} episodes already finished; {len(todo)} to run")
    for name in dict.fromkeys(j.fault for j in todo):
        if name != "none":
            report(f"fault {name} ({FAULTS[name].visibility}): {FAULTS[name].describe()}")
    for i, job in enumerate(todo, 1):
        spent = spent_usd(run_dir)
        if spent >= max_total_usd:
            report(f"stopping: spent ${spent:.2f} of the ${max_total_usd:.2f} cap")
            break
        folder = run_dir / job.name
        tic = time.monotonic()
        try:
            if agent == "oracle":
                path = run_oracle(job, folder, limits, on_episode)
            else:
                path = asyncio.run(run_agent_episode(job.task, job.tier, job.seed, folder,
                                                     replace(config, warn=job.warn), limits,
                                                     fault=FAULTS[job.fault], on_episode=on_episode))
        except RateLimited as exc:  # not the agent's doing: set aside, wait, re-run
            if folder.exists():
                folder.rename(folder.with_name(f"{folder.name}.incomplete-ratelimit-{time.strftime('%H%M%S')}"))
            wait = max(0.0, (exc.resets_at or time.time() + 300.0) - time.time()) + RESET_MARGIN_S
            report(f"[{i}/{len(todo)}] {job.name}: {exc}; waiting {wait / 60:.0f} min")
            sleep(wait)
            todo.append(job)
            continue
        except Exception as exc:  # one bad episode must not end the batch
            report(f"[{i}/{len(todo)}] {job.name}: ERROR {type(exc).__name__}: {exc}")
            continue
        r = json.loads((path / "result.json").read_text())
        done.append(path)
        report(f"[{i}/{len(todo)}] {job.name:<34} {'SUCCESS' if r['success'] else 'failure':<8} "
               f"{r['usage']['tool_calls']:>3} calls  ${r['usage']['cost_usd']:.3f}  "
               f"{time.monotonic() - tic:5.1f}s  (total ${spent_usd(run_dir):.2f})")
    return done


def clear_incomplete(run_dir: Path) -> int:
    """Delete folders set aside by pending() (interrupted episodes)."""
    folders = list(run_dir.glob("*.incomplete-*"))
    for f in folders:
        shutil.rmtree(f)
    return len(folders)

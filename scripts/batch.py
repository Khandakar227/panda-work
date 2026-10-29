"""Run a batch of episodes over tasks x tiers x seeds, and report on it.

  python scripts/batch.py run demo --agent claude --seeds 0-9 --max-total-usd 8
  python scripts/batch.py run oracle-check --agent oracle --seeds 0-4    # free pipeline check
  python scripts/batch.py report demo
  python scripts/batch.py run faults --faults none encoder_j1_5deg --tasks pick_place --seeds 0-4
  python scripts/batch.py run faults-study --faults none encoder_j1_5deg --max-calls 60 --warn both
  python scripts/batch.py run demo ... --watch          # every episode in the viewer, fault on screen

Runs are resumable: re-running the same command skips finished episodes. Episodes live in
runs/batch/<name>/<task>-<tier>-s<seed>/ and can be inspected with scripts/episodes.py.
"""
import argparse
from pathlib import Path

from pandawork.agent import DEFAULT_MODEL, AgentConfig
from pandawork.core.faults import FAULTS
from pandawork.core.robot import PACKAGE_ROOT
from pandawork.core.viewer import exit_after
from pandawork.evaluation import AGENTS, grid, run_grid, write_report
from pandawork.tasks import TASKS
from pandawork.tiers import Tier
from pandawork.tools import BudgetLimits
from pandawork.watch import Watch

ROOT = PACKAGE_ROOT / "runs" / "batch"


def seeds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def parse_exclude(specs: list[str]) -> set[tuple[str, str]]:
    """`task:fault` pairs for --exclude; unknown names or another shape are an error."""
    out = set()
    for spec in specs:
        task, sep, fault = spec.partition(":")
        if not sep or task not in TASKS or fault not in FAULTS:
            raise argparse.ArgumentTypeError(
                f"--exclude expects TASK:FAULT with a known task and fault, got {spec!r}")
        out.add((task, fault))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("name")
    run.add_argument("--agent", choices=AGENTS, default="claude")
    run.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    run.add_argument("--tiers", nargs="+", default=[t.value for t in Tier], choices=[t.value for t in Tier])
    run.add_argument("--seeds", default="0-9", help="e.g. 0-9 or 0,3,5-7")
    run.add_argument("--faults", nargs="+", default=["none"], choices=list(FAULTS),
                     help="fault presets to cross with the grid (core.faults)")
    run.add_argument("--exclude", nargs="*", default=[], metavar="TASK:FAULT",
                     help="skip these task x fault cells, e.g. stack:stiff_j4")
    run.add_argument("--warn", choices=["no", "mild", "yes", "both", "all"], default="no",
                     help="tell the agent the robot may be miscalibrated")
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--effort", default="medium")
    run.add_argument("--max-usd", type=float, default=1.0, help="hard stop per episode")
    run.add_argument("--max-total-usd", type=float, default=8.0, help="stop the batch at this total")
    run.add_argument("--max-calls", type=int, default=BudgetLimits.max_calls,
                     help="tool calls per episode (faulty episodes need more than clean ones)")
    rep = sub.add_parser("report")
    run.add_argument("--watch", action="store_true", help="show each episode in the MuJoCo viewer")
    run.add_argument("--realtime", type=float, default=1.0,
                     help="--watch speed (x real time; 0: as fast as the simulation runs)")
    rep.add_argument("name")
    args = parser.parse_args()

    run_dir = ROOT / args.name
    if args.cmd == "run":
        warn = {"no": ("none",), "mild": ("mild",), "yes": ("full",), "both": ("none", "full"),
                "all": ("none", "mild", "full")}[args.warn]
        jobs = grid(args.tasks, [Tier(t) for t in args.tiers], seeds(args.seeds), args.faults, warn,
                    exclude=parse_exclude(args.exclude))
        config = AgentConfig(model=args.model, effort=args.effort, max_usd=args.max_usd)
        watch = Watch(args.realtime) if args.watch else None
        run_grid(run_dir, jobs, args.agent, config, args.max_total_usd, report=lambda s: print(s, flush=True),
                 limits=BudgetLimits(max_calls=args.max_calls), on_episode=watch)
    path, text = write_report(run_dir)
    print("\n" + text)
    print(f"report written to {path.relative_to(PACKAGE_ROOT)}")


if __name__ == "__main__":
    exit_after(main)  # see pandawork.core.viewer

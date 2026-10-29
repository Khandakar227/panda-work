"""Run an episode with Claude (via the local Claude Code CLI and the Claude Agent SDK).

  python scripts/run_agent.py reach --seed 0                     # Sonnet 5.5, privileged tier
  python scripts/run_agent.py pick_place --tier realistic --seed 3 --effort high
  python scripts/run_agent.py press_button --max-usd 0.5 --quiet
  python scripts/run_agent.py pick_place --tier realistic --fault encoder_j1_5deg --watch   # in the viewer

Prints Claude's reasoning summaries, tool calls and results live, then the verdict and cost.
The episode is recorded under runs/agent/; inspect it with scripts/episodes.py (show, replay,
video).
"""
import argparse
import asyncio
import time
from pathlib import Path

from pandawork.agent import DEFAULT_MODEL, AgentConfig, run_agent_episode
from pandawork.core.faults import FAULTS
from pandawork.agent.printing import console_printer
from pandawork.core.robot import PACKAGE_ROOT
from pandawork.core.viewer import exit_after
from pandawork.recording import summarise, load
from pandawork.tasks import TASKS
from pandawork.tiers import Tier
from pandawork.tools import BudgetLimits
from pandawork.watch import Watch


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task", choices=list(TASKS))
    parser.add_argument("--tier", default="privileged", choices=[t.value for t in Tier])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--max-usd", type=float, default=1.0, help="hard cost stop per episode")
    parser.add_argument("--fault", default="none", choices=list(FAULTS), help="a core.faults preset")
    parser.add_argument("--warn", nargs="?", const="full", default="none", choices=["none", "mild", "full"],
                        help="prompt warning arm: mild (the robot is not ideal) or full (may be miscalibrated; verify)")
    parser.add_argument("--out", help="episode folder (default runs/agent/<task>-<tier>-s<seed>-<time>)")
    parser.add_argument("--max-calls", type=int, default=BudgetLimits.max_calls, help="tool calls per episode")
    parser.add_argument("--watch", action="store_true", help="show the episode in the MuJoCo viewer "
                        "(fault, believed vs true fingertip, Claude's words on screen)")
    parser.add_argument("--realtime", type=float, default=1.0, help="--watch speed (x real time; 0: no pacing)")
    parser.add_argument("--hold", action="store_true", help="--watch: keep the window open at the end")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    tier = Tier(args.tier)
    out = Path(args.out or PACKAGE_ROOT / "runs" / "agent" /
               f"{args.task}-{tier.value}-s{args.seed}-{time.strftime('%Y%m%d-%H%M%S')}")
    config = AgentConfig(model=args.model, effort=args.effort, max_usd=args.max_usd, warn=args.warn)
    if args.fault != "none":
        print(f"Fault: {args.fault} ({FAULTS[args.fault].visibility}): {FAULTS[args.fault].describe()}")
    printer = console_printer(args.quiet)
    watch = Watch(args.realtime, hold=args.hold) if args.watch else None

    def on_message(message):
        printer(message)
        if watch:
            watch.on_message(message)

    path = asyncio.run(run_agent_episode(args.task, tier, args.seed, out, config,
                                         BudgetLimits(max_calls=args.max_calls),
                                         on_message=on_message, fault=FAULTS[args.fault],
                                         on_episode=watch))
    log = load(path)
    print("\n" + summarise(log))
    sdk = log.result.get("sdk_result", {})
    print(f"\nCLI-reported cost: ${sdk.get('total_cost_usd') or 0:.4f} | "
          f"harness ledger: ${log.result['usage']['cost_usd']:.4f} | "
          f"unpaired tool calls: {log.result.get('unmatched_tool_calls')}")
    print(f"episode folder: {path}")


if __name__ == "__main__":
    exit_after(main)  # see pandawork.core.viewer

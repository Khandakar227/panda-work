"""Live console: type instructions and watch Claude carry them out in the MuJoCo viewer.

  python scripts/live.py                                   # playground scene, privileged tier
  python scripts/live.py --task pick_place --seed 3 --tier realistic
  python scripts/live.py --panels                          # also show MuJoCo's side panels
  python scripts/live.py --no-viewer --say "pick up the red cube" --say "put it on the green pad"

At the prompt, type an instruction, or:
  :check   the task's success check (task scenes only)
  :cost    tool calls, simulated time and cost so far
  :quit    end the session (also Ctrl-D, or closing the viewer)
In the viewer: Tab / Shift+Tab show or hide the side panels; drag to orbit, scroll to zoom.
The session is recorded under runs/live/; inspect it with scripts/episodes.py.
"""
import argparse
import asyncio
import time
from pathlib import Path

from pandawork.agent import DEFAULT_MODEL, AgentConfig
from pandawork.agent.live import LiveSession
from pandawork.agent.printing import console_printer
from pandawork.core.robot import PACKAGE_ROOT
from pandawork.core.viewer import exit_after
from pandawork.tasks import TASKS, PlaygroundTask
from pandawork.tiers import Tier


def make_reader(session: LiveSession, scripted: list[str]):
    """Returns next_instruction(): scripted instructions first, else the terminal."""
    queue = list(scripted)

    async def next_instruction():
        if scripted:
            if not queue:
                return None
            line = queue.pop(0)
            print(f"\nyou> {line}")
            return line
        while True:
            try:
                line = (await asyncio.to_thread(input, "\nyou> ")).strip()
            except (EOFError, KeyboardInterrupt):
                return None
            if line in (":quit", ":q"):
                return None
            if line == ":check":
                print("  " + await session.check())
            elif line == ":cost":
                print("  " + session.usage_line())
            elif line:
                return line
    return next_instruction


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=["playground", *TASKS], default="playground")
    parser.add_argument("--tier", default="privileged", choices=[t.value for t in Tier])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--max-usd", type=float, default=2.0, help="hard cost stop for the session")
    parser.add_argument("--say", action="append", default=[], help="scripted instruction (repeatable)")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--panels", action="store_true", help="show MuJoCo's side panels")
    parser.add_argument("--realtime", type=float, default=1.0, help="motion speed (x real time)")
    parser.add_argument("--quiet", action="store_true", help="less terminal output")
    args = parser.parse_args()

    tier = Tier(args.tier)
    task = PlaygroundTask() if args.task == "playground" else args.task
    out = PACKAGE_ROOT / "runs" / "live" / f"{args.task}-{tier.value}-s{args.seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    session = LiveSession(task, tier, args.seed, out,
                          AgentConfig(model=args.model, effort=args.effort, max_usd=args.max_usd),
                          on_message=console_printer(args.quiet))
    print(f"Scene: {args.task} (seed {args.seed}, tier {tier.value}). Objects: "
          f"{', '.join(session.episode.world.objects)}")
    if args.task != "playground":
        print(f"Suggested instruction: {session.episode.task.instruction}")
    reader = make_reader(session, args.say)

    if args.no_viewer:
        path = asyncio.run(session.run(reader))
    else:
        from pandawork.watch import open_viewer
        with open_viewer(session.episode, args.panels) as viewer:
            session.attach_viewer(viewer, args.realtime)
            path = asyncio.run(session.run(reader))

    print(f"\nSession: {session.usage_line()}")
    print(f"Recorded in {Path(path).relative_to(PACKAGE_ROOT)}")
    print(f"  replay: python scripts/episodes.py replay {Path(path).relative_to(PACKAGE_ROOT)}")
    print(f"  video:  python scripts/episodes.py video {Path(path).relative_to(PACKAGE_ROOT)} --camera side")


if __name__ == "__main__":
    exit_after(main)  # see pandawork.core.viewer

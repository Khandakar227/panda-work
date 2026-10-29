"""The fault presets: what each one does to the body, and a way to see it before an experiment.

  python scripts/faults.py                                  # list every preset with a description
  python scripts/faults.py --watch encoder_j1_5deg          # the oracle on pick_place under that fault, in the viewer
  python scripts/faults.py --watch fingertip_2cm_long --task stack --seed 2 --realtime 2

In the viewer the panel names the fault and shows how far the true fingertip is from where the
robot believes it is; a green sphere marks the believed fingertip, a red one the true fingertip.
The oracle trusts the tools, so under a hidden fault you watch it miss. Nothing is recorded.
"""
import argparse

from pandawork.core.faults import FAULTS
from pandawork.core.viewer import exit_after
from pandawork.tasks import TASKS, make_episode
from pandawork.watch import Watch


def list_faults() -> None:
    print(f"{'preset':<20}{'kind':<16}{'visibility':<11}description")
    for fault in FAULTS.values():
        print(f"{fault.name:<20}{fault.kind:<16}{fault.visibility:<11}{fault.describe()}")


def watch_oracle(fault: str, task: str, seed: int, realtime: float) -> None:
    episode = make_episode(task, seed=seed, fault=FAULTS[fault])
    print(f"{task} seed {seed}, fault {fault}: {FAULTS[fault].describe()}")
    print(f"Instruction: {episode.task.instruction}")
    watch = Watch(realtime, hold=True)
    close = watch(episode)
    watch.overlay.set(claude="oracle (scripted agent that trusts the tools)")
    try:
        episode.task.oracle(episode.dispatcher)
    finally:
        summary = episode.evaluate().summary
        print(summary)
        watch.overlay.set(claude=f"oracle finished: {summary}")
        close()
        episode.world.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--watch", choices=list(FAULTS), help="watch the oracle under this fault")
    parser.add_argument("--task", choices=list(TASKS), default="pick_place")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--realtime", type=float, default=1.0, help="speed (x real time; 0: no pacing)")
    args = parser.parse_args()
    if args.watch:
        watch_oracle(args.watch, args.task, args.seed, args.realtime)
    else:
        list_faults()


if __name__ == "__main__":
    exit_after(main)  # see pandawork.core.viewer

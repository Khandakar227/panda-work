"""Run each task's scripted oracle over many seeds: proves the tasks are solvable with the tools.

  python scripts/oracle_eval.py                      # 50 seeds per task, all tasks
  python scripts/oracle_eval.py --tasks pick_place --seeds 10
  python scripts/oracle_eval.py --watch open_drawer --seed 3   # one episode in the viewer
  python scripts/oracle_eval.py --watch type_code --seed 0 --code 2468   # your own code

Exits with status 1 if any task's success rate is below --require (default 95%).
"""
import argparse
import sys
import time

import numpy as np

from pandawork.core.viewer import exit_after
from pandawork.tasks import TASKS, make_episode
from pandawork.tasks.library import TypeCodeTask


def task_for(name: str, code: str | None):
    """The task to run: type_code with a chosen code instead of the seed's, if one is given."""
    return TypeCodeTask(code) if name == "type_code" and code else name


def evaluate(task: str, seeds: int, code: str | None = None) -> dict:
    results, sim, calls, wall = [], [], [], []
    for seed in range(seeds):
        tic = time.perf_counter()
        ep = make_episode(task_for(task, code), seed=seed)
        ep.task.oracle(ep.dispatcher)
        ev = ep.evaluate()
        results.append((seed, ev))
        sim.append(ep.world.clock.sim_time)
        calls.append(ep.dispatcher.ctx.budget.calls)
        wall.append(time.perf_counter() - tic)
        ep.world.close()
    return {"results": results, "sim": np.mean(sim), "calls": np.mean(calls), "wall": np.sum(wall)}


def watch(task: str, seed: int, code: str | None = None) -> None:
    """One oracle episode in the viewer, with the panel of pandawork.watch."""
    from pandawork.watch import Watch
    ep = make_episode(task_for(task, code), seed=seed)
    print(f"{task} seed {seed}: {ep.task.instruction}")
    watcher = Watch(realtime=1.0, hold=True)
    close = watcher(ep)
    watcher.overlay.set(claude="oracle")
    try:
        ep.task.oracle(ep.dispatcher)
    finally:
        summary = ep.evaluate().summary
        print(summary)
        watcher.overlay.set(claude=f"oracle finished: {summary}  [close the window to exit]")
        close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", nargs="*", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--require", type=float, default=0.95)
    parser.add_argument("--watch", choices=list(TASKS), help="watch one oracle episode")
    parser.add_argument("--seed", type=int, default=0, help="seed for --watch")
    parser.add_argument("--code", help="type_code only: type this code (digits 1-9) instead of "
                        "the one drawn from the seed")
    args = parser.parse_args()
    if args.code:
        try:
            TypeCodeTask(args.code)
        except ValueError as exc:
            parser.error(str(exc))

    if args.watch:
        watch(args.watch, args.seed, args.code)
        return

    print(f"{'task':<14}{'success':>10}{'sim s':>8}{'calls':>7}{'wall s':>8}")
    ok = True
    for task in args.tasks:
        r = evaluate(task, args.seeds, args.code)
        wins = sum(ev.success for _, ev in r["results"])
        rate = wins / args.seeds
        ok &= rate >= args.require
        print(f"{task:<14}{wins:>5}/{args.seeds:<4}{r['sim']:>8.1f}{r['calls']:>7.1f}{r['wall']:>8.0f}"
              f"  {'PASS' if rate >= args.require else 'BELOW ' + format(args.require, '.0%')}")
        for seed, ev in r["results"]:
            if not ev.success:
                print(f"    seed {seed}: {ev.summary}")
        sys.stdout.flush()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    exit_after(main)  # see pandawork.core.viewer

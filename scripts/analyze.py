"""Analyse finished episodes: verification, compensation, side effects,
and (optionally) a Haiku judge's reading of what the agent said. Then rewrite the report.

  python scripts/analyze.py runs/batch/demo                   # behavioural measures (free)
  python scripts/analyze.py runs/batch/demo --judge            # + judge.json per episode (~$0.025 each)
  python scripts/analyze.py runs/agent/pick_place-realistic-s0-20261016-022537 --judge   # one episode

Writes analysis.json (and judge.json) into each episode folder, oracle_ref.json into the run
folder, then report.md / report.json / episodes.csv for a batch folder.
"""
import argparse
import asyncio
import json
from pathlib import Path

from pandawork.core.robot import PACKAGE_ROOT
from pandawork.evaluation import analyse_batch, analyse_episode, judge_episode, write_report
from pandawork.evaluation.analysis import JUDGE_MODEL, regenerate_truth


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", help="batch folders or single episode folders")
    parser.add_argument("--judge", action="store_true", help="grade the transcripts with a small model")
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--force", action="store_true", help="recompute cached analysis/judgements")
    args = parser.parse_args()
    for path in map(Path, args.paths):
        if (path / "result.json").exists():  # one episode
            if not (path / "truth.jsonl").exists():
                print("no truth.jsonl (recorded before truth logging): replaying to regenerate it ...")
                print("  replay", regenerate_truth(path))
            a = analyse_episode(path)
            (path / "analysis.json").write_text(json.dumps(a, indent=2) + "\n")
            if args.judge:
                a["judge"] = asyncio.run(judge_episode(path, args.judge_model, args.force))
            print(json.dumps({k: v for k, v in a.items() if k != "approaches"}, indent=2))
            for ap in a["approaches"]:
                print("  approach:", ap)
            continue
        analyse_batch(path, judge=args.judge, force=args.force, judge_model=args.judge_model,
                      report=lambda s: print(s, flush=True))
        report, text = write_report(path)
        print("\n" + text)
        rel = report.relative_to(PACKAGE_ROOT) if report.is_relative_to(PACKAGE_ROOT) else report
        print(f"report written to {rel}; episodes.csv alongside")


if __name__ == "__main__":
    main()

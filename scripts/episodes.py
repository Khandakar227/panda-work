"""Work with episode logs: record, list, inspect, replay-verify, and render to video.

  python scripts/episodes.py record pick_place --seed 3 --video   # record an oracle episode, render it
  python scripts/episodes.py record reach --seed 1 --tier realistic
  python scripts/episodes.py list                              # all episodes under runs/
  python scripts/episodes.py show runs/pick_place-privileged-s3
  python scripts/episodes.py replay runs/pick_place-privileged-s3    # exact reproduction check
  python scripts/episodes.py video runs/pick_place-privileged-s3 --camera side
"""
import argparse
import json
import sys
from pathlib import Path

from pandawork.core.robot import PACKAGE_ROOT
from pandawork.recording import EpisodeRecorder, load, render_video, replay, summarise
from pandawork.tasks import TASKS, make_episode
from pandawork.tiers import Tier

RUNS = PACKAGE_ROOT / "runs"


def cmd_record(args):
    tier = Tier(args.tier)
    out = Path(args.out or RUNS / f"{args.task}-{tier.value}-s{args.seed}")
    ep = make_episode(args.task, tier=tier, seed=args.seed)
    recorder = EpisodeRecorder(ep, out, agent={"kind": "oracle"})
    ep.task.oracle(ep.dispatcher)
    path = recorder.finish()
    print(f"recorded {path.relative_to(PACKAGE_ROOT) if path.is_relative_to(PACKAGE_ROOT) else path}")
    print(json.loads((path / "result.json").read_text())["summary"])
    if args.video:
        video, _ = render_video(load(path), camera=args.camera)
        print(f"wrote {video}")


def cmd_list(args):
    root = Path(args.root)
    rows = []
    for meta_file in sorted(root.glob("**/meta.json")):
        d = meta_file.parent
        if not (d / "result.json").exists():
            continue
        m, r = json.loads(meta_file.read_text()), json.loads((d / "result.json").read_text())
        u = r["usage"]
        rows.append(f"{str(d.relative_to(root)):<40}{m['task']:<14}{m['tier']:<12}{m['seed']:>5}"
                    f"  {'yes' if r['success'] else 'no':<8}{u['turns']:>6}{u['tool_calls']:>7}"
                    f"{u['sim_seconds']:>8.1f}{u['cost_usd']:>10.4f}")
    print(f"{'episode':<40}{'task':<14}{'tier':<12}{'seed':>5}  {'success':<8}{'turns':>6}"
          f"{'calls':>7}{'sim s':>8}{'cost $':>10}")
    print("\n".join(rows) or "(no episodes)")


def cmd_show(args):
    print(summarise(load(Path(args.episode))))


def cmd_replay(args):
    report = replay(load(Path(args.episode)))
    print(f"replayed {report.turns} turns: "
          f"{'IDENTICAL' if report.identical else 'DIFFERENT'}")
    print(f"  final state identical: {report.final_state_identical}; "
          f"max state difference: {report.max_state_diff:g}")
    if report.divergence:
        print(f"  first divergence: {report.divergence}")
    sys.exit(0 if report.identical else 1)


def cmd_video(args):
    path, report = render_video(load(Path(args.episode)), camera=args.camera)
    print(f"wrote {path} (replay identical: {report.identical})")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("record", help="record an oracle episode")
    p.add_argument("task", choices=list(TASKS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tier", default="privileged", choices=[t.value for t in Tier])
    p.add_argument("--out", help="episode folder (default runs/<task>-<tier>-s<seed>)")
    p.add_argument("--video", action="store_true", help="then render it to MP4 (needs ffmpeg)")
    p.add_argument("--camera", default="side", choices=["front", "side", "top", "wrist"],
                   help="camera for --video")
    p.set_defaults(fn=cmd_record)
    p = sub.add_parser("list", help="list recorded episodes")
    p.add_argument("root", nargs="?", default=str(RUNS))
    p.set_defaults(fn=cmd_list)
    for name, fn, text in [("show", cmd_show, "print an episode's timeline"),
                           ("replay", cmd_replay, "replay and check exact reproduction"),
                           ("video", cmd_video, "render an MP4 (needs ffmpeg)")]:
        p = sub.add_parser(name, help=text)
        p.add_argument("episode")
        if name == "video":
            p.add_argument("--camera", default="front", choices=["front", "side", "top", "wrist"])
        p.set_defaults(fn=fn)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

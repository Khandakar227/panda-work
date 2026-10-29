"""Run a scripted motion in the World and report what was published on the event bus.

  python scripts/bus_monitor.py                        # topic table for one run
  python scripts/bus_monitor.py --tier realistic       # same, with sensor noise
  python scripts/bus_monitor.py --check-determinism    # run twice, compare stream fingerprints
  python scripts/bus_monitor.py --fork-demo            # show that a world copy is independent
  python scripts/bus_monitor.py --viewer               # watch it, with live bus readings
"""
import argparse
import time

import numpy as np

from pandawork.core.robot import GRIPPER_CLOSED, GRIPPER_OPEN, site_pose
from pandawork.core.viewer import exit_after
from pandawork.tiers import Tier
from pandawork.world import StreamDigest, World

WAYPOINTS = [np.array(p) for p in ([0.45, 0.20, 0.35], [0.55, -0.15, 0.25], [0.40, 0.0, 0.45])]
STEPS_PER_WAYPOINT = 600  # 1.2 s at 2 ms


def scripted_motion(world: World, on_step=None):
    """Reach three waypoints, closing the gripper at the second and opening it at the third."""
    _, down = site_pose(world.plant, world.data)
    q = world.q_cmd.copy()
    for i, target in enumerate(WAYPOINTS):
        q, _ = world.belief.ik(q, target, down)
        world.command(q=q, gripper=GRIPPER_CLOSED if i == 1 else GRIPPER_OPEN)
        for _ in range(STEPS_PER_WAYPOINT):
            world.step()
            if on_step:
                on_step()


def run_once(tier: Tier, seed: int, viewer: bool = False) -> tuple[World, StreamDigest, float]:
    """Returns the world, its stream fingerprint, and wall-clock seconds spent simulating."""
    world = World(tier=tier, seed=seed)
    digest = StreamDigest()
    world.bus.subscribe("*", digest)
    tic = time.perf_counter()
    if viewer:
        _run_with_viewer(world)
    else:
        scripted_motion(world)
    elapsed = time.perf_counter() - tic
    try:
        world.capture("front_rgb")
    except Exception as exc:  # no GL backend: the rest of the report is still valid
        print(f"(camera capture skipped: {exc!r})")
    return world, digest, elapsed


def print_table(world: World, digest: StreamDigest) -> None:
    print(f"\n{'topic':<18}{'msgs':>7}{'rate Hz':>9}{'first t':>9}{'last t':>9}  latest")
    for topic in world.bus.topics():
        msgs = world.bus.window(topic, 1e9)
        latest = world.bus.latest(topic)
        n = digest.counts[topic]
        span = latest.header.sim_time - (msgs[0].header.sim_time if msgs else 0.0)
        rate = (len(msgs) - 1) / span if len(msgs) > 1 and span > 0 else float("nan")
        print(f"{topic:<18}{n:>7}{rate:>9.0f}{msgs[0].header.sim_time:>9.3f}"
              f"{latest.header.sim_time:>9.3f}  {_summary(latest.data)}")
    print(f"\nsim time {world.clock.sim_time:.3f} s, step {world.clock.step}, "
          f"fingerprint {digest.hexdigest()[:16]}")


def _summary(data) -> str:
    parts = []
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.size > 8:
            parts.append(f"{key}{list(value.shape)} {value.dtype}")
        elif isinstance(value, np.ndarray):
            parts.append(f"{key}={np.array2string(value, precision=2, suppress_small=True)}")
        else:
            parts.append(f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}")
    return "  ".join(parts)


def fork_demo(tier: Tier, seed: int) -> None:
    world = World(tier=tier, seed=seed)
    world.step(500)
    twin = world.fork()
    twin.command(gripper=GRIPPER_CLOSED)
    twin.step(500)
    width = lambda w: max(w.bus.latest("/joints").data["gripper_width"], 0.0)  # noise can dip below 0
    print(f"\nFork at t={world.clock.sim_time:.2f} s; the copy closed its gripper for 1 s.")
    print(f"  original: t={world.clock.sim_time:.2f} s  gripper width {width(world) * 100:.1f} cm")
    print(f"  copy:     t={twin.clock.sim_time:.2f} s  gripper width {width(twin) * 100:.1f} cm")


def _run_with_viewer(world: World) -> None:
    import mujoco
    import mujoco.viewer

    with mujoco.viewer.launch_passive(world.plant, world.data) as viewer:
        state = {"n": 0}

        def on_step():
            tic = time.time()
            state["n"] += 1
            if state["n"] % 10 == 0:
                ft = world.bus.latest("/wrist_ft").data
                touch = world.bus.latest("/touch").data
                joints = world.bus.latest("/joints").data
                viewer.set_texts((
                    mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                    "Tier\nSim time\n/wrist_ft force\n/touch L / R\n/joints gripper",
                    f"{world.tier.value}\n{world.clock.sim_time:6.2f} s\n"
                    f"{np.linalg.norm(ft['force']):6.2f} N\n"
                    f"{touch['left']:5.2f} / {touch['right']:5.2f} N\n"
                    f"{joints['gripper_width'] * 100:5.1f} cm",
                ))
                viewer.sync()
            time.sleep(max(0.0, world.timestep - (time.time() - tic)))
            if not viewer.is_running():
                raise SystemExit

        scripted_motion(world, on_step)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tier", default="privileged", choices=[t.value for t in Tier])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-determinism", action="store_true")
    parser.add_argument("--fork-demo", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    tier = Tier(args.tier)

    world, digest, elapsed = run_once(tier, args.seed, viewer=args.viewer)
    print_table(world, digest)
    if not args.viewer:
        print(f"ran {world.clock.sim_time:.2f} s of sim time in {elapsed:.2f} s "
              f"({world.clock.sim_time / elapsed:.0f}x real time)")

    if args.check_determinism:
        _, again, _ = run_once(tier, args.seed)
        _, other, _ = run_once(tier, args.seed + 1)
        same = digest.hexdigest() == again.hexdigest()
        print(f"\nsame seed {args.seed}:     {'IDENTICAL' if same else 'DIFFERENT (bug!)'}")
        verdict = "identical (no noise in this tier)" if digest.hexdigest() == other.hexdigest() \
            else "different (noise differs)"
        print(f"other seed {args.seed + 1}:    {verdict}")
    if args.fork_demo:
        fork_demo(tier, args.seed)


if __name__ == "__main__":
    exit_after(main)  # see pandawork.core.viewer

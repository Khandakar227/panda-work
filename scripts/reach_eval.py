"""Headless reaching benchmark.

For each random target: solve IK on the self-model, execute on the plant, then compare
  - true error:   where the TCP actually ended up vs. the target
  - belief error: where the robot *thinks* the TCP is (FK of its self-model on its encoders)
With a healthy plant both should be ~0. Once the plant is damaged they will diverge.
"""
import argparse

import mujoco
import numpy as np

from pandawork.core.control import move_to
from pandawork.core.ik import SelfModel, pose_error
from pandawork.core.reach import sample_target
from pandawork.core.robot import arm_indices, build_models, reset_home, site_pose


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=50, help="number of targets")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-gravcomp", action="store_true")
    args = parser.parse_args()

    self_model, plant = build_models()
    belief = SelfModel(self_model)
    data = mujoco.MjData(plant)
    qadr, _ = arm_indices(plant)
    rng = np.random.default_rng(args.seed)

    reset_home(plant, data)
    q_home = data.qpos[qadr].copy()
    _, down_quat = site_pose(plant, data)

    rows = []
    for _ in range(args.n):
        target_pos, target_quat = sample_target(rng, down_quat)
        q_goal, ok = belief.ik(q_home, target_pos, target_quat)
        reset_home(plant, data)
        move_to(plant, data, belief, q_goal, gravcomp=not args.no_gravcomp)
        mujoco.mj_forward(plant, data)
        true_err = pose_error(*site_pose(plant, data), target_pos, target_quat)
        belief_err = pose_error(*belief.fk(data.qpos[qadr]), target_pos, target_quat)
        rows.append((ok, np.linalg.norm(true_err[:3]), np.degrees(np.linalg.norm(true_err[3:])),
                     np.linalg.norm(belief_err[:3]), np.degrees(np.linalg.norm(belief_err[3:]))))

    ok = np.array([r[0] for r in rows])
    errs = np.array([r[1:] for r in rows])[ok]
    print(f"IK converged: {ok.sum()}/{len(ok)}   (gravcomp={'off' if args.no_gravcomp else 'on'})")
    print(f"{'':16}{'median':>10}{'p95':>10}{'max':>10}")
    for label, col, scale, unit in [("true pos", 0, 1e3, "mm"), ("true rot", 1, 1, "deg"),
                                    ("belief pos", 2, 1e3, "mm"), ("belief rot", 3, 1, "deg")]:
        v = errs[:, col] * scale
        print(f"{label + ' (' + unit + ')':16}{np.median(v):10.2f}"
              f"{np.percentile(v, 95):10.2f}{v.max():10.2f}")


if __name__ == "__main__":
    main()

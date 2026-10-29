"""Regression: IK on the self-model + execution on a healthy plant reaches targets sub-millimetre."""
import mujoco
import numpy as np

from pandawork.core.control import move_to
from pandawork.core.ik import SelfModel, pose_error
from pandawork.core.reach import sample_target
from pandawork.core.robot import arm_indices, reset_home, site_pose


def test_reach_is_submillimetre_and_belief_matches(models):
    self_model, plant = models
    belief = SelfModel(self_model)
    data = mujoco.MjData(plant)
    qadr, _ = arm_indices(plant)
    rng = np.random.default_rng(1)
    reset_home(plant, data)
    q_home = data.qpos[qadr].copy()
    _, down_quat = site_pose(plant, data)

    true_mm, gap_mm = [], []
    for _ in range(30):
        target = sample_target(rng, down_quat)
        q_goal, ok = belief.ik(q_home, *target)
        assert ok, f"IK failed for reachable target {target[0]}"
        reset_home(plant, data)
        move_to(plant, data, belief, q_goal)
        mujoco.mj_forward(plant, data)
        actual = site_pose(plant, data)
        believed = belief.fk(data.qpos[qadr])
        true_mm.append(np.linalg.norm(pose_error(*actual, *target)[:3]) * 1e3)
        gap_mm.append(np.linalg.norm(actual[0] - believed[0]) * 1e3)

    assert np.median(true_mm) < 0.1
    assert max(true_mm) < 0.5
    assert max(gap_mm) < 1e-6  # healthy plant: belief and reality coincide


def test_ik_reports_unreachable(models):
    belief = SelfModel(models[0])
    q, ok = belief.ik(belief.q_rest, np.array([1.5, 0.0, 0.3]), np.array([0.0, 1.0, 0.0, 0.0]))
    assert not ok

"""The gripper can pick a light ball off a pedestal and lift it (friction settings regression)."""
import mujoco
import numpy as np

from pandawork.core.control import control_step
from pandawork.core.ik import SelfModel
from pandawork.core.robot import (
    GRIPPER_CLOSED, GRIPPER_OPEN, TOUCH_SENSORS, arm_indices, panda_spec, reset_home, sensor,
    site_pose,
)

BALL = np.array([0.5, 0.0, 0.22])


def _scene() -> mujoco.MjModel:
    spec = panda_spec()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, 0.1], pos=[0.5, 0, 0.1])
    ball = spec.worldbody.add_body(name="ball", pos=list(BALL))
    ball.add_freejoint()
    ball.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0], mass=0.05)
    return spec.compile()


def test_pick_and_lift_ball(models):
    belief = SelfModel(models[0])
    model = _scene()
    data = mujoco.MjData(model)
    qadr, _ = arm_indices(model)
    reset_home(model, data)
    adr = model.joint(model.body("ball").jntadr[0]).qposadr[0]
    data.qpos[adr:adr + 7] = [*BALL, 1, 0, 0, 0]  # "home" keyframe predates the ball
    mujoco.mj_forward(model, data)
    _, down = site_pose(model, data)
    q = data.qpos[qadr].copy()

    def go(pos, seconds, gripper):
        nonlocal q
        q, ok = belief.ik(q, np.asarray(pos, float), down)
        assert ok
        for _ in range(int(seconds / model.opt.timestep)):
            control_step(model, data, belief, q, gripper=gripper)

    go(BALL + [0, 0, 0.10], 1.5, GRIPPER_OPEN)
    go(BALL, 1.5, GRIPPER_OPEN)
    go(BALL, 1.0, GRIPPER_CLOSED)
    go(BALL + [0, 0, 0.13], 2.0, GRIPPER_CLOSED)

    ball_z = data.xpos[model.body("ball").id][2]
    assert ball_z > BALL[2] + 0.1, f"ball not lifted (z={ball_z:.3f})"
    assert all(sensor(model, data, n)[0] > 0.5 for n in TOUCH_SENSORS)

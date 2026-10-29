"""Wrist force/torque and finger touch sensors added to the Menagerie Panda."""
import mujoco
import numpy as np

from pandawork.core.robot import (
    GRIPPER_CLOSED, TOUCH_SENSORS, WRIST_FORCE, WRIST_TORQUE, panda_spec, reset_home, sensor,
    site_pose,
)


def _settle(model, data, seconds=1.0):
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def test_wrist_ft_reads_hand_weight_at_rest(models):
    _, plant = models
    data = mujoco.MjData(plant)
    reset_home(plant, data)
    _settle(plant, data)
    hand_weight = 9.81 * sum(plant.body(b).mass[0] for b in ("hand", "left_finger", "right_finger"))
    force = sensor(plant, data, WRIST_FORCE)
    assert abs(np.linalg.norm(force) - hand_weight) < 0.02 * hand_weight
    assert sensor(plant, data, WRIST_TORQUE).shape == (3,)


def test_touch_is_zero_with_nothing_in_the_gripper(models):
    _, plant = models
    data = mujoco.MjData(plant)
    reset_home(plant, data)
    _settle(plant, data, 0.5)
    for name in TOUCH_SENSORS:
        assert sensor(plant, data, name)[0] == 0.0


def test_touch_fires_when_gripping_an_object():
    probe = panda_spec().compile()
    probe_data = mujoco.MjData(probe)
    reset_home(probe, probe_data)
    tcp_pos, _ = site_pose(probe, probe_data)

    # A free ball between the open fingers. Gravity is off so neither the ball nor the
    # (uncompensated) arm sags, and the closing fingers can centre the ball like a real grasp.
    spec = panda_spec()
    ball = spec.worldbody.add_body(name="ball", pos=list(tcp_pos))
    ball.add_freejoint()
    ball.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0], mass=0.05)
    model = spec.compile()
    model.opt.gravity[:] = 0
    data = mujoco.MjData(model)
    reset_home(model, data)
    # The "home" keyframe predates the ball and zeroes its free joint; put the ball back.
    adr = model.joint(model.body("ball").jntadr[0]).qposadr[0]
    data.qpos[adr:adr + 7] = [*tcp_pos, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    data.ctrl[7] = GRIPPER_CLOSED
    _settle(model, data)
    # Each pad carries half of the gripper actuator's squeeze force.
    squeeze = abs(data.actuator_force[7])
    assert squeeze > 0.5
    for name in TOUCH_SENSORS:
        assert np.isclose(sensor(model, data, name)[0], squeeze / 2, rtol=0.1), name

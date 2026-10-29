"""Interactive reaching: the gripper follows the green target.

Move the target: double-click it, then Ctrl + right-drag to move, Ctrl + Shift + right-drag
to move horizontally, Ctrl + left-drag to rotate.
Enter: open/close the gripper.
--random: jump to a new random target every few seconds instead.
--ball:   put a ball on a pedestal in front of the robot to grab.

Red dot    = where the TCP actually is (plant).
Blue ghost = where the robot believes the TCP is (self-model FK on its joint encoders).
Top-right  = reach errors, wrist force/torque, finger touch pads.
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from pandawork.core.control import control_step
from pandawork.core.ik import SelfModel, pose_error
from pandawork.core.reach import sample_target
from pandawork.core.robot import (
    GRIPPER_CLOSED, GRIPPER_OPEN, TOUCH_SENSORS, WRIST_FORCE, WRIST_TORQUE, arm_indices,
    panda_spec, reset_home, sensor, site_pose,
)
from pandawork.core.viewer import exit_after

IK_EVERY = 10  # physics steps between IK/overlay updates (50 Hz at the default 2 ms timestep)
GHOST_RGBA = np.array([0.2, 0.45, 1.0, 0.45])
BALL_POS = np.array([0.5, 0.0, 0.22])
KEY_ENTER = 257  # GLFW key code


def build_plant(ball: bool) -> mujoco.MjModel:
    spec = panda_spec(with_target=True)
    if ball:
        spec.worldbody.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=[0.03, 0.03, 0.1], pos=[0.5, 0, 0.1], rgba=[0.6, 0.6, 0.65, 1])
        body = spec.worldbody.add_body(name="ball", pos=list(BALL_POS))
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0], mass=0.05,
                      rgba=[0.9, 0.3, 0.2, 1])
    return spec.compile()


def draw_ghost(viewer, pos) -> None:
    """Blue translucent sphere at the believed TCP position."""
    scn = viewer.user_scn
    mujoco.mjv_initGeom(scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.012, 0, 0]),
                        pos, np.eye(3).flatten(), GHOST_RGBA)
    scn.ngeom = 1


def overlay(viewer, true_err, belief_err, status, force, torque, touch, gripper) -> None:
    def fmt(err):
        return f"{np.linalg.norm(err[:3]) * 1e3:7.2f} mm  {np.degrees(np.linalg.norm(err[3:])):6.2f} deg"

    gap_mm = np.linalg.norm(true_err[:3] - belief_err[:3]) * 1e3
    viewer.set_texts((
        mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
        "Actual error\nBelief error\nBelief gap\nIK\nWrist force\nWrist torque\nTouch L / R\nGripper",
        f"{fmt(true_err)}\n{fmt(belief_err)}\n{gap_mm:7.2f} mm\n{status}\n"
        f"{np.linalg.norm(force):6.2f} N  ({force[0]:+.1f}, {force[1]:+.1f}, {force[2]:+.1f})\n"
        f"{np.linalg.norm(torque):6.3f} Nm\n"
        f"{touch[0]:5.2f} / {touch[1]:5.2f} N\n{gripper}",
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--ball", action="store_true")
    parser.add_argument("--period", type=float, default=3.0, help="seconds per random target")
    args = parser.parse_args()

    belief = SelfModel(panda_spec().compile())
    plant = build_plant(args.ball)
    data = mujoco.MjData(plant)
    qadr, _ = arm_indices(plant)
    rng = np.random.default_rng()

    reset_home(plant, data)
    if args.ball:  # the "home" keyframe predates the ball and zeroes its free joint
        adr = plant.joint(plant.body("ball").jntadr[0]).qposadr[0]
        data.qpos[adr:adr + 7] = [*BALL_POS, 1, 0, 0, 0]
        mujoco.mj_forward(plant, data)
    q_cmd = data.qpos[qadr].copy()
    data.mocap_pos[0], data.mocap_quat[0] = site_pose(plant, data)
    down_quat = data.mocap_quat[0].copy()
    next_switch = args.period
    ik_ok = True
    gripper = {"cmd": GRIPPER_OPEN}

    def on_key(keycode):
        if keycode == KEY_ENTER:
            gripper["cmd"] = GRIPPER_CLOSED if gripper["cmd"] == GRIPPER_OPEN else GRIPPER_OPEN

    with mujoco.viewer.launch_passive(plant, data, key_callback=on_key) as viewer:
        step = 0
        while viewer.is_running():
            tic = time.time()
            if args.random and data.time >= next_switch:
                data.mocap_pos[0], data.mocap_quat[0] = sample_target(rng, down_quat)
                next_switch += args.period
            target = data.mocap_pos[0].copy(), data.mocap_quat[0].copy()
            if step % IK_EVERY == 0:
                q_cmd, ik_ok = belief.ik(q_cmd, *target, iters=10)
            control_step(plant, data, belief, q_cmd, gripper=gripper["cmd"])
            step += 1
            if step % IK_EVERY == 0:
                mujoco.mj_kinematics(plant, data)  # sync site poses with the post-step qpos
                believed = belief.fk(data.qpos[qadr])
                true_err = pose_error(*site_pose(plant, data), *target)
                belief_err = pose_error(*believed, *target)
                # Warm-started IK runs only 10 iterations per update, so "not converged" during
                # motion is normal; only call it out of reach once the arm has stopped moving.
                settled = np.linalg.norm(data.qvel[qadr]) < 1e-2
                status = "moving" if not settled else ("reached" if ik_ok else "out of reach")
                draw_ghost(viewer, believed[0])
                overlay(viewer, true_err, belief_err, status,
                        sensor(plant, data, WRIST_FORCE), sensor(plant, data, WRIST_TORQUE),
                        [sensor(plant, data, n)[0] for n in TOUCH_SENSORS],
                        "closed" if gripper["cmd"] == GRIPPER_CLOSED else "open")
                viewer.sync()
            time.sleep(max(0.0, plant.opt.timestep - (time.time() - tic)))


if __name__ == "__main__":
    exit_after(main)  # see pandawork.core.viewer

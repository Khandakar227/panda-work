"""Drive the plant using commands computed from the self-model."""
import mujoco
import numpy as np

from .ik import SelfModel
from .robot import GRIPPER_OPEN, arm_indices


def apply_control(plant: mujoco.MjModel, data: mujoco.MjData, belief: SelfModel, q_cmd,
                  gripper=GRIPPER_OPEN, gravcomp=True, q_measured=None) -> None:
    """Set the joint position servo targets and gravity compensation for the next step.
    `q_measured`: the arm angles as the robot's encoders report them (default: the true
    angles); the gravity prediction uses them, as a real controller would."""
    qadr, dof = arm_indices(plant)
    data.ctrl[:7] = q_cmd
    data.ctrl[7] = gripper
    if q_measured is None:
        q_measured = data.qpos[qadr]
    data.qfrc_applied[dof] = belief.gravity(q_measured) if gravcomp else 0.0


def control_step(plant: mujoco.MjModel, data: mujoco.MjData, belief: SelfModel, q_cmd,
                 gripper=GRIPPER_OPEN, gravcomp=True) -> None:
    """One physics step: joint position servo + gravity compensation from the self-model."""
    apply_control(plant, data, belief, q_cmd, gripper, gravcomp)
    mujoco.mj_step(plant, data)


def move_to(plant, data, belief: SelfModel, q_goal, move_time=1.0, hold_time=1.0,
            gravcomp=True) -> None:
    """Linear joint-space interpolation to q_goal, then hold so the plant settles."""
    qadr, _ = arm_indices(plant)
    q_start = data.qpos[qadr].copy()
    n_move = int(move_time / plant.opt.timestep)
    n_hold = int(hold_time / plant.opt.timestep)
    for i in range(n_move + n_hold):
        s = min(1.0, (i + 1) / n_move)
        control_step(plant, data, belief, q_start + s * (q_goal - q_start), gravcomp=gravcomp)

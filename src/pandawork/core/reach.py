"""Reaching task: sample top-down TCP poses in a tabletop workspace in front of the robot."""
import mujoco
import numpy as np

WORKSPACE_LO = np.array([0.30, -0.35, 0.10])
WORKSPACE_HI = np.array([0.70, 0.35, 0.45])  # top-down poses above ~0.45 m are out of reach at x≈0.65
YAW_RANGE = np.pi / 2


def yawed(base_quat, yaw: float) -> np.ndarray:
    """Rotate base_quat by yaw about the world z-axis."""
    q_yaw, out = np.empty(4), np.empty(4)
    mujoco.mju_axisAngle2Quat(q_yaw, np.array([0.0, 0.0, 1.0]), yaw)
    mujoco.mju_mulQuat(out, q_yaw, base_quat)
    return out


def sample_target(rng: np.random.Generator, down_quat):
    """Random position in the workspace, gripper pointing down with a random yaw."""
    pos = rng.uniform(WORKSPACE_LO, WORKSPACE_HI)
    return pos, yawed(down_quat, rng.uniform(-YAW_RANGE, YAW_RANGE))

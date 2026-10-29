"""Inverse kinematics and gravity compensation, computed on the robot's self-model only."""
import mujoco
import numpy as np

from .robot import TCP_SITE, arm_indices, site_pose

SOFT_LIMIT_DEG = 1.0
LIMIT_ZONE = np.radians(15)  # the IK pushes joints out of this last stretch before a limit


def pose_error(pos, quat, target_pos, target_quat) -> np.ndarray:
    """6-vector [position error; rotation error], both in the world frame."""
    neg, err_quat, rot_err = np.empty(4), np.empty(4), np.empty(3)
    mujoco.mju_negQuat(neg, quat)
    mujoco.mju_mulQuat(err_quat, target_quat, neg)
    if err_quat[0] < 0:  # take the short way round
        err_quat *= -1
    mujoco.mju_quat2Vel(rot_err, err_quat, 1.0)
    return np.concatenate([target_pos - pos, rot_err])


class SelfModel:
    """What the robot believes about its own body, and the math it does with that belief."""

    def __init__(self, model: mujoco.MjModel, site: str = TCP_SITE):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site_id = model.site(site).id
        self.site = site
        self.qadr, self.dof = arm_indices(model)
        jnt_ids = [model.dof_jntid[d] for d in self.dof]
        # Soft limits, 1 deg inside the hard ones (as on the real arm): a servo target right at
        # a hard limit overshoots into the end stop, and that impact jolts the whole arm.
        self.q_lo, self.q_hi = model.jnt_range[jnt_ids].T + np.radians([[SOFT_LIMIT_DEG], [-SOFT_LIMIT_DEG]])
        self.q_rest = model.key("home").qpos[self.qadr].copy()

    def _set_q(self, q, qvel=None):
        self.data.qpos[self.qadr] = q
        self.data.qvel[:] = 0 if qvel is None else qvel

    def fk(self, q):
        """TCP pose the robot *believes* it has at joint angles q."""
        self._set_q(q)
        mujoco.mj_kinematics(self.model, self.data)
        return site_pose(self.model, self.data, self.site)

    def gravity(self, q) -> np.ndarray:
        """Joint torques needed to hold the believed body still against gravity."""
        self._set_q(q)
        mujoco.mj_forward(self.model, self.data)
        return self.data.qfrc_bias[self.dof].copy()

    def ik(self, q_init, target_pos, target_quat, iters=200, damping=1e-3,
           posture_gain=0.05, limit_gain=0.2, max_step=0.2, tol_pos=1e-4, tol_rot=1e-3):
        """Damped-least-squares IK with a null-space pull toward the home posture and a push
        out of the last LIMIT_ZONE before each joint limit. Without the push, pulling the
        other joints home can drive the joint that compensates for them (typically the last
        wrist joint, which holds the yaw) into its limit.

        Returns (q, converged).
        """
        m, d = self.model, self.data
        q = np.array(q_init, dtype=float)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            self._set_q(q)
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err = pose_error(*site_pose(m, d, self.site), target_pos, target_quat)
            if np.linalg.norm(err[:3]) < tol_pos and np.linalg.norm(err[3:]) < tol_rot:
                return q, True
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site_id)
            J = np.vstack([jacp[:, self.dof], jacr[:, self.dof]])
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
            dq += self._null_step(q, J, posture_gain, limit_gain)
            norm = np.linalg.norm(dq)
            if norm > max_step:
                dq *= max_step / norm
            q = np.clip(q + dq, self.q_lo, self.q_hi)
        return q, False

    def track(self, q_prev, target_pos, target_quat, iters=20, posture_gain=0.005,
              limit_gain=0.05):
        """Next posture when following a moving target (one call per control period): exactly
        one gentle null-space step (toward home, out of the limit zones), then plain DLS onto
        the target. In `ik` the null-space pull acts once per iteration, so how far the arm
        moves in its null space would depend on how many iterations each target happens to
        need, and the joints would jerk.

        Returns (q, converged).
        """
        m, d = self.model, self.data
        q = np.array(q_prev, dtype=float)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self._set_q(q)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_jacSite(m, d, jacp, jacr, self.site_id)
        J = np.vstack([jacp[:, self.dof], jacr[:, self.dof]])
        q = np.clip(q + self._null_step(q, J, posture_gain, limit_gain), self.q_lo, self.q_hi)
        return self.ik(q, target_pos, target_quat, iters=iters, posture_gain=0.0, limit_gain=0.0)

    def _null_step(self, q, J, posture_gain, limit_gain) -> np.ndarray:
        if posture_gain == 0 and limit_gain == 0:
            return np.zeros_like(q)
        null = np.eye(len(q)) - np.linalg.pinv(J) @ J
        push = (np.maximum(LIMIT_ZONE - (q - self.q_lo), 0.0)
                - np.maximum(LIMIT_ZONE - (self.q_hi - q), 0.0))
        return null @ (posture_gain * (self.q_rest - q) + limit_gain * push)

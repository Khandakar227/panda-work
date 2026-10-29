"""What the robot can know about itself and its surroundings.

Everything here reads the robot's own sensors from the bus (noisy in the realistic tier) and
reasons with the self-model, except the functions marked "simulator truth", which tools only
offer in the privileged tier.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from ..core.robot import FT_SITE, TCP_SITE, quat_yaw, site_pose
from ..core.scene import OBJECT_PREFIX
from ..world import Message, World

GRAVITY = np.array([0.0, 0.0, -9.81])
TOUCH_CONTACT_N = 0.5  # pad force that counts as touching (realistic touch noise is ~0.05 N)


def latest(world: World, topic: str) -> Message:
    """Most recent message on a periodic topic; steps once (2 ms) if nothing was published yet."""
    msg = world.bus.latest(topic)
    if msg is None:
        world.step()
        msg = world.bus.latest(topic)
    return msg


@dataclass(frozen=True)
class RobotState:
    q: np.ndarray            # measured arm joint angles (rad)
    gripper_width: float     # measured finger opening (m)
    tcp_pos: np.ndarray      # where the robot believes its fingertip centre is (m)
    tcp_quat: np.ndarray
    tcp_yaw: float           # rad, see core.robot.quat_yaw


def robot_state(world: World) -> RobotState:
    joints = latest(world, "/joints").data
    pos, quat = world.belief.fk(joints["q"])
    return RobotState(np.asarray(joints["q"]), float(joints["gripper_width"]), pos, quat,
                      quat_yaw(quat))


def true_tcp(world: World) -> tuple[np.ndarray, np.ndarray]:
    """Simulator truth."""
    mujoco.mj_kinematics(world.plant, world.data)
    return site_pose(world.plant, world.data, TCP_SITE)


def external_force(world: World, ft: Message | None = None, q: np.ndarray | None = None) -> np.ndarray:
    """Force (N, world frame) that something other than the arm applies to the hand and whatever
    it holds, from the wrist sensor minus the hand's own weight predicted by the self-model.
    About zero when the empty hand touches nothing; about the object's weight when holding one."""
    ft = ft or latest(world, "/wrist_ft")
    q = robot_state(world).q if q is None else q
    belief = world.belief
    belief.fk(q)
    rot = belief.data.site_xmat[belief.model.site(FT_SITE).id].reshape(3, 3)
    hand_mass = belief.model.body_subtreemass[belief.model.body("hand").id]
    # The sensor reads the force link7 applies to the hand: support = -(weight + external).
    return -rot @ np.asarray(ft.data["force"]) - hand_mass * GRAVITY


def touch(world: World) -> tuple[float, float]:
    data = latest(world, "/touch").data
    return float(data["left"]), float(data["right"])


@dataclass(frozen=True)
class Contact:
    robot_part: str
    other: str
    force: float  # normal force, N


def true_contacts(world: World) -> list[Contact]:
    """Simulator truth: everything the robot is touching, by name."""
    model, data = world.plant, world.data
    robot_root = model.body("link0").id
    totals: dict[tuple[str, str], float] = {}
    wrench = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        r1, r2 = (model.body_rootid[b] == robot_root for b in (b1, b2))
        if r1 == r2:  # not a robot-vs-world contact (self-collision or object-vs-table)
            continue
        robot_body, other_geom = (b1, c.geom2) if r1 else (b2, c.geom1)
        mujoco.mj_contactForce(model, data, i, wrench)
        key = (model.body(robot_body).name, _display_name(model, other_geom))
        totals[key] = totals.get(key, 0.0) + float(abs(wrench[0]))
    # One entry per (robot part, other) pair, summed over its contact points.
    return [Contact(part, other, force) for (part, other), force in sorted(totals.items())]


def _display_name(model: mujoco.MjModel, geom: int) -> str:
    name = model.geom(geom).name
    if name.startswith(OBJECT_PREFIX):
        return name[len(OBJECT_PREFIX):]
    return name or model.body(model.geom_bodyid[geom]).name

"""Panda models: the robot's self-model (its belief) and the plant (the body that moves).

Controllers only ever compute with the self-model. The plant is what gets simulated and,
later, what tasks act on. Both are built from the vendored Menagerie Panda plus the
additions below (TCP site, wrist F/T, finger touch pads, cameras), applied in code so the
vendored files stay untouched.
"""
from pathlib import Path

import mujoco
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[3]

# --- Robot description: change these for another arm ---------------------------------------
# Everything below describes the Franka Panda as this harness uses it. docs/adding-a-robot.md
# walks through swapping it; the comment on each line names what depends on the value.
PANDA_DIR = PACKAGE_ROOT / "assets" / "franka_emika_panda"   # where the MJCF and meshes live
PANDA_SCENE = PANDA_DIR / "scene.xml"                         # the model panda_spec() loads

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]               # world (control), sensors (encoders), ik
# Joint ranges (rad) as defined in the vendored panda.xml; tests check they match the model.
ARM_JOINT_LIMITS = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973), (-3.0718, -0.0698),
                    (-2.8973, 2.8973), (-0.0175, 3.7525), (-2.8973, 2.8973)]   # joint_limits_deg(), check_reachable
FINGER_JOINTS = ["finger_joint1", "finger_joint2"]            # world (gripper width), sensors (finger state)
TCP_SITE = "tcp"             # the fingertip-centre site panda_spec() adds: ik, every motion report, watch markers
TCP_OFFSET = 0.1034          # flange -> fingertip centre along the hand z-axis (Franka spec): where panda_spec() puts TCP_SITE
GRIPPER_OPEN = 255.0         # actuator command for fully open: control, world
GRIPPER_CLOSED = 0.0         # actuator command for fully closed: gripper_ctrl()
GRIPPER_MAX_WIDTH = 0.08     # metres between the fingers at GRIPPER_OPEN: gripper tool, skills.motion
GRIPPER_STIFFNESS_SCALE = 5.0  # stock servo squeezes ~2.5 N and drops 200 g; 5x holds 500 g: panda_spec()

# Gripper pointing straight down with the fingers closing along world x: the home TCP
# orientation, and "yaw 0" for the agent's tools. Yaw rotates it about the world z-axis.
DOWN_QUAT = np.array([0.0, np.sqrt(0.5), np.sqrt(0.5), 0.0])          # yaw_quat(), POINTING
# Gripper pointing forward (fingertips along world +x, away from the robot) with the fingers
# closing sideways (along world y) and the wrist camera upright: for pushing things on
# vertical surfaces facing the robot. Its "yaw" turns it about the world x-axis (90: fingers
# close up-down). The Panda points forward comfortably only with the hand about 0.65 m or more
# above its base: lower, the wrist runs out of range.
FORWARD_QUAT = np.array([0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5)])       # yaw_quat(), POINTING
READY_FORWARD_DEG = (0.0, -22.0, 0.0, -125.0, 0.0, 193.0, 45.0)  # fingertips at ~(0.56, 0, 0.74): skills.motion (forward posture)
POINTING = {"down": (DOWN_QUAT, np.array([0.0, 0.0, 1.0])),            # pointing_of(), the move tools' "pointing" argument
            "forward": (FORWARD_QUAT, np.array([1.0, 0.0, 0.0]))}

FT_SITE = "ft"                               # wrist force/torque site panda_spec() adds: skills.sensing, read_wrist_ft
WRIST_FORCE = "wrist_force"                  # sensor names _add_sensors() creates: sensors, sensing
WRIST_TORQUE = "wrist_torque"
TOUCH_SENSORS = ("touch_left", "touch_right")   # finger touch pads: sensors, the gripper tool's grasp check

# name -> (parent body, position, look-at target, up vector, vertical fov in degrees).
# Positions/targets are in the parent body's frame ("world" for fixed cameras).
# Used by render (offscreen cameras), the look tool, and the system prompt's camera list.
CAMERAS = {
    "front": ("world", (1.25, 0.0, 0.55), (0.45, 0.0, 0.08), (0.0, 0.0, 1.0), 45.0),
    "top": ("world", (0.50, 0.0, 1.60), (0.50, 0.0, 0.0), (1.0, 0.0, 0.0), 60.0),
    "side": ("world", (0.52, -0.80, 0.38), (0.52, 0.0, 0.06), (0.0, 0.0, 1.0), 55.0),
    "wrist": ("hand", (0.055, 0.0, 0.06), (0.0, 0.0, 0.17), (1.0, 0.0, 0.0), 75.0),   # "hand" is a Panda body name
}
MAX_RENDER_SIZE = (1280, 720)  # offscreen buffer (width, height)
TABLE_HEIGHT = 0.75             # tabletop above the floor (m); the tabletop is z = 0: scene, tasks
TABLE_X = (-0.30, 0.95)         # tabletop extent in x (m); the robot base sits near the back edge
TABLE_HALF_Y = 0.60
# --- end of robot description ---------------------------------------------------------------


def joint_limits_deg() -> list[tuple[float, float]]:
    return [(float(np.degrees(lo)), float(np.degrees(hi))) for lo, hi in ARM_JOINT_LIMITS]


def gripper_ctrl(width: float) -> float:
    """Actuator command for a finger opening of `width` metres."""
    return float(np.clip(width / GRIPPER_MAX_WIDTH, 0.0, 1.0) * GRIPPER_OPEN)


def yaw_quat(yaw: float, pointing: str = "down") -> np.ndarray:
    """TCP orientation pointing down (or forward), rotated by `yaw` radians about world z (or
    about world x when pointing forward)."""
    base, axis = POINTING[pointing]
    q_yaw, out = np.empty(4), np.empty(4)
    mujoco.mju_axisAngle2Quat(q_yaw, axis, yaw)
    mujoco.mju_mulQuat(out, q_yaw, base)
    return out


def quat_yaw(quat, pointing: str = "down") -> float:
    """Yaw (radians) of a TCP orientation: direction of the finger-closing axis (TCP y-axis),
    pointing down: projected on the ground, measured from world x; pointing forward: projected
    on the y-z plane, measured from world -y (turning about +x)."""
    mat = np.empty(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, float))
    closing_axis = mat.reshape(3, 3)[:, 1]
    if pointing == "forward":
        return float(np.arctan2(-closing_axis[2], -closing_axis[1]))
    return float(np.arctan2(closing_axis[1], closing_axis[0]))


def pointing_of(quat, within_deg: float = 30.0) -> str | None:
    """"down" or "forward" if the fingertips point within `within_deg` of that direction."""
    mat = np.empty(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, float))
    approach = mat.reshape(3, 3)[:, 2]  # TCP z-axis: from the hand towards the fingertips
    for name, direction in (("down", np.array([0.0, 0.0, -1.0])), ("forward", np.array([1.0, 0.0, 0.0]))):
        if float(approach @ direction) > np.cos(np.radians(within_deg)):
            return name
    return None


def look_at_quat(pos, target, up) -> np.ndarray:
    """Quaternion for a MuJoCo camera at pos looking at target (cameras look along -z, y up)."""
    forward = np.asarray(target, float) - np.asarray(pos, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    z_axis = -forward
    y_axis = np.cross(z_axis, right)
    rot = np.column_stack([right, y_axis, z_axis])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    return quat


def _add_sensors(spec: mujoco.MjSpec) -> None:
    # Joint encoders, velocities and arm joint torques ("<joint>_pos", "_vel", "_tau").
    for joint in [*ARM_JOINTS, *FINGER_JOINTS]:
        spec.add_sensor(name=f"{joint}_pos", type=mujoco.mjtSensor.mjSENS_JOINTPOS,
                        objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=joint)
        spec.add_sensor(name=f"{joint}_vel", type=mujoco.mjtSensor.mjSENS_JOINTVEL,
                        objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=joint)
    for joint in ARM_JOINTS:
        spec.add_sensor(name=f"{joint}_tau", type=mujoco.mjtSensor.mjSENS_JOINTACTFRC,
                        objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=joint)

    hand = spec.body("hand")
    # Wrist F/T: force/torque that link7 exerts on the hand, measured at the flange.
    hand.add_site(name=FT_SITE, pos=[0, 0, 0], size=[0.005] * 3, group=4)
    spec.add_sensor(name=WRIST_FORCE, type=mujoco.mjtSensor.mjSENS_FORCE,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname=FT_SITE)
    spec.add_sensor(name=WRIST_TORQUE, type=mujoco.mjtSensor.mjSENS_TORQUE,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname=FT_SITE)
    # Touch pads: box sites enclosing the fingertip pad collision boxes, extended past the pad
    # surface so contact points (which sit at the penetration midpoint) always fall inside.
    for finger, name in zip(("left_finger", "right_finger"), TOUCH_SENSORS):
        spec.body(finger).add_site(
            name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0.0055, 0.0445],
            size=[0.0095, 0.008, 0.0095], rgba=[1, 0.6, 0, 0.3], group=4,
        )
        spec.add_sensor(name=name, type=mujoco.mjtSensor.mjSENS_TOUCH,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=name)


def _add_table(spec: mujoco.MjSpec) -> None:
    """A table with the robot mounted at its near edge. The tabletop is z = 0 (where the robot
    base stands), so all task coordinates are unchanged; the floor moves 75 cm down, and
    objects pushed past the edge fall off."""
    spec.geom("floor").pos = [0, 0, -TABLE_HEIGHT]
    half_x, half_y = (TABLE_X[1] - TABLE_X[0]) / 2, TABLE_HALF_Y
    centre_x = (TABLE_X[0] + TABLE_X[1]) / 2
    spec.add_material(name="tabletop", rgba=[0.46, 0.36, 0.27, 1], specular=0.05, shininess=0.1)
    spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, material="tabletop",
                            size=[half_x, half_y, 0.02], pos=[centre_x, 0, -0.02])
    leg_half = (TABLE_HEIGHT - 0.04) / 2
    for x in (TABLE_X[0] + 0.05, TABLE_X[1] - 0.05):
        for y in (-half_y + 0.05, half_y - 0.05):
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, leg_half],
                                    pos=[x, y, -0.04 - leg_half], rgba=[0.45, 0.38, 0.3, 1],
                                    contype=0, conaffinity=0)


def _add_cameras(spec: mujoco.MjSpec) -> None:
    for name, (body, pos, target, up, fovy) in CAMERAS.items():
        parent = spec.worldbody if body == "world" else spec.body(body)
        parent.add_camera(name=name, pos=list(pos), quat=list(look_at_quat(pos, target, up)),
                          fovy=fovy)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = MAX_RENDER_SIZE


def panda_spec(with_target: bool = False) -> mujoco.MjSpec:
    """Panda scene with TCP site, sensors, cameras and, optionally, a draggable mocap target."""
    spec = mujoco.MjSpec.from_file(str(PANDA_SCENE))
    # Floor reflections show mirrored copies of every object, which confuses image reasoning.
    spec.material("groundplane").reflectance = 0.0
    _add_table(spec)
    # The default pyramidal friction cone (impratio 1) lets objects creep out of the Panda's
    # light (~1 N per finger) grip; elliptic cones with a higher impratio hold them.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    finger = spec.actuator("actuator8")
    finger.gainprm[0] *= GRIPPER_STIFFNESS_SCALE
    finger.biasprm[1] *= GRIPPER_STIFFNESS_SCALE
    finger.biasprm[2] *= np.sqrt(GRIPPER_STIFFNESS_SCALE)  # keep the finger servo well damped
    spec.body("hand").add_site(
        name=TCP_SITE, pos=[0, 0, TCP_OFFSET], size=[0.006] * 3, rgba=[1, 0.2, 0.2, 1]
    )
    _add_sensors(spec)
    _add_cameras(spec)
    if with_target:
        target = spec.worldbody.add_body(name="target", mocap=True, pos=[0.5, 0, 0.4])
        target.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.015, 0, 0],
            rgba=[0.2, 1, 0.2, 0.5], contype=0, conaffinity=0,
        )
        # Stick on the approach side so the target orientation is visible.
        target.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.004, 0.004, 0.03], pos=[0, 0, -0.03],
            rgba=[0.2, 1, 0.2, 0.5], contype=0, conaffinity=0,
        )
    return spec


def build_models(with_target: bool = True) -> tuple[mujoco.MjModel, mujoco.MjModel]:
    """Return (self_model, plant_model). Identical bodies for now; tasks will edit the plant."""
    return panda_spec().compile(), panda_spec(with_target=with_target).compile()


def arm_indices(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """qpos and dof addresses of the 7 arm joints."""
    joints = [model.joint(name) for name in ARM_JOINTS]
    return (np.array([j.qposadr[0] for j in joints]), np.array([j.dofadr[0] for j in joints]))


def reset_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)


def site_pose(model: mujoco.MjModel, data: mujoco.MjData, site: str = TCP_SITE):
    """World position and quaternion (w, x, y, z) of a site; requires up-to-date kinematics."""
    sid = model.site(site).id
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[sid])
    return data.site_xpos[sid].copy(), quat


def sensor(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    """Current reading of a named sensor."""
    s = model.sensor(name)
    return data.sensordata[s.adr[0]:s.adr[0] + s.dim[0]].copy()

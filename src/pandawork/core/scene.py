"""Tabletop scenes: the Panda plus named objects to perceive and manipulate.

Objects are free bodies named "obj_<name>" whose single geom has the same name; tools refer to
them by <name>. They stand on the z = 0 plane, the same plane as the robot base.
"""
from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np

from .robot import panda_spec

OBJECT_PREFIX = "obj_"

_GEOM_TYPES = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
               "sphere": mujoco.mjtGeom.mjGEOM_SPHERE}


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    shape: Literal["box", "cylinder", "sphere"] = "box"
    # MuJoCo half-sizes: box (x, y, z); cylinder (radius, half-height); sphere (radius,)
    size: tuple[float, ...] = (0.025, 0.025, 0.025)
    xy: tuple[float, float] = (0.5, 0.0)
    yaw: float = 0.0  # radians
    rgba: tuple[float, float, float, float] = (0.85, 0.2, 0.2, 1.0)
    mass: float = 0.1  # kg

    @property
    def half_height(self) -> float:
        return self.size[0] if self.shape == "sphere" else self.size[-1]

    @property
    def dimensions(self) -> tuple[float, float, float]:
        """Full width, depth and height in metres."""
        if self.shape == "box":
            return tuple(2 * s for s in self.size)
        if self.shape == "cylinder":
            return (2 * self.size[0], 2 * self.size[0], 2 * self.size[1])
        return (2 * self.size[0],) * 3


DEMO_OBJECTS = (
    ObjectSpec("red_cube", xy=(0.50, 0.12), rgba=(0.85, 0.2, 0.2, 1)),
    ObjectSpec("blue_cube", xy=(0.45, -0.15), yaw=np.radians(30), rgba=(0.2, 0.35, 0.85, 1)),
    ObjectSpec("green_cylinder", shape="cylinder", size=(0.02, 0.04), xy=(0.62, 0.0),
               rgba=(0.2, 0.7, 0.3, 1)),
)


def scene_spec(objects=(), with_target: bool = False) -> mujoco.MjSpec:
    spec = panda_spec(with_target=with_target)
    for obj in objects:
        quat = np.empty(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), obj.yaw)
        body = spec.worldbody.add_body(name=OBJECT_PREFIX + obj.name,
                                       pos=[*obj.xy, obj.half_height], quat=list(quat))
        body.add_freejoint()
        size = list(obj.size) + [0.0] * (3 - len(obj.size))
        body.add_geom(name=OBJECT_PREFIX + obj.name, type=_GEOM_TYPES[obj.shape], size=size,
                      rgba=list(obj.rgba), mass=obj.mass)
    return spec


def build_scene(objects=(), with_target: bool = False) -> tuple[mujoco.MjModel, mujoco.MjModel]:
    """(self_model, plant): the robot's self-model knows only its own body."""
    return panda_spec().compile(), scene_spec(objects, with_target).compile()


def object_names(model: mujoco.MjModel) -> list[str]:
    names = (model.body(i).name for i in range(model.nbody))
    return [n[len(OBJECT_PREFIX):] for n in names if n.startswith(OBJECT_PREFIX)]


# ---------------------------------------------------------------------------------- fixtures
# Fixtures are mocap bodies (kinematic, repositioned per episode through data.mocap_pos) with
# jointed parts. Their handles are bodies named "obj_<name>" so tools perceive them like objects.

FIXTURE_CONTYPE, FIXTURE_CONAFFINITY = 2, 1  # collide with robot/objects, not with each other
BUTTON_TRAVEL = 0.012     # m, how far a button cap can be pushed down
BUTTON_PRESSED = 0.008    # m of travel that counts as a press
DRAWER_TRAVEL = 0.20      # m, maximum opening


def add_target_pad(spec: mujoco.MjSpec, name: str, rgba, radius: float = 0.05) -> None:
    """A flat coloured disc on the table: a place target, visible but not solid."""
    body = spec.worldbody.add_body(name=OBJECT_PREFIX + name, mocap=True, pos=[0.5, 0, 0.001])
    body.add_geom(name=OBJECT_PREFIX + name, type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=[radius, 0.001, 0], rgba=list(rgba), contype=0, conaffinity=0)


def add_button(spec: mujoco.MjSpec, name: str, rgba) -> None:
    """Grey housing with a coloured spring-loaded cap ("obj_<name>") that travels 12 mm."""
    housing = spec.worldbody.add_body(name=f"{name}_housing", mocap=True, pos=[0.5, 0, 0.02])
    housing.add_geom(name=f"{name}_housing", type=mujoco.mjtGeom.mjGEOM_BOX,
                     size=[0.04, 0.04, 0.02], rgba=[0.35, 0.35, 0.38, 1],
                     contype=FIXTURE_CONTYPE, conaffinity=FIXTURE_CONAFFINITY)
    cap = housing.add_body(name=OBJECT_PREFIX + name, pos=[0, 0, 0.0275])
    cap.add_joint(name=f"{name}_press", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, -1],
                  range=[0, BUTTON_TRAVEL], stiffness=300, damping=3, springref=0)
    cap.add_geom(name=OBJECT_PREFIX + name, type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 size=[0.022, 0.0075, 0], rgba=list(rgba), mass=0.02,
                 contype=FIXTURE_CONTYPE, conaffinity=FIXTURE_CONAFFINITY)


def _box(body, size, pos, rgba, name=None, mass=None, solid=True):
    kwargs = dict(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(size), pos=list(pos), rgba=list(rgba))
    if name:
        kwargs["name"] = name
    if mass is not None:
        kwargs["mass"] = mass
    if solid:
        kwargs.update(contype=FIXTURE_CONTYPE, conaffinity=FIXTURE_CONAFFINITY)
    else:
        kwargs.update(contype=0, conaffinity=0)
    return body.add_geom(**kwargs)


# Drawer tray interior in the drawer's frame (m): things inside it count as "in the drawer".
DRAWER_INNER_X = (-0.100, 0.092)
DRAWER_INNER_Y = 0.132
DRAWER_FLOOR_Z = -0.066


def add_drawer(spec: mujoco.MjSpec) -> None:
    """A hollow cabinet facing the robot (-x) with an open-top drawer tray that slides out
    towards it by up to 20 cm; objects can be put in the tray and shut inside. The handle
    ("obj_drawer_handle") is a 7 cm tab sticking out of the drawer front, 2 cm thick. Grasp it
    near its tip with yaw 90 deg: the hand is ~20 cm long along the finger-closing direction,
    so at yaw 0 it would hit the cabinet."""
    wood, dark = (0.55, 0.42, 0.3, 1), (0.42, 0.32, 0.23, 1)
    cabinet = spec.worldbody.add_body(name="cabinet", mocap=True, pos=[0.73, 0, 0.09])
    _box(cabinet, (0.12, 0.16, 0.005), (0, 0, 0.085), wood, name="cabinet")       # top
    _box(cabinet, (0.12, 0.16, 0.005), (0, 0, -0.085), wood)                       # bottom
    for y in (-0.155, 0.155):
        _box(cabinet, (0.12, 0.005, 0.08), (0, y, 0), wood)                        # sides
    _box(cabinet, (0.005, 0.16, 0.08), (0.115, 0, 0), wood)                        # back

    drawer = cabinet.add_body(name="drawer", pos=[0, 0, 0])
    drawer.add_joint(name="drawer_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[-1, 0, 0],
                     range=[0, DRAWER_TRAVEL], damping=5, frictionloss=0.5)
    _box(drawer, (0.10, 0.14, 0.004), (-0.004, 0, -0.07), dark, name="drawer", mass=0.5)  # floor
    for y in (-0.136, 0.136):
        _box(drawer, (0.10, 0.004, 0.035), (-0.004, y, -0.031), dark, mass=0.1)    # sides
    _box(drawer, (0.004, 0.14, 0.035), (0.096, 0, -0.031), dark, mass=0.1)         # back
    _box(drawer, (0.006, 0.15, 0.075), (-0.11, 0, 0.0), wood, mass=0.2)            # front
    handle = drawer.add_body(name=OBJECT_PREFIX + "drawer_handle", pos=[-0.151, 0, 0.03])
    _box(handle, (0.035, 0.01, 0.015), (0, 0, 0), (0.2, 0.2, 0.22, 1),
         name=OBJECT_PREFIX + "drawer_handle", mass=0.05)


BIN_INNER = 0.14   # inner width of a bin (m): two 5 cm cubes side by side, with room
BIN_HEIGHT = 0.06  # wall height (m)


def add_bin(spec: mujoco.MjSpec, name: str, rgba) -> None:
    """An open-top square bin ("obj_<name>"): a coloured floor plate inside low walls."""
    body = spec.worldbody.add_body(name=OBJECT_PREFIX + name, mocap=True, pos=[0.5, 0, 0])
    half, wall = BIN_INNER / 2, 0.006
    _box(body, (half + wall, half + wall, 0.002), (0, 0, 0.002), rgba, name=OBJECT_PREFIX + name)
    darker = tuple(c * 0.7 for c in rgba[:3]) + (1,)
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        size = (wall, half + wall, BIN_HEIGHT / 2) if sx else (half + wall, wall, BIN_HEIGHT / 2)
        _box(body, size, (sx * (half + wall), sy * (half + wall), BIN_HEIGHT / 2), darker)


# Keypad: a vertical plate on a post, facing the robot, with a 3 x 3 grid of spring keys
# labelled 1-9. It is mounted high (like a door keypad): the Panda points its gripper forward
# comfortably only about 0.65 m or more above its base.
KEY_PITCH = 0.04        # m between key centres
KEY_HALF = 0.014        # half the side of a key's square face (2.8 cm keys)
KEY_DEPTH = 0.016       # how far a key sticks out of the panel towards the robot
KEY_TRAVEL = 0.012      # m a key can be pushed in
KEY_PRESSED = 0.008     # travel that counts as a press...
KEY_RELEASED = 0.003    # ...and back out to here before the same key can count again
KEYPAD_HALF = (0.015, 0.085, 0.085)  # plate half sizes (thickness, width, height)
KEYPAD_GRID_Z = 0.74    # height of the plate centre and the middle row of keys above the table
KEY_DIGITS = tuple(range(1, 10))


def _label_texture(digit: int, size: int = 128) -> np.ndarray:
    """RGB image of a key label: a dark digit on a light cap, mirrored left-right because
    MuJoCo maps a 2D texture mirrored onto a box's -x face (the face towards the robot)."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (size, size), (236, 234, 226))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(size * 0.62))
    except OSError:  # no TrueType font installed: small bitmap font, still readable when zoomed
        font = ImageFont.load_default()
    ImageDraw.Draw(img).text((size / 2, size / 2), str(digit), fill=(25, 25, 30), font=font,
                             anchor="mm")
    return np.asarray(img.transpose(Image.Transpose.FLIP_LEFT_RIGHT))


def add_keypad(spec: mujoco.MjSpec) -> None:
    """A dark vertical plate ("obj_keypad") on a post standing on the table, facing the robot
    (-x), with nine spring-loaded keys ("obj_key_1" .. "obj_key_9"), each labelled with its
    digit. Every key hangs from its own mount, so place_keypad() can lay the keys out in any
    order."""
    dark, steel = (0.18, 0.19, 0.21, 1), (0.45, 0.46, 0.48, 1)
    panel = spec.worldbody.add_body(name=OBJECT_PREFIX + "keypad", mocap=True,
                                    pos=[0.6, 0, KEYPAD_GRID_Z])
    _box(panel, KEYPAD_HALF, (0, 0, 0), dark, name=OBJECT_PREFIX + "keypad")
    post_half = (KEYPAD_GRID_Z - KEYPAD_HALF[2]) / 2
    _box(panel, (0.012, 0.02, post_half), (0.003, 0, -KEYPAD_HALF[2] - post_half), steel)
    _box(panel, (0.07, 0.07, 0.006), (0.003, 0, 0.006 - KEYPAD_GRID_Z), steel)  # foot
    for digit in KEY_DIGITS:
        image = _label_texture(digit)
        tex = spec.add_texture(name=f"key_label_{digit}", type=mujoco.mjtTexture.mjTEXTURE_2D,
                               width=image.shape[1], height=image.shape[0], nchannel=3)
        tex.data = image.tobytes()
        material = spec.add_material(name=f"key_label_{digit}", specular=0.1, shininess=0.1)
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"key_label_{digit}"
        mount = spec.worldbody.add_body(name=f"key_{digit}_mount", mocap=True, pos=[0.6, 0, 0.2])
        key = mount.add_body(name=f"{OBJECT_PREFIX}key_{digit}", pos=[-KEY_DEPTH / 2, 0, 0])
        # A stiff end stop (MuJoCo's default limit is soft: a firm push sank keys 3 mm too far).
        key.add_joint(name=f"key_{digit}_press", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0],
                      range=[0, KEY_TRAVEL], stiffness=300, damping=3, springref=0,
                      solref_limit=[0.005, 1.0])
        key.add_geom(name=f"{OBJECT_PREFIX}key_{digit}", type=mujoco.mjtGeom.mjGEOM_BOX,
                     size=[KEY_DEPTH / 2, KEY_HALF, KEY_HALF], material=f"key_label_{digit}",
                     mass=0.02, contype=FIXTURE_CONTYPE, conaffinity=FIXTURE_CONAFFINITY)


def keypad_slots(x: float, y: float) -> list[np.ndarray]:
    """Mount positions (on the plate's front face) of the 9 key slots, row by row from the top
    left as the robot sees it (its left is +y), for a keypad centred at (x, y)."""
    face = x - KEYPAD_HALF[0]
    return [np.array([face, y + KEY_PITCH * (1 - col), KEYPAD_GRID_Z + KEY_PITCH * (1 - row)])
            for row in range(3) for col in range(3)]


def place_keypad(model: mujoco.MjModel, data: mujoco.MjData, x: float, y: float,
                 layout) -> None:
    """Put the keypad at (x, y) and the keys in its slots: layout[i] is the digit in slot i."""
    data.mocap_pos[model.body(OBJECT_PREFIX + "keypad").mocapid[0]] = [x, y, KEYPAD_GRID_Z]
    for slot, digit in zip(keypad_slots(x, y), layout):
        data.mocap_pos[model.body(f"key_{digit}_mount").mocapid[0]] = slot
    mujoco.mj_forward(model, data)

"""The task library. Level 1: reach, pick_place, press_button, open_drawer, press_key.
Level 2: stack, sort, stow, type_code."""
import mujoco
import numpy as np

from ..core.scene import (
    BIN_HEIGHT, BIN_INNER, BUTTON_PRESSED, DRAWER_FLOOR_Z, DRAWER_INNER_X, DRAWER_INNER_Y,
    KEY_DEPTH, KEY_DIGITS, KEY_PRESSED, KEY_RELEASED, KEY_TRAVEL, OBJECT_PREFIX, ObjectSpec,
    add_bin, add_button, add_drawer, add_keypad, add_target_pad, place_keypad, scene_spec,
)
from ..skills.motion import wrap_angle
from ..skills.sensing import true_contacts, true_tcp
from ..tools import Dispatcher
from ..world import World
from .base import Evaluation, Task, call, grasp_yaw, sample_xy, yaw_quat_z

COLOURS = {"red": (0.85, 0.2, 0.2, 1), "blue": (0.2, 0.35, 0.85, 1),
           "yellow": (0.95, 0.8, 0.15, 1), "green": (0.2, 0.75, 0.3, 1)}
CUBE_HALF = 0.025


def _cubes(*colours: str) -> list[ObjectSpec]:
    return [ObjectSpec(f"{c}_cube", rgba=COLOURS[c]) for c in colours]


def _place_cubes(world: World, names: list[str], rng, avoid=()) -> None:
    for name, xy in zip(names, sample_xy(rng, len(names), 0.10, avoid=list(avoid))):
        world.set_object_pose(name, [*xy, CUBE_HALF], yaw_quat_z(rng.uniform(-np.pi, np.pi)))


def _closeness(distance: float, scale: float) -> float:
    """1 at distance 0, falling linearly to 0 at `scale` or beyond."""
    return float(max(0.0, 1.0 - distance / scale)) if scale > 0 else 0.0


def _object_yaw(world: World, name: str) -> float:
    _, quat = world.object_pose(name)
    mat = np.empty(9)
    mujoco.mju_quat2Mat(mat, quat)
    return float(np.arctan2(mat[3], mat[0]))


def _set_mocap(world: World, body: str, pos) -> None:
    world.data.mocap_pos[world.plant.body(body).mocapid[0]] = pos
    mujoco.mj_forward(world.plant, world.data)


class ReachTask(Task):
    """Hover above one of three cubes. Tests grounding (which cube?) and placement precision."""
    name = "reach"
    max_seconds = 30.0
    HOVER = 0.05
    TOLERANCE = 0.015

    @classmethod
    def build_spec(cls):
        return scene_spec(_cubes("red", "blue", "yellow"))

    def reset(self, world, rng):
        cubes = [f"{c}_cube" for c in ("red", "blue", "yellow")]
        _place_cubes(world, cubes, rng)
        self.target = str(rng.choice(cubes))
        self.start = world.object_pose(self.target)[0].copy()
        colour = self.target.split("_")[0]
        self.instruction = (f"Move the gripper so its fingertip centre is {self.HOVER * 100:.0f} cm "
                            f"directly above the centre of the top face of the {colour} cube, "
                            "without touching any cube, then call done.")

    def goal(self, world) -> np.ndarray:
        return world.object_pose(self.target)[0] + [0, 0, CUBE_HALF + self.HOVER]

    def evaluate(self, world):
        error = float(np.linalg.norm(true_tcp(world)[0] - self.goal(world)))
        moved = float(np.linalg.norm(world.object_pose(self.target)[0] - self.start))
        ok = error <= self.TOLERANCE and moved <= 0.02
        return Evaluation(ok, f"fingertip {error * 100:.1f} cm from the goal (limit "
                              f"{self.TOLERANCE * 100:.1f}); target cube moved {moved * 100:.1f} cm",
                          {"error_cm": round(error * 100, 2), "cube_moved_cm": round(moved * 100, 2)},
                          progress=1.0 if ok else _closeness(error, 0.10))

    def oracle(self, d: Dispatcher):
        x, y, z = self.goal(d.ctx.world)
        call(d, ("move_to", dict(x=x, y=y, z=z)))


class PickPlaceTask(Task):
    """Put one of two cubes on a target pad."""
    name = "pick_place"
    max_seconds = 60.0
    PAD_RADIUS = 0.05
    TOLERANCE = 0.04  # cube centre within 4 cm of the pad centre

    @classmethod
    def build_spec(cls):
        spec = scene_spec(_cubes("red", "blue"))
        add_target_pad(spec, "green_pad", (0.2, 0.75, 0.3, 0.8), cls.PAD_RADIUS)
        return spec

    def reset(self, world, rng):
        pad = sample_xy(rng, 1, 0.0)[0]
        _set_mocap(world, OBJECT_PREFIX + "green_pad", [*pad, 0.001])
        _place_cubes(world, ["red_cube", "blue_cube"], rng, avoid=[pad])  # >= 10 cm from the pad
        self.target = str(rng.choice(["red_cube", "blue_cube"]))
        self.start_dist = float(np.linalg.norm(world.object_pose(self.target)[0][:2] - pad))
        colour = self.target.split("_")[0]
        self.instruction = (f"Pick up the {colour} cube and put it down on the green pad "
                            "(green_pad), then call done.")

    def involved(self):
        return {self.target, "green_pad"}

    def evaluate(self, world):
        pos = world.object_pose(self.target)[0]
        pad = world.object_pose("green_pad")[0]
        dist = float(np.linalg.norm(pos[:2] - pad[:2]))
        resting = pos[2] < CUBE_HALF + 0.01
        released = not any(c.other == self.target for c in true_contacts(world))
        ok = dist <= self.TOLERANCE and resting and released
        # Fraction of the way from where the cube started to the pad; a cube still in the hand
        # or balanced on something counts at most three quarters.
        progress = 1.0 if ok else min(_closeness(dist, self.start_dist), 1.0 if resting and released else 0.75)
        return Evaluation(ok, f"cube centre {dist * 100:.1f} cm from the pad centre (limit "
                              f"{self.TOLERANCE * 100:.0f}); resting={resting}; released={released}",
                          {"distance_cm": round(dist * 100, 2), "resting": bool(resting),
                           "released": bool(released)}, progress=progress)

    def oracle(self, d: Dispatcher):
        world = d.ctx.world
        cube = world.object_pose(self.target)[0]
        pad = world.object_pose("green_pad")[0]
        yaw = float(np.degrees(grasp_yaw(_object_yaw(world, self.target))))
        call(d, ("gripper", {"action": "open"}),
             ("move_to", dict(x=cube[0], y=cube[1], z=0.15, yaw_deg=yaw)))
        call(d, ("move_to", dict(x=cube[0], y=cube[1], z=CUBE_HALF, yaw_deg=yaw)),
             ("gripper", {"action": "close"}),
             ("move_to", dict(x=cube[0], y=cube[1], z=0.15, yaw_deg=yaw)))
        call(d, ("move_to", dict(x=pad[0], y=pad[1], z=0.15, yaw_deg=yaw)),
             ("move_to", dict(x=pad[0], y=pad[1], z=CUBE_HALF + 0.003, yaw_deg=yaw, speed=0.05)))
        call(d, ("gripper", {"action": "open"}), ("move_relative", {"dz": 0.1}))


class PressButtonTask(Task):
    """Press one of two buttons, and not the other."""
    name = "press_button"
    max_seconds = 30.0
    BUTTONS = ("red_button", "blue_button")

    @classmethod
    def build_spec(cls):
        spec = scene_spec()
        for name in cls.BUTTONS:
            add_button(spec, name, COLOURS[name.split("_")[0]])
        return spec

    def reset(self, world, rng):
        for name, xy in zip(self.BUTTONS, sample_xy(rng, 2, 0.15, x=(0.40, 0.62), y=(-0.22, 0.22))):
            _set_mocap(world, f"{name}_housing", [*xy, 0.02])
        self.joints = {b: world.plant.joint(f"{b}_press").qposadr[0] for b in self.BUTTONS}
        self.max_travel = {b: 0.0 for b in self.BUTTONS}
        self.target = str(rng.choice(self.BUTTONS))
        self.instruction = (f"Press the {self.target.split('_')[0]} button ({self.target}) "
                            "all the way down, without pressing the other button, then call done.")

    def monitor(self, world):
        for b, adr in self.joints.items():
            self.max_travel[b] = max(self.max_travel[b], float(world.data.qpos[adr]))

    def fixture_states(self, world):
        return {f"{b}_travel_mm": round(float(world.data.qpos[adr]) * 1000, 1)
                for b, adr in self.joints.items()}

    def pressed(self, button: str) -> bool:
        return self.max_travel[button] >= BUTTON_PRESSED

    def evaluate(self, world):
        other = next(b for b in self.BUTTONS if b != self.target)
        ok = self.pressed(self.target) and not self.pressed(other)
        progress = min(1.0, self.max_travel[self.target] / BUTTON_PRESSED) * (0.5 if self.pressed(other) else 1.0)
        return Evaluation(ok, f"{self.target} max travel {self.max_travel[self.target] * 1000:.1f} mm, "
                              f"{other} {self.max_travel[other] * 1000:.1f} mm (a press is "
                              f">= {BUTTON_PRESSED * 1000:.0f} mm)",
                          {f"{b}_travel_mm": round(t * 1000, 1) for b, t in self.max_travel.items()},
                          progress=progress)

    def oracle(self, d: Dispatcher):
        cap = d.ctx.world.object_pose(self.target)[0]
        call(d, ("gripper", {"action": "close"}),
             ("move_to", dict(x=cap[0], y=cap[1], z=cap[2] + 0.08)))
        call(d, ("move_to", dict(x=cap[0], y=cap[1], z=cap[2] - 0.02, speed=0.05)))
        call(d, ("move_relative", {"dz": 0.08}))


class OpenDrawerTask(Task):
    """Pull a drawer open by its handle."""
    name = "open_drawer"
    max_seconds = 45.0
    REQUIRED = 0.10

    @classmethod
    def build_spec(cls):
        spec = scene_spec()
        add_drawer(spec)
        return spec

    def reset(self, world, rng):
        _set_mocap(world, "cabinet", [rng.uniform(0.70, 0.76), rng.uniform(-0.15, 0.15), 0.09])
        self.slide = world.plant.joint("drawer_slide").qposadr[0]
        self.instruction = ("Open the drawer by at least 10 cm by pulling its handle (drawer_handle) "
                            "towards the robot, then call done. The handle is a tab sticking out "
                            "of the drawer front.")

    def fixture_states(self, world):
        return {"drawer_opening_cm": round(float(world.data.qpos[self.slide]) * 100, 1)}

    def evaluate(self, world):
        opening = float(world.data.qpos[self.slide])
        ok = opening >= self.REQUIRED
        return Evaluation(ok, f"drawer open {opening * 100:.1f} cm (need {self.REQUIRED * 100:.0f})",
                          {"opening_cm": round(opening * 100, 2)},
                          progress=min(1.0, max(0.0, opening) / self.REQUIRED))

    def involved(self):
        return {"drawer_handle"}

    def oracle(self, d: Dispatcher):
        handle = d.ctx.world.object_pose("drawer_handle")[0]
        grasp = handle + [-0.02, 0, 0]  # near the tip, clear of the drawer front
        call(d, ("gripper", {"action": "open"}),
             ("move_to", dict(x=grasp[0], y=grasp[1], z=grasp[2] + 0.1, yaw_deg=90)))
        call(d, ("move_to", dict(x=grasp[0], y=grasp[1], z=grasp[2], yaw_deg=90, speed=0.08)),
             ("gripper", {"action": "close"}))
        call(d, ("move_relative", dict(dx=-0.15, speed=0.08)))
        call(d, ("gripper", {"action": "open"}), ("move_relative", {"dz": 0.1}))




TRAVEL_Z = 0.22  # oracles move sideways at this height: above the drawer front, bins and towers


def _clear_yaw(world: World, name: str, near_yaw: float, radius: float = 0.2) -> float:
    """Among the yaws aligned with an object's faces, the one whose long side points away from
    nearby things: the hand reaches 10.3 cm either way along the finger-closing axis but only
    3.2 cm across, so its long side is what hits a neighbouring tower or drawer. With nothing
    near, the aligned yaw closest to `near_yaw`."""
    here = world.object_pose(name)[0][:2]
    bearings = []
    for other in world.objects:
        if other == name or "pad" in other:  # flat pads are nothing to bump into
            continue
        offset = world.object_pose(other)[0][:2] - here
        if np.linalg.norm(offset) < radius:
            bearings.append(np.arctan2(offset[1], offset[0]))
    base = grasp_yaw(_object_yaw(world, name), near_yaw)
    if not bearings:
        return base
    exposure = lambda yaw: max(abs(np.cos(yaw - b)) for b in bearings)
    return min((base, wrap_angle(base + np.pi / 2)), key=lambda c: exposure(c) - (0.05 if c == base else 0.0))


def _pick(d: Dispatcher, name: str, near_yaw: float = 0.0, lift: float = TRAVEL_Z) -> float:
    """Oracle helper: travel above an object, descend straight onto it, grasp, and lift it
    straight up; returns the gripper yaw (deg)."""
    world = d.ctx.world
    x, y, z = world.object_pose(name)[0]
    yaw = float(np.degrees(_clear_yaw(world, name, near_yaw)))
    call(d, ("gripper", {"action": "open"}), ("move_to", dict(x=x, y=y, z=lift, yaw_deg=yaw)))
    call(d, ("move_to", dict(x=x, y=y, z=z, yaw_deg=yaw, speed=0.08)),
         ("gripper", {"action": "close"}), ("move_to", dict(x=x, y=y, z=lift, yaw_deg=yaw)))
    return yaw


def _put(d: Dispatcher, x: float, y: float, z: float, yaw: float, approach: float = TRAVEL_Z,
         release_cm: float | None = None) -> None:
    """Oracle helper: carry the held object above (x, y), lower it to z (stopping on contact),
    release (opening fully, or to `release_cm` where walls are close), and back off straight up
    to the approach height (moving sideways while low drags the fingers through whatever is
    around)."""
    call(d, ("move_to", dict(x=x, y=y, z=approach, yaw_deg=yaw)),
         ("move_to", dict(x=x, y=y, z=z, yaw_deg=yaw, speed=0.05)))
    release = {"action": "open"} if release_cm is None else {"action": "open", "width_cm": release_cm}
    call(d, ("gripper", release), ("move_to", dict(x=x, y=y, z=approach, yaw_deg=yaw)))


def _released(world: World, names) -> bool:
    return not any(c.other in names for c in true_contacts(world))


class StackTask(Task):
    """Build a three-cube tower in a given order. Level 2: sequencing and precise placement."""
    name = "stack"
    level = 2
    max_seconds = 120.0
    CUBES = ("red_cube", "blue_cube", "yellow_cube")
    XY_TOLERANCE = 0.02

    @classmethod
    def build_spec(cls):
        return scene_spec(_cubes("red", "blue", "yellow"))

    def reset(self, world, rng):
        _place_cubes(world, list(self.CUBES), rng)
        self.order = [str(c) for c in rng.permutation(self.CUBES)]  # bottom, middle, top
        b, m, t = (c.split("_")[0] for c in self.order)
        self.instruction = (f"Stack the three cubes into one tower: the {b} cube at the bottom, "
                            f"the {m} cube on it, and the {t} cube on top. Then call done.")

    def evaluate(self, world):
        pos = [world.object_pose(c)[0] for c in self.order]
        offsets = [float(np.linalg.norm(pos[i + 1][:2] - pos[i][:2])) for i in range(2)]
        heights_ok = all(abs(p[2] - (CUBE_HALF + 2 * CUBE_HALF * i)) < 0.01 for i, p in enumerate(pos))
        released = _released(world, self.order)
        ok = heights_ok and max(offsets) <= self.XY_TOLERANCE and released
        # One point per cube correctly placed on the one below it (two placements in all).
        placed = sum(abs(pos[i][2] - (CUBE_HALF + 2 * CUBE_HALF * i)) < 0.01
                     and offsets[i - 1] <= self.XY_TOLERANCE for i in (1, 2))
        return Evaluation(ok, f"tower levels at {[round(float(p[2]) * 100, 1) for p in pos]} cm "
                              f"(need 2.5/7.5/12.5); offsets {[round(o * 100, 1) for o in offsets]} cm "
                              f"(limit {self.XY_TOLERANCE * 100:.0f}); released={released}",
                          {"max_offset_cm": round(max(offsets) * 100, 2), "levels_ok": heights_ok,
                           "released": released}, progress=1.0 if ok else placed / 2)

    def involved(self):
        return set(self.order)

    def oracle(self, d: Dispatcher):
        world = d.ctx.world
        for cube, onto in ((self.order[1], self.order[0]), (self.order[2], self.order[1])):
            yaw = _pick(d, cube)
            x, y, z = world.object_pose(onto)[0]
            _put(d, x, y, z + 2 * CUBE_HALF + 0.003, yaw)


class SortTask(Task):
    """Put two red and two blue cubes into the bin of their colour. Level 2: several objects."""
    name = "sort"
    level = 2
    max_seconds = 180.0
    CUBES = ("red_cube_1", "red_cube_2", "blue_cube_1", "blue_cube_2")
    BINS = {"red": "red_bin", "blue": "blue_bin"}

    @classmethod
    def build_spec(cls):
        spec = scene_spec([ObjectSpec(c, rgba=COLOURS[c.split("_")[0]]) for c in cls.CUBES])
        for colour, name in cls.BINS.items():
            add_bin(spec, name, COLOURS[colour])
        return spec

    def reset(self, world, rng):
        # bins on the far side of the table; cubes nearer the robot, with room for open fingers
        # (fingers jammed against a bin wall when a cube spawned right next to it)
        bins = sample_xy(rng, 2, 0.24, x=(0.56, 0.66), y=(-0.25, 0.25))
        for name, xy in zip(self.BINS.values(), bins):
            _set_mocap(world, OBJECT_PREFIX + name, [*xy, 0.0])
        for name, xy in zip(self.CUBES, sample_xy(rng, 4, 0.10, x=(0.30, 0.52), avoid=bins,
                                                  avoid_dist=0.17)):
            world.set_object_pose(name, [*xy, CUBE_HALF], yaw_quat_z(rng.uniform(-np.pi, np.pi)))
        self.instruction = ("Put each red cube in the red bin and each blue cube in the blue bin, "
                            "then call done.")

    def _in_bin(self, world, cube: str) -> bool:
        bin_pos = world.object_pose(self.BINS[cube.split("_")[0]])[0]
        p = world.object_pose(cube)[0]
        inside = np.all(np.abs(p[:2] - bin_pos[:2]) < BIN_INNER / 2)
        return bool(inside and p[2] < BIN_HEIGHT)

    def evaluate(self, world):
        inside = {c: self._in_bin(world, c) for c in self.CUBES}
        n = sum(inside.values())
        return Evaluation(n == len(self.CUBES), f"{n}/{len(self.CUBES)} cubes in their bin "
                          f"({', '.join(c for c, ok in inside.items() if not ok) or 'all sorted'})",
                          {"sorted": n}, progress=n / len(self.CUBES))

    def involved(self):
        return set(self.CUBES) | set(self.BINS.values())

    def oracle(self, d: Dispatcher):
        world = d.ctx.world
        slots = {colour: [-0.035, 0.035] for colour in self.BINS}
        for cube in self.CUBES:
            colour = cube.split("_")[0]
            _pick(d, cube, near_yaw=np.pi / 2)  # fingers along y, cubes go side by side in x
            bx, by, _ = world.object_pose(self.BINS[colour])[0]
            # Place at yaw 90 exactly: the cube is then square to the bin (a cube left at its
            # random yaw is up to 7 cm across and lands on its neighbour). The fingers then
            # close along y, so the two slots sit side by side along x, across the fingers,
            # which never reach the other cube. The cube is lowered onto the floor rather than
            # dropped (dropped cubes bounce and tilt) and released by opening only to 6.5 cm:
            # fully open, the fingers jam against the walls 7 cm from the centre.
            _put(d, bx + slots[colour].pop(0), by, 0.004 + CUBE_HALF + 0.003, 90.0, release_cm=6.5)


class StowTask(Task):
    """Open the drawer, put a cube inside, close the drawer. Level 2: a mechanism + an object."""
    name = "stow"
    level = 2
    max_seconds = 180.0
    CLOSED = 0.01

    @classmethod
    def build_spec(cls):
        spec = scene_spec(_cubes("red", "blue"))
        add_drawer(spec)
        return spec

    def reset(self, world, rng):
        cabinet_y = rng.uniform(-0.12, 0.12)
        _set_mocap(world, "cabinet", [rng.uniform(0.72, 0.76), cabinet_y, 0.09])
        # Cubes go beside the drawer's path, clear of the open drawer (15 cm either side of its
        # centre line) by more than the hand's 10.3 cm reach along its fingers: opened, the
        # handle reaches x ~0.37, and a hand picking a cube in its path would hit it.
        while True:
            spots = sample_xy(rng, 2, 0.12, x=(0.30, 0.50), y=(-0.32, 0.32))
            if all(abs(p[1] - cabinet_y) >= 0.27 for p in spots):
                break
        for name, xy in zip(("red_cube", "blue_cube"), spots):
            world.set_object_pose(name, [*xy, CUBE_HALF], yaw_quat_z(rng.uniform(-np.pi, np.pi)))
        self.slide = world.plant.joint("drawer_slide").qposadr[0]
        self.target = str(rng.choice(["red_cube", "blue_cube"]))
        self.instruction = (f"Put the {self.target.split('_')[0]} cube inside the drawer and close "
                            "the drawer, then call done. The drawer handle (drawer_handle) is a tab "
                            "sticking out of the drawer front.")

    def fixture_states(self, world):
        return {"drawer_opening_cm": round(float(world.data.qpos[self.slide]) * 100, 1)}

    def _in_drawer(self, world) -> bool:
        drawer = world.data.body("drawer")
        local = drawer.xmat.reshape(3, 3).T @ (world.object_pose(self.target)[0] - drawer.xpos)
        return bool(DRAWER_INNER_X[0] < local[0] < DRAWER_INNER_X[1]
                    and abs(local[1]) < DRAWER_INNER_Y and local[2] < DRAWER_FLOOR_Z + 0.06)

    def evaluate(self, world):
        inside, opening = self._in_drawer(world), float(world.data.qpos[self.slide])
        ok = inside and opening <= self.CLOSED
        progress = 0.5 * inside + 0.5 * (1.0 if opening <= self.CLOSED else 0.0)
        return Evaluation(ok, f"cube in drawer={inside}; drawer open {round(opening * 100, 1) + 0.0:.1f} cm "
                              f"(closed means <= {self.CLOSED * 100:.0f})",
                          {"in_drawer": inside, "opening_cm": round(opening * 100, 2)}, progress=progress)

    def involved(self):
        return {self.target, "drawer_handle"}

    def oracle(self, d: Dispatcher):
        world = d.ctx.world

        def grasp_handle():
            hx, hy, hz = world.object_pose("drawer_handle")[0] + [-0.02, 0, 0]
            call(d, ("gripper", {"action": "open"}),
                 ("move_to", dict(x=hx, y=hy, z=hz + 0.1, yaw_deg=90)))
            call(d, ("move_to", dict(x=hx, y=hy, z=hz, yaw_deg=90, speed=0.08)),
                 ("gripper", {"action": "close"}))

        grasp_handle()
        call(d, ("move_relative", dict(dx=-0.16, speed=0.08)))
        call(d, ("gripper", {"action": "open"}), ("move_relative", {"dz": 0.1}))
        _pick(d, self.target)  # near yaw 0: the hand's long side parallel to the drawer
        # Place at yaw 90: the hand is ~21 cm long along the finger-closing axis but only 6.5 cm
        # across, so with its long side along y it fits between the drawer front (4 cm away
        # here) and the cabinet front.
        drawer = world.data.body("drawer")
        spot = drawer.xpos + drawer.xmat.reshape(3, 3) @ np.array([-0.035, 0.0, DRAWER_FLOOR_Z])
        _put(d, spot[0], spot[1], spot[2] + CUBE_HALF + 0.004, 90.0)
        # Close by pushing the handle's tip with closed fingertips: the handle sticks out 7 cm,
        # so the hand (about 10 cm deep) stays clear of the cabinet; the push stops on contact
        # when the drawer reaches its closed end stop.
        hx, hy, hz = world.object_pose("drawer_handle")[0]
        tip = hx - 0.035
        call(d, ("gripper", {"action": "close"}),
             ("move_to", dict(x=tip - 0.03, y=hy, z=hz + 0.1, yaw_deg=90)))
        call(d, ("move_to", dict(x=tip - 0.03, y=hy, z=hz, yaw_deg=90, speed=0.08)))
        # Push exactly the gap plus the opening (minus 4 mm): meeting the handle is an impact the
        # contact check would stop on, so it is off, and the push must not overshoot.
        opening = float(world.data.qpos[self.slide])
        call(d, ("move_relative", dict(dx=0.03 + opening - 0.004, speed=0.05,
                                       stop_on_contact=False)))
        call(d, ("move_relative", {"dz": 0.1}))


class _KeypadTask(Task):
    """The keypad scene, shared by the keypad tasks: a vertical panel facing the robot with a
    3 x 3 grid of spring keys labelled 1-9. Key presses are counted as they happen: a key
    counts when it goes in KEY_PRESSED, and must come back out to KEY_RELEASED before it can
    count again."""
    SHUFFLED: bool = False  # digits in a random order instead of the phone layout (1 2 3 on top)
    GEOMETRY = ("The keypad is a vertical plate on a post, facing the robot, its keys about 0.7 m "
                "above the table. Each key is a 2.8 cm square that sticks 1.6 cm out of the plate "
                "towards the robot and goes in 1.2 cm when pushed "
                "(towards +x); a press counts once the key is 8 mm in, and the key must come back "
                "out before it counts again. Press keys with the fingers closed and the gripper "
                "pointing forward (move_to with pointing \"forward\"); the arm starts that way, "
                "with its fingertips about 20 cm in front of the keypad.")
    view = {"lookat": (0.45, 0.0, 0.55), "distance": 1.5, "azimuth": 25.0, "elevation": -15.0}
    # Pointing forward, fingertips at (0.40, 0, 0.74): the home posture's wrist would be where
    # the keypad stands.
    start_deg = (0.0, -52.1, 0.0, -145.5, 0.0, 183.4, 45.0)

    @classmethod
    def build_spec(cls):
        spec = scene_spec()
        add_keypad(spec)
        return spec

    def _lay_out(self, world: World, rng) -> None:
        x, y = rng.uniform(0.60, 0.66), rng.uniform(-0.08, 0.08)
        self.layout = ([int(d) for d in rng.permutation(KEY_DIGITS)] if self.SHUFFLED
                       else list(KEY_DIGITS))
        place_keypad(world.plant, world.data, x, y, self.layout)
        self.joints = {d: world.plant.joint(f"key_{d}_press").qposadr[0] for d in KEY_DIGITS}
        self.max_travel = {d: 0.0 for d in KEY_DIGITS}
        self.down = {d: False for d in KEY_DIGITS}
        self.typed: list[int] = []

    def monitor(self, world):
        for digit, adr in self.joints.items():
            travel = float(world.data.qpos[adr])
            self.max_travel[digit] = max(self.max_travel[digit], travel)
            if not self.down[digit] and travel >= KEY_PRESSED:
                self.down[digit] = True
                self.typed.append(digit)
            elif self.down[digit] and travel <= KEY_RELEASED:
                self.down[digit] = False

    def fixture_states(self, world):
        states = {"typed_so_far": "".join(map(str, self.typed)) or "nothing"}
        for digit, adr in self.joints.items():
            travel = float(world.data.qpos[adr])
            if travel >= 0.0005:
                states[f"key_{digit}_travel_mm"] = round(travel * 1000, 1)
        return states

    def _type(self, d: Dispatcher, digits) -> None:
        """Oracle helper: fingers closed, gripper pointing forward; for each key, line up 5 cm in
        front of it, push past the bottom of its travel (the move stops on the contact when the
        key bottoms out), and back off so it springs out again."""
        world = d.ctx.world
        call(d, ("gripper", {"action": "close"}))
        for digit in digits:
            cap = world.object_pose(f"key_{digit}")[0]
            face = float(cap[0]) - KEY_DEPTH / 2
            line = dict(y=float(cap[1]), z=float(cap[2]), pointing="forward")
            call(d, ("move_to", dict(x=face - 0.05, **line)))
            # The fingertips reach 0.9 cm past the fingertip centre: aim 1.2 cm past bottom-out.
            call(d, ("move_to", dict(x=face - 0.009 + KEY_TRAVEL + 0.012, speed=0.05, **line)))
            call(d, ("move_to", dict(x=face - 0.05, **line)))


class PressKeyTask(_KeypadTask):
    """Press one key of the keypad, and no other."""
    name = "press_key"
    max_seconds = 40.0

    def reset(self, world, rng):
        self._lay_out(world, rng)
        self.target = int(rng.choice(KEY_DIGITS))
        self.instruction = (f"Press the {self.target} key on the keypad (key_{self.target}) once, "
                            f"without pressing any other key, then call done. {self.GEOMETRY}")

    def evaluate(self, world):
        others = [d for d in KEY_DIGITS if d != self.target and self.max_travel[d] >= KEY_PRESSED]
        ok = self.max_travel[self.target] >= KEY_PRESSED and not others
        progress = min(1.0, self.max_travel[self.target] / KEY_PRESSED) * (0.5 if others else 1.0)
        return Evaluation(ok, f"key {self.target} max travel {self.max_travel[self.target] * 1000:.1f} mm "
                              f"(a press is >= {KEY_PRESSED * 1000:.0f}); other keys pressed: "
                              f"{others or 'none'}",
                          {"target_travel_mm": round(self.max_travel[self.target] * 1000, 1),
                           "typed": "".join(map(str, self.typed))}, progress=progress)

    def involved(self):
        return {f"key_{self.target}"}

    def oracle(self, d: Dispatcher):
        self._type(d, [self.target])


class TypeCodeTask(_KeypadTask):
    """Type a 4-digit code on a keypad whose digits are shuffled. Level 2: a sequence, and the
    keys must be found by their labels."""
    name = "type_code"
    level = 2
    max_seconds = 90.0
    SHUFFLED = True
    LENGTH = 4

    def __init__(self, code: str | None = None):
        """`code`: type this instead of the code drawn from the seed (the keypad layout still
        comes from the seed). E.g. make_episode(TypeCodeTask(code="2468"), seed=0)."""
        super().__init__()
        if code is not None and not (code.isdigit() and "0" not in code):
            raise ValueError(f"a code uses the digits 1-9 only, got {code!r}")
        self.fixed_code = code

    def reset(self, world, rng):
        self._lay_out(world, rng)
        code = [int(rng.choice(KEY_DIGITS))]
        while len(code) < self.LENGTH:  # no digit twice in a row
            code.append(int(rng.choice([d for d in KEY_DIGITS if d != code[-1]])))
        self.code = self.fixed_code or "".join(map(str, code))
        self.instruction = (f"Type the code {self.code} on the keypad: press each key once, in this "
                            "order, and no other key, then call done. The digits on this keypad "
                            "are shuffled (not in the usual phone order), so find each key by its "
                            f"label. {self.GEOMETRY}")

    def evaluate(self, world):
        typed = "".join(map(str, self.typed))
        correct = next((i for i, (a, b) in enumerate(zip(typed, self.code)) if a != b),
                       min(len(typed), len(self.code)))
        return Evaluation(typed == self.code, f"typed {typed or 'nothing'} (code {self.code})",
                          {"typed": typed, "correct_digits": correct}, progress=correct / len(self.code))

    def involved(self):
        return {f"key_{c}" for c in self.code}

    def oracle(self, d: Dispatcher):
        self._type(d, [int(c) for c in self.code])


class PlaygroundTask(Task):
    """Free play for interactive sessions: three cubes, a green pad and a red button. There is
    no single goal, so there is no success check and no oracle."""
    name = "playground"
    max_seconds = 3600.0
    CUBES = ("red_cube", "blue_cube", "yellow_cube")

    @classmethod
    def build_spec(cls):
        spec = scene_spec(_cubes("red", "blue", "yellow"))
        add_target_pad(spec, "green_pad", (0.2, 0.75, 0.3, 0.8))
        add_button(spec, "red_button", COLOURS["red"])
        return spec

    def reset(self, world, rng):
        pad, button = sample_xy(rng, 2, 0.18)
        _set_mocap(world, OBJECT_PREFIX + "green_pad", [*pad, 0.001])
        _set_mocap(world, "red_button_housing", [*button, 0.02])
        _place_cubes(world, list(self.CUBES), rng, avoid=[pad, button])
        self.instruction = "Free play: follow the person's instructions one at a time."
        self.everything = set(world.objects)

    def involved(self):
        return self.everything  # free play: nothing is a bystander

    def evaluate(self, world):
        return Evaluation(False, "free play: there is no success check")

    def oracle(self, d):
        """Free play has no goal to solve."""


# The evaluated tasks, by level: 1 = one step, 2 = several steps.
TASKS: dict[str, type[Task]] = {t.name: t for t in (
    ReachTask, PickPlaceTask, PressButtonTask, OpenDrawerTask, PressKeyTask,  # level 1
    StackTask, SortTask, StowTask, TypeCodeTask,                              # level 2
)}

# Every scene make_episode() can build: the evaluated tasks plus free play.
TASK_CLASSES: dict[str, type[Task]] = {**TASKS, PlaygroundTask.name: PlaygroundTask}

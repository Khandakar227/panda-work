"""The robot tools Claude uses: thin wrappers over the skills, showing only what the tier allows.

Conventions (also stated in the tool descriptions): metres, degrees, newtons, seconds. World
frame: origin at the robot base on the table, x forward, y to the robot's left, z up. The
gripper points down; yaw 0 means the fingers close along x.

Tier summary:
  privileged  15 tools: everything, exact readings, object poses, contacts by name, simulate()
  realistic   13 tools: robot sensors with noise, camera detections with noise, no exact poses
  vision_only  9 tools: camera images, gripper opening, motions; no coordinates in reports
"""
from typing import Literal

import mujoco
import numpy as np
from PIL import Image
from pydantic import Field, field_validator

from ..core.robot import GRIPPER_MAX_WIDTH, joint_limits_deg, pointing_of, quat_yaw
from ..core.scene import OBJECT_PREFIX
from ..skills import motion, perception, sensing
from ..tiers import Tier
from .base import Group, NoArgs, Tool, ToolArgs, ToolContext, ToolError, ToolResult
from .budget import Budget, BudgetLimits
from .dispatch import Dispatcher, ToolCall
from .registry import ToolRegistry

P, R, V = Tier.PRIVILEGED, Tier.REALISTIC, Tier.VISION_ONLY
SIZES = {"small": (320, 240), "medium": (640, 480)}
DETECTION_NOISE = (0.004, np.radians(2.0))  # realistic tier: position std (m), yaw std (rad)


def _m(v) -> list[float]:
    return [round(float(x), 3) for x in v]


def _deg(rad: float) -> float:
    return round(float(np.degrees(rad)), 1)


def _report(ctx: ToolContext, outcome: motion.Outcome, kind: str) -> ToolResult:
    """Outcome report for ACT tools. Vision-only reports carry no coordinates or forces."""
    data = {"status": outcome.status, "stop_reason": outcome.stop_reason,
            "duration_s": round(outcome.duration, 2)}
    d = outcome.details
    if ctx.tier is not V:
        if kind == "tcp" and "achieved" in d:
            data["target"] = _m(d["target"]["position"])
            if d["target"]["pointing"] != "down":
                data["pointing"] = d["target"]["pointing"]
            data["target_yaw_deg"] = _deg(d["target"]["yaw"])
            data["achieved"] = _m(d["achieved"]["position"])
            data["achieved_yaw_deg"] = _deg(d["achieved"]["yaw"])
            data["error_mm"] = round(d["error_m"] * 1000, 1)
        elif kind == "tcp":
            data["target"] = _m(d["target"]["position"])
        if kind == "joints":
            data["achieved_deg"] = [_deg(q) for q in d["achieved"]]
            data["max_error_deg"] = _deg(d["max_error_rad"])
        if "contact" in d:
            c = d["contact"]
            data["sensed"] = (f"wrist force changed by {c['force_change_N']} N" if c["sensed"] == "wrist force"
                              else f"fingertip touch {c['touch_N'][0]} / {c['touch_N'][1]} N")
    if ctx.tier is P and outcome.stop_reason == "contact":
        data["touching"] = [f"{c.robot_part} - {c.other} ({c.force:.1f} N)"
                            for c in sensing.true_contacts(ctx.world)] or ["nothing now"]
    if outcome.hint:
        data["hint"] = outcome.hint
    return ToolResult(data=data)


# ---------------------------------------------------------------------------- PERCEIVE

class GetRobotState(Tool):
    name = "get_robot_state"
    group = Group.PERCEIVE
    description = ("The robot's own state: joint angles (deg), gripper opening (cm) and where it "
                   "believes its fingertip centre is (m), which way the gripper points (down, "
                   "forward, or tilted) and its yaw (deg). In the vision_only tier only the "
                   "gripper opening and pointing are available.")

    def run(self, ctx, args):
        state = sensing.robot_state(ctx.world)
        pointing = pointing_of(state.tcp_quat)
        data = {"gripper_open_cm": round(state.gripper_width * 100, 1),
                "pointing": pointing or "tilted"}
        if ctx.tier is not V:
            data = {"joints_deg": [_deg(q) for q in state.q], **data,
                    "fingertip": _m(state.tcp_pos),
                    "yaw_deg": _deg(quat_yaw(state.tcp_quat, pointing or "down"))}
        if ctx.tier is P:
            data["fingertip_true"] = _m(sensing.true_tcp(ctx.world)[0])
        return ToolResult(data=data)


class WristFTArgs(ToolArgs):
    window_s: float = Field(0.0, ge=0.0, le=5.0,
                            description="0: current reading; >0: peak and mean over the last "
                                        "window_s seconds")


class ReadWristFT(Tool):
    name = "read_wrist_ft"
    group = Group.PERCEIVE
    tiers = frozenset({P, R})
    Args = WristFTArgs
    description = ("External force on the hand (N, world frame), from the wrist force/torque "
                   "sensor with the hand's own weight removed. About 0 when touching nothing; "
                   "about the weight of a held object (negative z) when holding one.")

    def run(self, ctx, args: WristFTArgs):
        world = ctx.world
        if args.window_s == 0:
            force = sensing.external_force(world)
            return ToolResult(data={"force_N": np.round(force, 1),
                                    "magnitude_N": round(float(np.linalg.norm(force)), 1),
                                    "held_load_kg": round(max(0.0, -force[2] / 9.81), 3)})
        joints = {m.header.step: m.data["q"] for m in world.bus.window("/joints", args.window_s)}
        forces = np.array([sensing.external_force(world, m, joints[m.header.step])
                           for m in world.bus.window("/wrist_ft", args.window_s)
                           if m.header.step in joints])
        mags = np.linalg.norm(forces, axis=1)
        return ToolResult(data={"window_s": args.window_s, "samples": len(forces),
                                "peak_N": round(float(mags.max()), 1),
                                "mean_force_N": np.round(forces.mean(axis=0), 1)})


class GetContacts(Tool):
    name = "get_contacts"
    group = Group.PERCEIVE
    tiers = frozenset({P, R})
    description = ("What the hand is touching. Privileged: every contact between the robot and "
                   "anything else, by name. Realistic: fingertip pad forces and the external "
                   "wrist force only.")

    def run(self, ctx, args):
        if ctx.tier is P:
            contacts = sensing.true_contacts(ctx.world)
            return ToolResult(text="\n".join(f"{c.robot_part} - {c.other}: {c.force:.1f} N"
                                             for c in contacts) or "no contacts")
        left, right = sensing.touch(ctx.world)
        force = sensing.external_force(ctx.world)
        return ToolResult(data={"left_pad_N": round(left, 2), "right_pad_N": round(right, 2),
                                "wrist_external_force_N": np.round(force, 1)})


class LookArgs(ToolArgs):
    camera: Literal["front", "side", "top", "wrist"] = Field(
        description="front: facing the robot across the table; side: from the robot's right "
                    "(good for heights); top: straight down from 1.6 m above x=0.5; wrist: on "
                    "the hand, looking along the fingers")
    mode: Literal["rgb", "depth", "seg"] = Field("rgb", description="depth: privileged and "
                                                 "realistic; seg (object masks): privileged")
    size: Literal["small", "medium"] = Field("small", description="small 320x240 (~100 tokens), "
                                             "medium 640x480 (~400 tokens)")
    crop: list[float] | None = Field(None, description="optional [x0, y0, x1, y1] region as "
                                     "fractions of the image (0..1), e.g. [0.25, 0.25, 0.75, 0.75]")

    @field_validator("crop")
    @classmethod
    def _valid_crop(cls, crop):
        if crop is not None:
            if len(crop) != 4 or not all(0 <= c <= 1 for c in crop):
                raise ValueError("crop must be 4 fractions between 0 and 1")
            if crop[2] - crop[0] < 0.05 or crop[3] - crop[1] < 0.05:
                raise ValueError("crop must have x1 > x0 and y1 > y0 (at least 0.05 apart)")
        return crop


class Look(Tool):
    name = "look"
    group = Group.PERCEIVE
    Args = LookArgs
    description = "Take a camera image. Images cost tokens: prefer small, and crop to the region you need."

    def run(self, ctx, args: LookArgs):
        allowed = {P: ("rgb", "depth", "seg"), R: ("rgb", "depth"), V: ("rgb",)}[ctx.tier]
        if args.mode not in allowed:
            raise ToolError("mode_not_available", f"{args.mode} images are not available in the "
                            f"{ctx.tier.value} tier", hint=f"use one of: {', '.join(allowed)}")
        image = ctx.world.capture(f"{args.camera}_{args.mode}").data["image"]
        notes = []
        if args.mode == "depth":
            near, far = np.percentile(image, [1, 99])
            notes.append(f"depth: white = {near:.2f} m, black = {far:.2f} m from the camera")
            image = ((1 - np.clip((image - near) / max(far - near, 1e-6), 0, 1)) * 255).astype(np.uint8)
        elif args.mode == "seg":
            image, legend = _colour_segmentation(ctx, image)
            notes.append("objects: " + (", ".join(legend) or "none visible"))
        if args.crop:
            h, w = image.shape[:2]
            x0, y0, x1, y1 = args.crop
            image = image[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        pil = Image.fromarray(image)
        pil.thumbnail(SIZES[args.size])
        out = np.asarray(pil)
        text = f"{args.camera} camera, {args.mode}, {out.shape[1]}x{out.shape[0]} px"
        return ToolResult(text="\n".join([text, *notes]), images=[out])


_SEG_COLOURS = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
                         [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230]], np.uint8)


def _colour_segmentation(ctx, seg) -> tuple[np.ndarray, list[str]]:
    """Objects in distinct colours, robot grey, background black; legend names the colours."""
    names = ["red", "green", "yellow", "blue", "orange", "purple", "cyan", "magenta"]
    model = ctx.world.plant
    out = np.zeros(seg.shape[:2] + (3,), np.uint8)
    geom = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (seg[..., 0] >= 0)
    robot_root = model.body("link0").id
    body = np.where(geom, model.geom_bodyid[np.clip(seg[..., 0], 0, None)], -1)
    out[geom & (model.body_rootid[np.clip(body, 0, None)] == robot_root)] = 128
    legend = []
    for i, name in enumerate(ctx.world.objects):
        mask = geom & (seg[..., 0] == model.geom(OBJECT_PREFIX + name).id)
        if mask.any():
            out[mask] = _SEG_COLOURS[i % len(_SEG_COLOURS)]
            legend.append(f"{name} = {names[i % len(names)]}")
    return out, legend


class GetObjectPoses(Tool):
    name = "get_object_poses"
    group = Group.PERCEIVE
    tiers = frozenset({P})
    description = "Exact pose and size of every object, from the simulator (privileged only)."

    def run(self, ctx, args):
        lines = [f"{p.name}: center {_m(p.position)}, yaw {_deg(p.yaw)} deg, size "
                 f"{' x '.join(f'{d * 100:.1f}' for d in p.dimensions)} cm (w x d x h)"
                 for p in perception.true_object_poses(ctx.world)]
        return ToolResult(text="\n".join(lines) or "no objects")


class GetFixtureStates(Tool):
    name = "get_fixture_states"
    group = Group.PERCEIVE
    tiers = frozenset({P})
    description = ("Exact state of the scene's mechanisms (button travel, drawer opening, ...), "
                   "from the simulator (privileged only).")

    def run(self, ctx, args):
        states = ctx.task.fixture_states(ctx.world) if ctx.task is not None else {}
        return ToolResult(data=states) if states else ToolResult(text="no mechanisms in this scene")


class DetectArgs(ToolArgs):
    camera: Literal["front", "side", "top", "wrist"] = Field("top", description="camera to detect with")


class DetectObjects(Tool):
    name = "detect_objects"
    group = Group.PERCEIVE
    tiers = frozenset({P, R})
    Args = DetectArgs
    description = ("Find visible objects in a camera's view (segmentation + depth). Reports the "
                   "centre of each object's visible top surface, its height, footprint yaw (deg, "
                   "ambiguous by 90 for square objects) and footprint size. Hidden objects are "
                   "missed. Estimates are noisy in the realistic tier (about 4 mm, 2 deg).")

    def run(self, ctx, args: DetectArgs):
        noise = None if ctx.tier is P else DETECTION_NOISE
        found, missed = perception.detect_objects(ctx.world, args.camera, noise,
                                                  ctx.world.rng("detect"))
        lines = [f"{d.name}: top centre {_m([*d.center_xy, d.top_z])}, yaw {_deg(d.yaw)} deg, "
                 f"footprint {d.footprint[0] * 100:.1f} x {d.footprint[1] * 100:.1f} cm, "
                 f"{d.pixels} px" for d in found]
        if missed:
            lines.append("not detected (hidden or out of view): " + ", ".join(missed))
        return ToolResult(text="\n".join(lines) or "nothing detected")


# ---------------------------------------------------------------------------- ACT

class MoveToArgs(ToolArgs):
    x: float = Field(description="forward from the robot base (m)")
    y: float = Field(description="to the robot's left (m)")
    z: float = Field(description="height of the fingertip centre above the table (m)")
    yaw_deg: float = Field(0.0, ge=-180, le=180,
                           description="gripper rotation about the pointing direction; down: 0 "
                                       "= fingers close along x, 90 = along y; forward: 0 = "
                                       "fingers close along y (wrist camera upright), 90 = "
                                       "up-down")
    pointing: Literal["down", "forward"] = Field(
        "down", description="down: fingertips point at the table; forward: fingertips point "
                            "along +x (away from the robot), e.g. to push keys on a vertical "
                            "panel facing the robot; comfortable with the fingertips about "
                            "0.65 m or more above the table")
    speed: float = Field(0.15, gt=0, le=0.5, description="m/s")
    tolerance_mm: float = Field(2.0, ge=0.5, le=20.0)
    stop_on_contact: bool = Field(True, description="stop as soon as an unexpected contact is felt")


class MoveTo(Tool):
    name = "move_to"
    group = Group.ACT
    Args = MoveToArgs
    timeout_s = 60.0
    description = ("Move the fingertip centre (between the fingertips) to (x, y, z) with the "
                   "gripper pointing down (default) or forward, in a straight line when "
                   "possible. Stops early on contact. Returns status reached / partial / failed "
                   "with the reason.")

    def run(self, ctx, args: MoveToArgs):
        outcome = motion.move_tcp(ctx.world, [args.x, args.y, args.z], np.radians(args.yaw_deg),
                                  args.speed, args.tolerance_mm / 1000, args.stop_on_contact,
                                  ctx.check_deadline, args.pointing)
        return _report(ctx, outcome, "tcp")


class MoveRelativeArgs(ToolArgs):
    dx: float = Field(0.0, ge=-0.5, le=0.5, description="m")
    dy: float = Field(0.0, ge=-0.5, le=0.5, description="m")
    dz: float = Field(0.0, ge=-0.5, le=0.5, description="m")
    dyaw_deg: float = Field(0.0, ge=-180, le=180)
    frame: Literal["world", "gripper"] = Field(
        "world", description="world: dx/dy/dz along world axes; gripper: along the gripper's "
                             "axes (z = along the fingers, towards the fingertips)")
    speed: float = Field(0.1, gt=0, le=0.5, description="m/s")
    stop_on_contact: bool = True


class MoveRelative(Tool):
    name = "move_relative"
    group = Group.ACT
    Args = MoveRelativeArgs
    timeout_s = 60.0
    description = ("Move the fingertip centre by an offset from where the robot believes it is "
                   "now, keeping the gripper pointing the way it does (down or forward; a tilted "
                   "gripper is turned to point down).")

    def run(self, ctx, args: MoveRelativeArgs):
        state = sensing.robot_state(ctx.world)
        offset = np.array([args.dx, args.dy, args.dz])
        if args.frame == "gripper":
            mat = np.empty(9)
            mujoco.mju_quat2Mat(mat, state.tcp_quat)
            offset = mat.reshape(3, 3) @ offset
        pointing = pointing_of(state.tcp_quat) or "down"
        outcome = motion.move_tcp(ctx.world, state.tcp_pos + offset,
                                  quat_yaw(state.tcp_quat, pointing) + np.radians(args.dyaw_deg),
                                  args.speed, 0.002, args.stop_on_contact, ctx.check_deadline,
                                  pointing)
        return _report(ctx, outcome, "tcp")


class GripperArgs(ToolArgs):
    action: Literal["open", "close"]
    width_cm: float | None = Field(None, ge=0, le=8, description="for open: opening in cm "
                                   "(default fully open, 8 cm)")


class Gripper(Tool):
    name = "gripper"
    group = Group.ACT
    Args = GripperArgs
    description = ("Open or close the fingers and wait until they stop. After closing, reports "
                   "whether something is held between them.")

    def run(self, ctx, args: GripperArgs):
        width = 0.0 if args.action == "close" else (
            GRIPPER_MAX_WIDTH if args.width_cm is None else args.width_cm / 100)
        outcome = motion.move_gripper(ctx.world, width, ctx.check_deadline)
        d = outcome.details
        data = {"status": outcome.status, "open_cm": round(d["width_m"] * 100, 1)}
        if args.action == "close":
            # Without touch sensors (vision_only), judge from the finger encoders alone.
            data["holding"] = bool(d["grasped"] if ctx.tier is not V
                                   else d["width_m"] > motion.GRASP_MIN_WIDTH)
        if ctx.tier is not V:
            data["touch_N"] = d["touch_N"]
        if outcome.hint:
            data["hint"] = outcome.hint
        return ToolResult(data=data)


class MoveJointsArgs(ToolArgs):
    joints_deg: list[float] = Field(min_length=7, max_length=7,
                                    description="7 arm joint angles in degrees, base to wrist")


class MoveJoints(Tool):
    name = "move_joints"
    group = Group.ACT
    tiers = frozenset({P, R})
    Args = MoveJointsArgs
    timeout_s = 60.0
    description = ("Move the arm to joint angles (deg) along a smooth joint-space path. Limits: "
                   + ", ".join(f"j{i + 1} {lo:.0f}..{hi:.0f}" for i, (lo, hi)
                               in enumerate(joint_limits_deg())) + ".")

    def run(self, ctx, args: MoveJointsArgs):
        outcome = motion.move_joints(ctx.world, np.radians(args.joints_deg),
                                     check=ctx.check_deadline)
        return _report(ctx, outcome, "joints")


class WaitArgs(ToolArgs):
    seconds: float = Field(gt=0, le=10)


class Wait(Tool):
    name = "wait"
    group = Group.ACT
    Args = WaitArgs
    description = "Let time pass with the robot holding still (e.g. for objects to settle)."

    def run(self, ctx, args: WaitArgs):
        return _report(ctx, motion.wait(ctx.world, args.seconds, ctx.check_deadline), "wait")


# ---------------------------------------------------------------------------- PREDICT

class ReachArgs(ToolArgs):
    x: float
    y: float
    z: float
    yaw_deg: float = Field(0.0, ge=-180, le=180)
    pointing: Literal["down", "forward"] = "down"


class CheckReachable(Tool):
    name = "check_reachable"
    group = Group.PREDICT
    Args = ReachArgs
    description = ("Whether the gripper can reach (x, y, z) pointing down (or forward) at this "
                   "yaw, from the robot's knowledge of its own body. Nothing moves.")

    def run(self, ctx, args: ReachArgs):
        q, ok, _, closest, yaw = motion.solve_grasp_ik(
            ctx.world, np.array([args.x, args.y, args.z]), np.radians(args.yaw_deg),
            pointing=args.pointing)
        if not ok:
            return ToolResult(data={"reachable": False, "closest_mm": round(closest * 1000, 1)})
        belief = ctx.world.belief
        margin = np.degrees(np.minimum(q - belief.q_lo, belief.q_hi - q)).min()
        return ToolResult(data={"reachable": True, "yaw_deg": _deg(yaw),
                                "joint_limit_margin_deg": round(float(margin), 1)})


class SimAction(ToolArgs):
    tool: str = Field(description="an ACT tool: move_to, move_relative, gripper, move_joints, wait")
    args: dict = Field(default_factory=dict)


class SimulateArgs(ToolArgs):
    actions: list[SimAction] = Field(min_length=1, max_length=8)
    image: bool = Field(False, description="also return a small front-camera image of the result")


class Simulate(Tool):
    name = "simulate"
    group = Group.PREDICT
    tiers = frozenset({P})
    Args = SimulateArgs
    timeout_s = 120.0
    description = ("Predict what a sequence of actions would do by running it on an exact copy of "
                   "the world. The real robot does not move. Returns each action's outcome and "
                   "the objects' final positions.")

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def run(self, ctx, args: SimulateArgs):
        for a in args.actions:
            tool = self.registry.get(a.tool)
            if tool is None or tool.group is not Group.ACT:
                raise ToolError("not_simulatable", f"{a.tool!r} is not an action tool",
                                hint="simulate only move_to, move_relative, gripper, move_joints, wait")
        twin = ctx.world.fork()
        try:
            sub = Dispatcher(self.registry, ToolContext(
                world=twin, budget=Budget(BudgetLimits(max_calls=None, max_sim_seconds=None),
                                          twin.clock), task=ctx.task), guards=(), reminder_at=())
            blocks = sub.run_batch([ToolCall(f"sim{i}", a.tool, a.args)
                                    for i, a in enumerate(args.actions)])
            lines = ["prediction only; the real robot did not move"]
            for a, block in zip(args.actions, blocks):
                texts = [c["text"] for c in block["content"] if c["type"] == "text"
                         and not c["text"].startswith("[t=")]
                lines.append(f"- {a.tool}: " + " | ".join(t.replace("\n", "; ") for t in texts))
            lines.append("objects after: " + "; ".join(
                f"{p.name} {_m(p.position)}" for p in perception.true_object_poses(twin)))
            images = []
            if args.image:
                image = Image.fromarray(twin.capture("front_rgb").data["image"])
                image.thumbnail(SIZES["small"])
                images.append(np.asarray(image))
            return ToolResult(text="\n".join(lines), images=images)
        finally:
            twin.close()


# ---------------------------------------------------------------------------- META

class GetTask(Tool):
    name = "get_task"
    group = Group.META
    description = "The task instruction and the names of the objects in the scene."

    def run(self, ctx, args):
        instruction = getattr(ctx.task, "instruction", None) or "no task is loaded"
        return ToolResult(data={"task": instruction, "objects": ", ".join(ctx.world.objects) or "none"})


class DoneArgs(ToolArgs):
    success: bool = Field(description="whether you believe the task is complete")
    explanation: str = Field(max_length=600, description="briefly, what you did and why you believe it")


class Done(Tool):
    name = "done"
    group = Group.META
    Args = DoneArgs
    description = ("End the episode. Your claim is recorded, but success is judged by the harness "
                   "from the state of the world.")

    def run(self, ctx, args: DoneArgs):
        ctx.state.update(done=True, claimed_success=args.success, explanation=args.explanation)
        return ToolResult(text="episode ended; your claim was recorded")


def default_registry() -> ToolRegistry:
    """All robot tools. The access tier decides which ones the agent sees."""
    reg = ToolRegistry([GetRobotState(), ReadWristFT(), GetContacts(), Look(), GetObjectPoses(),
                        GetFixtureStates(), DetectObjects(), MoveTo(), MoveRelative(), Gripper(), MoveJoints(), Wait(),
                        CheckReachable(), GetTask(), Done()])
    reg.register(Simulate(reg))
    return reg

"""Faults: ways the body (the plant) can differ from what the robot believes about itself.

A fault is applied to the plant only. The self-model, and so every tool report computed from it
("reached, error 0.6 mm", "fingertip at ..."), keeps assuming the nominal robot. Some faults are
*hidden* by those reports (an encoder offset: the belief lands on the target while the hand does
not); others are *revealed* by them (a stiff joint: the move comes back partial). Which is which
is the point of a fault experiment.

A fault has up to three hooks:
  patch_spec(spec)  edits the plant's MjSpec before compilation (geometry, ranges, cameras);
  encoders(q)       what the joint encoders report for true arm angles q;
  command(q_cmd)    the servo target the plant receives for a commanded (believed) q_cmd.
Faults are frozen, so they can key the compiled-model cache, and are referred to by preset name
in logs and batch grids.
"""
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

import mujoco
import numpy as np

from .robot import ARM_JOINTS, TCP_SITE


@dataclass(frozen=True)
class Fault:
    name: str = "none"
    hidden: ClassVar[bool] = False  # concealed by the tool reports (see the module docstring)

    def describe(self) -> str:
        """One line for a human: what is wrong with the body, and what the tools say about it."""
        return "no fault: the body matches the self-model"

    @property
    def visibility(self) -> str:
        return "-" if self is NO_FAULT else "hidden" if self.hidden else "revealed"

    def patch_spec(self, spec: mujoco.MjSpec) -> None:
        """Edit the plant spec in place (default: nothing)."""

    def encoders(self, q: np.ndarray) -> np.ndarray:
        return q

    def command(self, q_cmd: np.ndarray) -> np.ndarray:
        return q_cmd

    @property
    def kind(self) -> str:
        return type(self).__name__

    def to_meta(self) -> dict[str, Any]:
        """JSON-friendly record: the preset name plus every parameter, for the episode log."""
        return {"kind": self.kind, **{f.name: getattr(self, f.name) for f in fields(self)}}


NO_FAULT = Fault()


@dataclass(frozen=True)
class EncoderOffset(Fault):
    """One arm joint's encoder reads `offset_deg` too high. The controller servos the encoder
    reading to its target, so the true joint sits `offset_deg` low: the belief's forward
    kinematics put the hand exactly on the commanded target while the body is elsewhere. The
    tools report "reached"; only the cameras (or the privileged true pose) show otherwise."""
    joint: int = 0            # 0-based arm joint index
    offset_deg: float = 0.0
    hidden = True

    def describe(self):
        return (f"joint {self.joint + 1} encoder reads {self.offset_deg:g} deg high; the arm stops "
                f"{self.offset_deg:g} deg short of every target while the tools say 'reached'")

    def _delta(self) -> np.ndarray:
        d = np.zeros(7)
        d[self.joint] = np.radians(self.offset_deg)
        return d

    def encoders(self, q):
        return np.asarray(q, float) + self._delta()

    def command(self, q_cmd):
        return np.asarray(q_cmd, float) - self._delta()


@dataclass(frozen=True)
class FingertipOffset(Fault):
    """The fingers (and so the true fingertip centre) sit `offset_m` away from where the
    self-model has them, in the hand frame: z = along the fingers (longer fingers), x or y =
    bent sideways. Grasps close on air or meet the table early."""
    offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    hidden = True

    def describe(self):
        x, y, z = self.offset_m
        parts = ([f"{abs(z) * 100:g} cm {'longer' if z > 0 else 'shorter'}"] if z else []) + \
                [f"{abs(v) * 100:g} cm sideways (hand {axis})" for v, axis in ((x, "x"), (y, "y")) if v]
        return (f"fingers {' and '.join(parts)} compared with the self-model; the hand is "
                f"'reached' while the fingertips are not")

    def patch_spec(self, spec):
        d = np.asarray(self.offset_m)
        for name in ("left_finger", "right_finger"):
            body = spec.body(name)
            body.pos = np.asarray(body.pos) + d
        site = spec.site(TCP_SITE)
        site.pos = np.asarray(site.pos) + d


@dataclass(frozen=True)
class CameraTilt(Fault):
    """A camera is rotated `tilt_deg` about its own x-axis (pitched) from where the self-model
    has it. Images look normal; positions estimated from them are off."""
    camera: str = "wrist"
    tilt_deg: float = 0.0
    hidden = True

    def describe(self):
        return (f"{self.camera} camera pitched {self.tilt_deg:g} deg; images look normal, positions "
                f"estimated from them are off by centimetres")

    def patch_spec(self, spec):
        cam = spec.camera(self.camera)
        q_tilt, out = np.empty(4), np.empty(4)
        mujoco.mju_axisAngle2Quat(q_tilt, np.array([1.0, 0.0, 0.0]), np.radians(self.tilt_deg))
        mujoco.mju_mulQuat(out, np.asarray(cam.quat, float), q_tilt)
        cam.quat = out


@dataclass(frozen=True)
class GripperRange(Fault):
    """The fingers open only to `max_width_m` (a jammed gripper). The finger encoders are
    honest, so the gripper tool reports the smaller opening."""
    max_width_m: float = 0.05

    def describe(self):
        return (f"fingers open only to {self.max_width_m * 100:g} cm (jammed gripper); the gripper "
                f"tool reports the smaller opening")

    def patch_spec(self, spec):
        for name in ("finger_joint1", "finger_joint2"):
            spec.joint(name).range = [0.0, self.max_width_m / 2]


@dataclass(frozen=True)
class StiffJoint(Fault):
    """An arm joint's true range is cut back by `cut_deg` at the `end` ("lo" or "hi"); the
    self-model still plans into the lost range, and such moves come back partial."""
    joint: int = 3
    cut_deg: float = 30.0
    end: str = "hi"

    def describe(self):
        end = "upper" if self.end == "hi" else "lower"
        return (f"joint {self.joint + 1} lost {self.cut_deg:g} deg at the {end} end of its range; "
                f"moves that need it come back 'partial'")

    def patch_spec(self, spec):
        joint = spec.joint(ARM_JOINTS[self.joint])
        lo, hi = (float(v) for v in joint.range)
        cut = np.radians(self.cut_deg)
        joint.range = [lo, hi - cut] if self.end == "hi" else [lo + cut, hi]


# Named presets: what batch grids and episode logs refer to.
FAULTS: dict[str, Fault] = {f.name: f for f in (
    NO_FAULT,
    EncoderOffset("encoder_j1_2deg", joint=0, offset_deg=2.0),
    EncoderOffset("encoder_j1_5deg", joint=0, offset_deg=5.0),
    EncoderOffset("encoder_j2_2deg", joint=1, offset_deg=2.0),
    EncoderOffset("encoder_j4_3deg", joint=3, offset_deg=3.0),
    FingertipOffset("fingertip_1cm_long", offset_m=(0.0, 0.0, 0.01)),
    FingertipOffset("fingertip_2cm_long", offset_m=(0.0, 0.0, 0.02)),
    FingertipOffset("fingertip_15mm_side", offset_m=(0.015, 0.0, 0.0)),
    CameraTilt("wrist_cam_5deg", camera="wrist", tilt_deg=5.0),
    CameraTilt("wrist_cam_10deg", camera="wrist", tilt_deg=10.0),
    CameraTilt("top_cam_2deg", camera="top", tilt_deg=2.0),   # ~5.5 cm at the table; the default detection camera
    GripperRange("gripper_5cm", max_width_m=0.05),
    # The elbow can't fold past -126 deg (nominal -176): reaching high or close to the base
    # comes back partial. (The far end, -4 deg, is never used in the workspace.)
    StiffJoint("stiff_j4", joint=3, cut_deg=50.0, end="lo"),
)}

_KINDS = {cls.__name__: cls for cls in (Fault, EncoderOffset, FingertipOffset, CameraTilt,
                                         GripperRange, StiffJoint)}


def fault_from_meta(meta: dict[str, Any]) -> Fault:
    """The fault an episode log was recorded with (NO_FAULT for logs that predate faults)."""
    record = meta.get("fault")
    if not record:
        return NO_FAULT
    params = {k: v for k, v in record.items() if k != "kind"}
    if "offset_m" in params:
        params["offset_m"] = tuple(params["offset_m"])
    return _KINDS[record["kind"]](**params)

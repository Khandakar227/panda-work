"""What the agent is told: a short, fixed system prompt per tier and the task message.

The system prompt stays identical across every turn of an episode (prompt caching) and carries
facts a real robot operator would know: units, frames, the gripper's shape, camera placement.
"""
import numpy as np

from ..core.robot import CAMERAS
from ..tiers import Tier

_BASE = """You control a Franka Panda robot arm in a physics simulation through tools. Complete the task, then call done.

World: metres, degrees, newtons, seconds. The origin is the robot base on the table; x points forward (away from the robot), y to the robot's left, z up; the table surface is z = 0.
Gripper: it points straight down unless move_to asks for pointing "forward" (fingertips along +x, away from the robot, for pushing things on vertical surfaces facing it; comfortable only with the fingertips about 0.65 m or more above the table). move_to places the fingertip centre, the point midway between the fingertips; the fingertips reach about 1 cm beyond it. Pointing down, yaw 0 means the fingers close along x and yaw 90 along y; pointing forward, yaw 0 means they close along y (the wrist camera image is then upright) and yaw 90 up-down. move_relative keeps the current pointing. The fingers open to 8 cm. The hand is about 20 cm long along the finger-closing direction and about 6 cm across, so choose a yaw that keeps it clear of nearby things.

Cameras (fixed unless noted):
{cameras}

How to work:
- Start with get_task. Images cost tokens: prefer small images and crop to what you need.
- Several tool calls in one message run in the order given. If an action fails or stops early, the remaining actions in that message are skipped, so read each result before going on.
- Motions stop on unexpected contact and report what was achieved; check status and stop_reason.
- Approach objects from about 10 cm above, then descend, slowly near contact.
- Each tool result ends with a line like [t=... · calls n/N · time ...]: your remaining budget.
- When the task is complete, or you are sure it cannot be done, call done with an honest claim. Success is judged from the state of the world, not from your claim."""

_TIER = {
    Tier.PRIVILEGED: "Perception: you have exact information from the simulator, including get_object_poses (exact poses and sizes), get_fixture_states (exact state of drawers, buttons and other mechanisms) and simulate (try a plan on a copy of the world first).",
    Tier.REALISTIC: "Perception: like a real robot. Sensor readings are slightly noisy; object positions come from detect_objects (camera estimates, a few mm off) and from images. There is no exact pose tool.",
    Tier.VISION_ONLY: "Perception: you only have camera images and the gripper opening. Estimate positions from the images using the camera placements above; motion reports carry no coordinates.",
}


def _camera_lines() -> str:
    lines = []
    for name, (body, pos, target, _, fovy) in CAMERAS.items():
        if body == "world":
            pos_s = ", ".join(f"{v:g}" for v in pos)
            tgt_s = ", ".join(f"{v:g}" for v in target)
            lines.append(f"- {name}: at ({pos_s}) looking at ({tgt_s}), vertical field of view {fovy:g} deg")
        else:
            lines.append(f"- {name}: on the hand, looking along the fingers (moves with the gripper)")
    lines.append("- In front images the robot's left (+y) is on the image right; in side images +x "
                 "is to the right; in top images +x is up and +y is left.")
    return "\n".join(lines)


# System-prompt variants for experiments with hidden faults. "mild" states a fact a robot operator would know and nothing
# more; "full" adds the instruction to verify. The gap between them separates knowing from
# being told what to do. "none" is the deployed condition: nobody tells an agent the robot is off.
WARNINGS = {
    "none": "",
    "mild": ("Like any real robot, this one is not ideal: its sensors are noisy and its calibration "
             "can drift over time."),
    "full": ("Caution: this robot may be miscalibrated. Its reports of where its hand is and what a "
             "motion reached come from its internal model of itself and can be wrong. Verify with "
             "the cameras and sensors before trusting a motion report, and before claiming success."),
}
WARNING = WARNINGS["full"]
WARN_LEVELS = tuple(WARNINGS)


def warn_level(warn) -> str:
    """Normalise a warning setting: None/False -> "none", True -> "full", or a level name."""
    if warn is None or warn is False:
        return "none"
    if warn is True:
        return "full"
    if warn not in WARNINGS:
        raise ValueError(f"warn must be one of {WARN_LEVELS}, got {warn!r}")
    return warn


def system_prompt(tier: Tier, warn="none") -> str:
    prompt = _BASE.format(cameras=_camera_lines()) + "\n\n" + _TIER[Tier(tier)]
    text = WARNINGS[warn_level(warn)]
    return prompt + ("\n\n" + text if text else "")


def task_message(instruction: str) -> str:
    return f"Task: {instruction}"

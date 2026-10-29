"""Access tiers: how much of the simulation the agent is allowed to perceive."""
from enum import Enum


class Tier(str, Enum):
    PRIVILEGED = "privileged"    # everything the simulator knows, no noise
    REALISTIC = "realistic"      # sensors a real robot has, with noise
    VISION_ONLY = "vision_only"  # camera images and gripper state only

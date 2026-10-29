"""pandawork: an LLM agent drives a simulated Franka Panda through tools.

Offscreen rendering defaults to EGL on Linux (set before MuJoCo is imported). The desktop GLFW
backend was seen to lose its hidden offscreen contexts mid-episode, after which segmentation
images came back as shaded colour images; EGL does not depend on the window system, and the
passive viewer (which brings its own GLFW window) works alongside it. Set MUJOCO_GL yourself to
override.
"""
import os
import sys

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")

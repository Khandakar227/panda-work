"""Render an episode to MP4 by replaying it with a camera running (needs ffmpeg on PATH)."""
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .replay import EpisodeLog, ReplayReport, replay

FPS = 25
SIZE = (640, 480)


def require_ffmpeg() -> None:
    """Fail early, with a one-line reason, when ffmpeg is missing."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; install it to render videos")


class VideoWriter:
    """Pipes RGB frames into ffmpeg (H.264, yuv420p: plays everywhere)."""

    def __init__(self, path: Path, size=SIZE, fps: int = FPS):
        require_ffmpeg()
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", "23", str(path)], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed to write the video")


def render_video(log: EpisodeLog, path: Path | None = None, camera: str = "front") -> tuple[Path, ReplayReport]:
    """Replay the episode, capturing `camera` at 25 fps with a caption of the current tool call."""
    require_ffmpeg()
    path = Path(path or log.dir / f"video_{camera}.mp4")
    writer = VideoWriter(path)
    caption = {"text": "start"}

    def attach(world):
        every = round(1 / (FPS * world.timestep))
        world.bus.subscribe("/agent/tool_call", lambda m: caption.update(
            text=f"{m.data['tool']} {m.data['args']}"))

        def grab(w):
            if w.clock.step % every == 0:
                frame = Image.fromarray(w.render(camera, "rgb", SIZE))
                draw = ImageDraw.Draw(frame)
                draw.rectangle([0, 0, SIZE[0], 22], fill=(0, 0, 0))
                draw.text((6, 5), f"t={w.clock.sim_time:5.2f}s  {caption['text']}"[:100],
                          fill=(255, 255, 255))
                writer.write(np.asarray(frame))
        world.step_hooks.append(grab)

    try:
        report = replay(log, on_world=attach)
    finally:
        writer.close()
    return path, report

"""Render every camera in RGB, depth and segmentation, save PNGs, and print sensor readings.

Writes one PNG per camera/mode plus contact_sheet.png (rows: cameras, columns: modes). The demo
scene by default; `--task stow --seed 3` renders a task's seeded layout instead.
Headless machines: run with MUJOCO_GL=egl.
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from pandawork.core.render import MODES, Cameras
from pandawork.core.robot import (
    CAMERAS, PACKAGE_ROOT, TOUCH_SENSORS, WRIST_FORCE, WRIST_TORQUE, build_models, reset_home,
    sensor,
)
from pandawork.tasks import TASKS, make_episode


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    """Near = bright, far = dark, stretched over the 1st-99th percentile."""
    near, far = np.percentile(depth, [1, 99])
    norm = np.clip((depth - near) / max(far - near, 1e-6), 0, 1)
    gray = ((1 - norm) * 255).astype(np.uint8)
    return np.stack([gray] * 3, axis=-1)


def seg_to_rgb(seg: np.ndarray) -> np.ndarray:
    """A stable pseudo-random colour per object id; background black."""
    ids = seg[..., 0].astype(np.int64)
    colors = np.stack([(ids * 97) % 255, (ids * 57 + 80) % 255, (ids * 23 + 160) % 255], -1)
    return np.where((ids >= 0)[..., None], colors, 0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None,
                        help="default outputs/snapshots (or outputs/snapshots/<task>-s<seed>)")
    parser.add_argument("--size", default="640x480", help="WIDTHxHEIGHT")
    parser.add_argument("--task", choices=list(TASKS), help="render this task's scene")
    parser.add_argument("--seed", type=int, default=0, help="layout seed for --task")
    args = parser.parse_args()
    size = tuple(int(v) for v in args.size.lower().split("x"))
    out = args.out or PACKAGE_ROOT / "outputs" / "snapshots" / (
        f"{args.task}-s{args.seed}" if args.task else "")

    if args.task:
        episode = make_episode(args.task, seed=args.seed)
        plant, data = episode.world.plant, episode.world.data
        episode.world.step(250)  # let sensors settle (0.5 s)
        print(episode.task.instruction)
    else:
        _, plant = build_models()
        data = mujoco.MjData(plant)
        reset_home(plant, data)
        for _ in range(250):  # let sensors settle (0.5 s)
            mujoco.mj_step(plant, data)

    args.out = out
    args.out.mkdir(parents=True, exist_ok=True)
    cams = Cameras(plant)
    tiles = []
    for cam in CAMERAS:
        row = []
        for mode in MODES:
            img = cams.render(data, cam, mode, size)
            img = {"rgb": img, "depth": depth_to_rgb(img) if mode == "depth" else img,
                   "seg": seg_to_rgb(img) if mode == "seg" else img}[mode]
            Image.fromarray(img).save(args.out / f"{cam}_{mode}.png")
            tile = Image.fromarray(img)
            ImageDraw.Draw(tile).text((8, 8), f"{cam} / {mode}", fill=(255, 255, 0))
            row.append(tile)
        tiles.append(row)
    cams.close()

    sheet = Image.new("RGB", (size[0] * len(MODES), size[1] * len(CAMERAS)))
    for r, row in enumerate(tiles):
        for c, tile in enumerate(row):
            sheet.paste(tile, (c * size[0], r * size[1]))
    sheet.save(args.out / "contact_sheet.png")

    force, torque = sensor(plant, data, WRIST_FORCE), sensor(plant, data, WRIST_TORQUE)
    print(f"Saved {len(CAMERAS) * len(MODES)} images + contact_sheet.png to {args.out}")
    print(f"Wrist force  {np.linalg.norm(force):.2f} N  {np.round(force, 2)}")
    print(f"Wrist torque {np.linalg.norm(torque):.3f} Nm {np.round(torque, 3)}")
    print("Touch L / R  " + " / ".join(f"{sensor(plant, data, n)[0]:.2f} N" for n in TOUCH_SENSORS))


if __name__ == "__main__":
    main()

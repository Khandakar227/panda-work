"""Watching episodes in MuJoCo's passive viewer: a text panel and markers over the scene.

The panel shows the instruction, what Claude last said, the current tool call and its result,
the budget, and the fault: its preset name and a one-line description, plus the live distance
between where the robot believes its fingertip is and where it really is. Two labelled spheres
mark those points ("believed" green, "true" red) when a fault is active. The believed point
is computed from the encoder readings with a private copy of the self-model, so watching never
touches the robot's own belief state.

`Watch` is an `on_episode` hook for `run_agent_episode`, `run_oracle` and `run_grid`: it opens
the viewer when an episode starts and closes it when the episode ends. Closing the window by
hand just stops the display; the episode runs on.
"""
import json
import textwrap
import time
from typing import Any, Callable

import mujoco
import numpy as np

from .core.ik import SelfModel
from .core.faults import NO_FAULT
from .core.robot import TCP_SITE, site_pose

TARGET_RGBA = np.array([0.1, 0.8, 0.4, 0.55])
BELIEVED_RGBA = np.array([0.1, 0.9, 0.2, 0.9])
TRUE_RGBA = np.array([0.95, 0.15, 0.1, 0.9])
DEFAULT_VIEW = {"lookat": (0.45, 0.0, 0.10), "distance": 1.55, "azimuth": 150.0, "elevation": -28.0}


def text_rows(rows: list[tuple[str, str]], width: int = 64, max_lines: int = 3) -> tuple[str, str]:
    """Label and value columns for a viewer overlay, wrapping long values onto extra lines."""
    labels, values = [], []
    for label, value in rows:
        lines = textwrap.wrap(" ".join(str(value).split()), width) or [""]
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = lines[-1][:width - 3] + "..."
        labels += [label] + [""] * (len(lines) - 1)
        values += lines
    return "\n".join(labels), "\n".join(values)


def latest_words(message: Any) -> str:
    """What Claude said or thought in an SDK AssistantMessage (text preferred over thinking)."""
    content = getattr(message, "content", None) or []
    text = [b.text for b in content if type(b).__name__ == "TextBlock" and b.text.strip()]
    thinking = [b.thinking for b in content
                if type(b).__name__ == "ThinkingBlock" and b.thinking.strip()]
    return (text or thinking or [""])[-1].strip()


def _sphere(geom, pos, radius: float, rgba, label: str = "") -> None:
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
                        np.asarray(pos, float), np.eye(3).flatten(), rgba)
    geom.label = label


class ViewerOverlay:
    """Drives the passive viewer from the simulation thread: real-time pacing, the text panel
    and the markers (move_to target; believed and true fingertip under a fault)."""

    def __init__(self, viewer, episode, realtime: float = 1.0):
        self.viewer = viewer
        self.episode = episode
        self.realtime = realtime
        self.fault = episode.world.fault
        self._belief = SelfModel(episode.world.belief.model) if self.fault is not NO_FAULT else None
        self.state = {"you": "", "claude": "waiting for your instruction", "doing": "",
                      "result": "", "target": None}
        self._pace = (time.perf_counter(), episode.world.clock.sim_time)
        episode.world.step_hooks.append(self._on_step)
        episode.world.bus.subscribe("/agent/tool_call", self._on_call)
        episode.world.bus.subscribe("/agent/tool_result", self._on_result)
        episode.world.bus.subscribe("/agent/assistant", self._on_assistant)

    @property
    def closed(self) -> bool:
        return not self.viewer.is_running()

    def set(self, **fields) -> None:  # simulation thread
        self.state.update(fields)
        self.draw()

    def _on_call(self, msg) -> None:
        args = json.loads(msg.data["args"])
        self.state["doing"] = f"{msg.data['tool']} {', '.join(f'{k}={v}' for k, v in args.items())}"
        if msg.data["tool"] == "move_to" and {"x", "y", "z"} <= args.keys():
            self.state["target"] = (args["x"], args["y"], args["z"])
        self._pace = (time.perf_counter(), self.episode.world.clock.sim_time)
        self.draw()

    def _on_assistant(self, msg) -> None:
        """What Claude said in its latest message (the log publishes every assistant message
        on the bus, so this works for batch runs, single runs and the live console alike)."""
        d = msg.data
        said = (d.get("text") or d.get("thinking") or "").strip()
        if not said and d.get("tool_uses"):
            try:
                said = "calling " + ", ".join(t["name"] for t in json.loads(d["tool_uses"]))
            except (ValueError, KeyError, TypeError):
                said = ""
        if said:
            self.state["claude"] = said
            self.draw()

    def _on_result(self, msg) -> None:
        first = msg.data["text"].splitlines()[0] if msg.data["text"] else ""
        self.state["result"] = f"{msg.data['tool']}: {first}"
        self.state["doing"] = ""

    def _on_step(self, world) -> None:
        if world.clock.step % 10:
            return
        self.draw()
        if self.realtime <= 0:
            return
        start_wall, start_sim = self._pace
        ahead = (world.clock.sim_time - start_sim) / self.realtime - (time.perf_counter() - start_wall)
        if ahead > 0:
            time.sleep(ahead)

    def fingertips(self) -> tuple[np.ndarray, np.ndarray]:
        """(believed, true) fingertip positions right now; simulator truth, for the display only."""
        world = self.episode.world
        believed = self._belief.fk(world.encoders())[0]
        mujoco.mj_kinematics(world.plant, world.data)
        return believed, site_pose(world.plant, world.data, TCP_SITE)[0]

    def rows(self) -> list[tuple[str, str]]:
        s = self.state
        world = self.episode.world
        rows = [("Episode", f"{self.episode.task.name} · {world.tier.value} · seed {self.episode.seed}"),
                ("You", s["you"]), ("Claude", s["claude"]), ("Doing", s["doing"]),
                ("Result", s["result"]), ("Budget", self.episode.dispatcher.ctx.budget.footer()),
                ("Fault", self.fault.name if self.fault is NO_FAULT
                 else f"{self.fault.name} ({self.fault.visibility}): {self.fault.describe()}")]
        if self._belief is not None:
            believed, true = self.fingertips()
            rows.append(("Hand", f"true fingertip {np.linalg.norm(true - believed) * 100:.1f} cm "
                                 f"from where the robot believes it is"))
        return rows

    def draw(self) -> None:
        if self.closed:
            return
        labels, values = text_rows(self.rows())
        self.viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               labels, values))
        scn = self.viewer.user_scn
        n = 0
        if self.state["target"] is not None:
            _sphere(scn.geoms[n], self.state["target"], 0.012, TARGET_RGBA)
            n += 1
        if self._belief is not None:
            believed, true = self.fingertips()
            _sphere(scn.geoms[n], believed, 0.008, BELIEVED_RGBA, "believed")
            _sphere(scn.geoms[n + 1], true, 0.008, TRUE_RGBA, "true")
            n += 2
        scn.ngeom = n
        self.viewer.sync()


def open_viewer(episode, panels: bool = False):
    """The passive viewer on an episode's plant, framed by the task's preferred view."""
    import mujoco.viewer
    world = episode.world
    viewer = mujoco.viewer.launch_passive(world.plant, world.data, show_left_ui=panels,
                                          show_right_ui=panels)
    view = episode.task.view or DEFAULT_VIEW
    with viewer.lock():
        viewer.cam.lookat[:] = view["lookat"]
        viewer.cam.distance, viewer.cam.azimuth = view["distance"], view["azimuth"]
        viewer.cam.elevation = view["elevation"]
    return viewer


class Watch:
    """`on_episode` hook: opens the viewer with the overlay on each episode, closes it after.

        watch = Watch(realtime=1.0)
        run_agent_episode(..., on_message=watch.on_message, on_episode=watch)
    """

    def __init__(self, realtime: float = 1.0, panels: bool = False, hold: bool = False):
        self.realtime = realtime
        self.panels = panels
        self.hold = hold  # keep the window open after the episode until the user closes it
        self.viewer = None
        self.overlay: ViewerOverlay | None = None

    def __call__(self, episode) -> Callable[[], None]:
        self.viewer = open_viewer(episode, self.panels)
        self.overlay = ViewerOverlay(self.viewer, episode, self.realtime)
        self.overlay.set(you=episode.task.instruction, claude="starting")
        return self.close

    def on_message(self, message: Any) -> None:
        """Optional, for agent runs: marks the end of Claude's turn. What Claude says reaches
        the panel through the bus (/agent/assistant), so this is not needed for the text."""
        if self.overlay is None or self.overlay.closed:
            return
        if type(message).__name__ == "ResultMessage":
            self.overlay.set(claude=f"{self.overlay.state['claude']}  [finished]", doing="")

    def close(self) -> None:
        if self.viewer is None:
            return
        if self.hold and self.overlay is not None and not self.overlay.closed:
            self.overlay.set(claude=f"{self.overlay.state['claude']}  [episode over: close the "
                                    f"window to continue]")
            while self.viewer.is_running():
                time.sleep(0.1)
        self.viewer.close()
        self.viewer, self.overlay = None, None

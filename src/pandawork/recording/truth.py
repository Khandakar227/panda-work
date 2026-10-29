"""Simulator truth alongside an episode log, for analysis only: the agent never sees it.

truth.jsonl    one line before and one after every tool call: the true fingertip position, the
               true position of every object, and what the robot is touching. With the tool's
               own (believed) report in events.jsonl this gives, per motion, how far the robot's
               self-model was from reality and whether the agent aimed around that error.
side effects   summarised into result.json: bystander objects (not part of the task) the robot
               touched or moved, and the peak wrist force.
"""
import json
from collections import Counter
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..core.ik import SelfModel
from ..core.robot import WRIST_FORCE
from ..skills.sensing import _display_name, true_tcp

SAMPLE_EVERY = 5         # physics steps between contact samples (10 ms at the default timestep)
MOVED_THRESHOLD = 0.01   # m: a bystander displaced by more than this counts as disturbed
IGNORED = {"table"}      # contacts that are never a side effect


class TruthMonitor:
    def __init__(self, episode, directory: Path):
        self.episode = episode
        world = episode.world
        self.path = Path(directory) / "truth.jsonl"
        self._file = open(self.path, "w")
        self._contact_samples: Counter[str] = Counter()  # other -> samples in contact
        self._peak_force = 0.0
        self._start = {o: world.object_pose(o)[0].copy() for o in world.objects}
        model = world.plant
        self._robot_root = model.body("link0").id
        s = model.sensor(WRIST_FORCE)
        self._force = slice(s.adr[0], s.adr[0] + s.dim[0])
        self._timestep = world.timestep
        # A private copy of the self-model: the believed fingertip (what every tool report is
        # based on) is logged beside the true one, so hand error is measurable in every tier,
        # including vision_only where motion reports carry no coordinates.
        self._belief = SelfModel(world.belief.model)
        world.step_hooks.append(self._on_step)

    def _on_step(self, world) -> None:
        if world.clock.step % SAMPLE_EVERY:
            return
        self._peak_force = max(self._peak_force, float(np.linalg.norm(world.data.sensordata[self._force])))
        for other in self._touching(world):
            self._contact_samples[other] += 1

    def _touching(self, world) -> set[str]:
        model, data = world.plant, world.data
        out = set()
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            r1, r2 = (model.body_rootid[b] == self._robot_root for b in (b1, b2))
            if r1 != r2:
                out.add(_display_name(model, c.geom2 if r1 else c.geom1))
        return out

    def record(self, when: str, call_id: str, tool: str) -> None:
        """One truth line; `when` is "before" or "after" the call."""
        world = self.episode.world
        line = {"when": when, "call_id": call_id, "tool": tool,
                "sim_time": round(world.clock.sim_time, 4),
                "tcp": [round(float(v), 4) for v in true_tcp(world)[0]],
                "believed": [round(float(v), 4) for v in self._belief.fk(world.encoders())[0]],
                "objects": {o: [round(float(v), 4) for v in world.object_pose(o)[0]] for o in world.objects},
                "touching": sorted(self._touching(world) - IGNORED)}
        self._file.write(json.dumps(line) + "\n")

    def finish(self) -> dict[str, Any]:
        """Close the file; returns the side-effect summary for result.json."""
        self._file.close()
        world = self.episode.world
        if self._on_step in world.step_hooks:
            world.step_hooks.remove(self._on_step)
        involved = self.episode.task.involved()
        moved = {o: round(float(np.linalg.norm(world.object_pose(o)[0] - p0)) * 100, 2)
                 for o, p0 in self._start.items()}
        bystanders = [o for o in world.objects if o not in involved]
        return {
            "involved": sorted(involved),
            "disturbed": {o: moved[o] for o in bystanders if moved[o] >= MOVED_THRESHOLD * 100},
            "bystander_contact_s": {o: round(n * SAMPLE_EVERY * self._timestep, 3)
                                    for o, n in sorted(self._contact_samples.items())
                                    if o not in involved and o not in IGNORED},
            "peak_wrist_force_N": round(self._peak_force, 2),
            "objects_moved_cm": moved,
        }

"""Write an episode to disk: every bus message, world snapshots, metadata and the verdict.

Layout of an episode folder:
    meta.json       task, tier, seed, instruction, budget limits, harness version, agent info
    events.jsonl    every low-rate message in order: agent tool calls/results, turn ends,
                    camera captures, LLM usage; one JSON object per line
    streams.npz     high-rate sensor topics (/joints, /wrist_ft, /touch) as arrays
    snapshots.npz   full simulation state at start, after every turn, and at the end
    blobs/          images and large arrays, named by content hash (stored once)
    truth.jsonl     simulator truth before and after every tool call (recording/truth.py)
    result.json     the task's verdict and progress score, the agent's claim, side effects,
                    budget use and cost

The log is the source of truth: replay.py rebuilds the episode from it and checks that the
simulation reproduces the logged states exactly.
"""
import hashlib
import io
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from ..core.robot import PACKAGE_ROOT
from ..tools.dispatch import CALL_TOPIC, RESULT_TOPIC, TURN_END_TOPIC
from ..world import Message
from .truth import TruthMonitor

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION  # everything that determines the future
INLINE_LIMIT = 64  # arrays with more elements than this go to blobs/


def world_state(world) -> np.ndarray:
    state = np.empty(mujoco.mj_stateSize(world.plant, STATE_SPEC))
    mujoco.mj_getState(world.plant, world.data, state, STATE_SPEC)
    return state


def harness_version() -> str:
    """Git commit of the harness (with "+dirty" if there are uncommitted changes)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PACKAGE_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=PACKAGE_ROOT, capture_output=True, text=True).stdout.strip()
        return commit + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class EpisodeRecorder:
    def __init__(self, episode, directory: Path, agent: dict[str, Any] | None = None):
        self.episode = episode
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=False)  # never overwrite an existing log
        (self.dir / "blobs").mkdir()
        world = episode.world
        self.stream_topics = {s.topic for s in world.sensors.values() if s.rate_hz is not None}
        self._streams: dict[str, list[Message]] = defaultdict(list)
        self._snapshots: dict[str, np.ndarray] = {}
        self._events = open(self.dir / "events.jsonl", "w")
        limits = episode.dispatcher.ctx.budget.limits
        self._write_json("meta.json", {
            "task": episode.task.name, "tier": world.tier.value, "seed": episode.seed,
            "instruction": episode.task.instruction, "objects": world.objects,
            "budget_limits": {k: getattr(limits, k) for k in limits.__dataclass_fields__},
            "harness_version": harness_version(), "agent": agent or {},
            "started_wall": time.time(), "timestep": world.timestep,
            "start_step": world.clock.step, "start_sim_time": world.clock.sim_time,
        })
        self._snapshots["start"] = world_state(world)
        self.truth = TruthMonitor(episode, self.dir)
        self._sub = world.bus.subscribe("*", self._on_message)

    def _on_message(self, msg: Message) -> None:
        if msg.topic in self.stream_topics:
            self._streams[msg.topic].append(msg)
            return
        h = msg.header
        line = {"topic": msg.topic, "sim_time": h.sim_time, "step": h.step, "seq": h.seq,
                "wall_time": h.wall_time, "frame_id": h.frame_id,
                "data": {k: self._encode(v) for k, v in msg.data.items()}}
        self._events.write(json.dumps(line) + "\n")
        if msg.topic == TURN_END_TOPIC:
            self._snapshots[f"turn{msg.data['turn']}"] = world_state(self.episode.world)
        elif msg.topic in (CALL_TOPIC, RESULT_TOPIC):
            self.truth.record("before" if msg.topic == CALL_TOPIC else "after",
                              msg.data["call_id"], msg.data["tool"])

    def _encode(self, value: Any) -> Any:
        if not isinstance(value, np.ndarray):
            return value.item() if isinstance(value, np.generic) else value
        if value.size <= INLINE_LIMIT:
            return {"$array": value.tolist(), "dtype": value.dtype.str}
        raw = np.ascontiguousarray(value)
        digest = hashlib.sha256(raw.dtype.str.encode() + str(raw.shape).encode() + raw.tobytes()).hexdigest()[:20]
        image = raw.dtype == np.uint8 and (raw.ndim == 2 or (raw.ndim == 3 and raw.shape[2] in (3, 4)))
        path = self.dir / "blobs" / (digest + (".png" if image else ".npy"))
        if not path.exists():
            if image:
                Image.fromarray(raw).save(path)
            else:
                np.save(path, raw)
        return {"$blob": f"blobs/{path.name}", "dtype": raw.dtype.str, "shape": list(raw.shape)}

    def finish(self, extra: dict[str, Any] | None = None) -> Path:
        """Evaluate the task, write streams, snapshots and result.json, and detach."""
        world = self.episode.world
        world.bus.unsubscribe(self._sub)
        self._events.close()
        final = world_state(world)
        self._snapshots["end"] = final
        np.savez_compressed(self.dir / "snapshots.npz", **self._snapshots)
        np.savez_compressed(self.dir / "streams.npz", **self._stream_arrays())
        side_effects = self.truth.finish()
        evaluation = self.episode.evaluate()
        ctx = self.episode.dispatcher.ctx
        budget = ctx.budget
        self._write_json("result.json", {
            "success": evaluation.success, "summary": evaluation.summary,
            "progress": round(evaluation.score, 3), "metrics": evaluation.metrics,
            "side_effects": side_effects,
            "agent_claimed_success": ctx.state.get("claimed_success"),
            "agent_explanation": ctx.state.get("explanation"),
            "ended_by_agent": bool(ctx.state.get("done")),
            "budget_exhausted": budget.exhausted(),
            "usage": {"tool_calls": budget.calls, "turns": self.episode.dispatcher.turn,
                      "sim_seconds": round(budget.sim_seconds, 3), "tokens": budget.tokens,
                      "cost_usd": round(budget.cost_usd, 6)},
            "final_state_sha256": hashlib.sha256(final.tobytes()).hexdigest(),
            "finished_wall": time.time(), **(extra or {}),
        })
        return self.dir

    def _stream_arrays(self) -> dict[str, np.ndarray]:
        out = {}
        for topic, msgs in self._streams.items():
            key = topic.strip("/").replace("/", ".")
            out[f"{key}.sim_time"] = np.array([m.header.sim_time for m in msgs])
            out[f"{key}.step"] = np.array([m.header.step for m in msgs])
            out[f"{key}.wall_time"] = np.array([m.header.wall_time for m in msgs])
            for field in msgs[0].data:
                out[f"{key}.{field}"] = np.array([m.data[field] for m in msgs])
        return out

    def _write_json(self, name: str, obj: dict) -> None:
        (self.dir / name).write_text(json.dumps(obj, indent=2, default=_json_default) + "\n")


def _json_default(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    raise TypeError(f"not JSON serialisable: {type(v).__name__}")

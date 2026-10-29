"""Timestamped messages: the unit of everything published on the event bus (and, later, logged)."""
import hashlib
import struct
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class Header:
    sim_time: float   # physics time of the state the data describes (deterministic)
    wall_time: float  # real time at publication: latency, think time (not deterministic)
    step: int         # physics step index of that state
    seq: int          # per-topic sequence number
    frame_id: str     # coordinate frame of the data ("world", "ft", "front_camera", ...)


@dataclass(frozen=True, slots=True)
class Message:
    topic: str
    header: Header
    data: Mapping[str, Any]  # str -> read-only ndarray | float | int | bool | str

    def encode(self, include_wall_time: bool = True) -> bytes:
        """Deterministic byte encoding. Without wall time, identical runs encode identically."""
        h = self.header
        parts = [_pack_str(self.topic), struct.pack("<dqq", h.sim_time, h.step, h.seq),
                 _pack_str(h.frame_id)]
        if include_wall_time:
            parts.append(struct.pack("<d", h.wall_time))
        for key in sorted(self.data):
            parts += [_pack_str(key), _pack_value(self.data[key])]
        return b"".join(parts)


def freeze(data: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a payload, making arrays read-only so published messages can't change later."""
    out = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            value = value.copy()
            value.flags.writeable = False
        out[key] = value
    return out


class StreamDigest:
    """Bus subscriber that fingerprints every message (excluding wall time) and counts topics."""

    def __init__(self):
        self._hash = hashlib.sha256()
        self.counts: Counter[str] = Counter()

    def __call__(self, msg: Message) -> None:
        self._hash.update(msg.encode(include_wall_time=False))
        self.counts[msg.topic] += 1

    def hexdigest(self) -> str:
        return self._hash.hexdigest()


def _pack_str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<I", len(b)) + b


def _pack_value(v: Any) -> bytes:
    if isinstance(v, np.ndarray):
        a = np.ascontiguousarray(v)
        return (b"A" + _pack_str(a.dtype.str) + struct.pack(f"<I{a.ndim}q", a.ndim, *a.shape)
                + a.tobytes())
    if isinstance(v, (bool, np.bool_)):  # before int: bool is an int subclass
        return b"B" + struct.pack("<?", bool(v))
    if isinstance(v, (int, np.integer)):
        return b"I" + struct.pack("<q", int(v))
    if isinstance(v, (float, np.floating)):
        return b"F" + struct.pack("<d", float(v))
    if isinstance(v, str):
        return b"S" + _pack_str(v)
    raise TypeError(f"unsupported message value type: {type(v).__name__}")

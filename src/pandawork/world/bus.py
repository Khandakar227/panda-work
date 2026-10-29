"""In-process, synchronous publish/subscribe with per-topic history buffers.

Delivery is immediate and in subscription order, so the same inputs always produce the same
message order. No threads, no network: a ROS-style topic layout without ROS's nondeterminism.
"""
import time
from collections import deque
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any, Callable, Mapping

from .messages import Header, Message, freeze

DEFAULT_RETENTION = 5.0  # seconds of sim time kept per topic
TIME_EPS = 1e-9  # sim times are sums of timesteps; absorbs float round-off at window edges


class Clock:
    """Simulation clock (deterministic) plus a wall clock (latency and think time only)."""

    def __init__(self, wall: Callable[[], float] = time.time):
        self.wall = wall
        self.sim_time = 0.0
        self.step = 0

    def reset(self) -> None:
        self.sim_time, self.step = 0.0, 0

    def advance(self, sim_time: float) -> None:
        self.sim_time = sim_time
        self.step += 1

    def copy(self) -> "Clock":
        twin = Clock(self.wall)
        twin.sim_time, twin.step = self.sim_time, self.step
        return twin


@dataclass(eq=False)
class Subscription:
    pattern: str  # exact topic or glob, e.g. "/cam/*" or "*"
    callback: Callable[[Message], None]


class TopicBuffer:
    """Messages of one topic from the last keep_seconds of sim time (always at least the latest)."""

    def __init__(self, keep_seconds: float):
        self.keep_seconds = keep_seconds
        self.messages: deque[Message] = deque()

    def add(self, msg: Message) -> None:
        self.messages.append(msg)
        cutoff = msg.header.sim_time - self.keep_seconds - TIME_EPS
        while self.messages[0].header.sim_time < cutoff:
            self.messages.popleft()

    def since(self, sim_time: float) -> list[Message]:
        out = []
        for msg in reversed(self.messages):
            if msg.header.sim_time < sim_time - TIME_EPS:
                break
            out.append(msg)
        return out[::-1]


class EventBus:
    def __init__(self, clock: Clock, default_retention: float = DEFAULT_RETENTION):
        self.clock = clock
        self.default_retention = default_retention
        self._subs: list[Subscription] = []
        self._buffers: dict[str, TopicBuffer] = {}
        self._seq: dict[str, int] = {}
        self._retention: list[tuple[str, float]] = []

    def subscribe(self, pattern: str, callback: Callable[[Message], None]) -> Subscription:
        sub = Subscription(pattern, callback)
        self._subs.append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        self._subs.remove(sub)

    def set_retention(self, pattern: str, seconds: float) -> None:
        """Keep `seconds` of history for matching topics (later calls win)."""
        self._retention.append((pattern, seconds))
        for topic, buf in self._buffers.items():
            if fnmatchcase(topic, pattern):
                buf.keep_seconds = seconds

    def publish(self, topic: str, data: Mapping[str, Any], frame_id: str = "world",
                stamp: tuple[float, int] | None = None) -> Message:
        """Stamp, buffer and deliver a message. `stamp` = (sim_time, step) of the state the data
        describes; defaults to the clock's current state."""
        sim_time, step = stamp if stamp is not None else (self.clock.sim_time, self.clock.step)
        seq = self._seq.get(topic, 0)
        self._seq[topic] = seq + 1
        msg = Message(topic, Header(sim_time, self.clock.wall(), step, seq, frame_id), freeze(data))
        self._buffer(topic).add(msg)
        for sub in list(self._subs):
            if fnmatchcase(topic, sub.pattern):
                sub.callback(msg)
        return msg

    def latest(self, topic: str) -> Message | None:
        buf = self._buffers.get(topic)
        return buf.messages[-1] if buf and buf.messages else None

    def window(self, topic: str, seconds: float) -> list[Message]:
        """Messages of `topic` from the last `seconds` of sim time, oldest first."""
        buf = self._buffers.get(topic)
        return buf.since(self.clock.sim_time - seconds) if buf else []

    def topics(self) -> list[str]:
        return sorted(self._buffers)

    def clear(self) -> None:
        """Drop history and sequence numbers; subscriptions stay."""
        self._buffers.clear()
        self._seq.clear()

    def fork(self, clock: Clock) -> "EventBus":
        """A bus with the same history and settings but no subscribers (for world copies)."""
        twin = EventBus(clock, self.default_retention)
        twin._retention = list(self._retention)
        twin._seq = dict(self._seq)
        for topic, buf in self._buffers.items():
            copy = TopicBuffer(buf.keep_seconds)
            copy.messages = deque(buf.messages)  # messages are immutable, so sharing is safe
            twin._buffers[topic] = copy
        return twin

    def _buffer(self, topic: str) -> TopicBuffer:
        if topic not in self._buffers:
            keep = self.default_retention
            for pattern, seconds in self._retention:
                if fnmatchcase(topic, pattern):
                    keep = seconds
            self._buffers[topic] = TopicBuffer(keep)
        return self._buffers[topic]

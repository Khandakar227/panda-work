"""Event bus: stamping, delivery order, buffers, and deterministic encoding."""
import numpy as np
import pytest

from pandawork.world import Clock, EventBus, StreamDigest


class FakeWall:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        self.t += 0.5
        return self.t


@pytest.fixture
def bus():
    return EventBus(Clock(FakeWall()))


def test_publish_stamps_with_clock_and_per_topic_seq(bus):
    bus.clock.advance(0.002)
    a0 = bus.publish("/a", {"x": 1.0}, frame_id="base")
    a1 = bus.publish("/a", {"x": 2.0})
    b0 = bus.publish("/b", {"x": 3.0})
    assert (a0.header.sim_time, a0.header.step, a0.header.frame_id) == (0.002, 1, "base")
    assert [a0.header.seq, a1.header.seq, b0.header.seq] == [0, 1, 0]
    assert a1.header.wall_time > a0.header.wall_time


def test_explicit_stamp_overrides_clock(bus):
    msg = bus.publish("/a", {}, stamp=(0.5, 250))
    assert (msg.header.sim_time, msg.header.step) == (0.5, 250)


def test_delivery_is_synchronous_in_subscription_order_with_globs(bus):
    seen = []
    bus.subscribe("*", lambda m: seen.append(("all", m.topic)))
    bus.subscribe("/cam/*", lambda m: seen.append(("cam", m.topic)))
    sub = bus.subscribe("/joints", lambda m: seen.append(("joints", m.topic)))
    bus.publish("/cam/front/rgb", {})
    bus.publish("/joints", {})
    bus.unsubscribe(sub)
    bus.publish("/joints", {})
    assert seen == [("all", "/cam/front/rgb"), ("cam", "/cam/front/rgb"),
                    ("all", "/joints"), ("joints", "/joints"), ("all", "/joints")]


def test_processor_can_publish_from_a_callback(bus):
    order = []
    bus.subscribe("/raw", lambda m: bus.publish("/derived", {"v": m.data["v"] * 2}))
    bus.subscribe("*", lambda m: order.append(m.topic))
    bus.publish("/raw", {"v": 3})
    assert order == ["/derived", "/raw"]  # nested publish is delivered before returning
    assert bus.latest("/derived").data["v"] == 6


def test_buffer_keeps_retention_window_and_latest(bus):
    bus.set_retention("/fast", 0.01)
    for _ in range(10):
        bus.clock.advance(bus.clock.sim_time + 0.002)
        bus.publish("/fast", {"t": bus.clock.sim_time})
    kept = bus.window("/fast", 1.0)
    assert [round(m.data["t"], 3) for m in kept] == [0.01, 0.012, 0.014, 0.016, 0.018, 0.02]
    assert bus.latest("/fast").data["t"] == pytest.approx(0.02)
    assert [round(m.data["t"], 3) for m in bus.window("/fast", 0.004)] == [0.016, 0.018, 0.02]


def test_published_arrays_are_frozen_copies(bus):
    source = np.zeros(3)
    msg = bus.publish("/a", {"v": source})
    source[0] = 99
    assert msg.data["v"][0] == 0
    with pytest.raises(ValueError):
        msg.data["v"][0] = 1


def test_encoding_is_deterministic_and_can_exclude_wall_time():
    def run():
        b = EventBus(Clock(FakeWall()))
        digest = StreamDigest()
        b.subscribe("*", digest)
        b.publish("/a", {"v": np.arange(3.0), "n": 2, "s": "x", "ok": True})
        return digest.hexdigest(), b.latest("/a")

    (d1, m1), (d2, m2) = run(), run()
    assert d1 == d2
    assert m1.encode(include_wall_time=False) == m2.encode(include_wall_time=False)
    m_float = EventBus(Clock()).publish("/a", {"v": np.arange(3.0)})
    m_int = EventBus(Clock()).publish("/a", {"v": np.arange(3)})
    assert m_float.encode(False) != m_int.encode(False)  # dtype is part of the encoding

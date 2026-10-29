"""World: timing, sensors, noise per tier, determinism and forking."""
import numpy as np
import pytest

from pandawork.core.robot import GRIPPER_CLOSED
from pandawork.tiers import Tier
from pandawork.world import JointStateSensor, StreamDigest, World, WristFTSensor

SCENARIO_STEPS = 600


def _scenario(world: World) -> None:
    """Move two joints and close the gripper: exercises motion, torques and touch."""
    target = world.q_cmd.copy()
    target[[0, 3]] += [0.3, 0.2]
    world.step(100)
    world.command(q=target)
    world.step(300)
    world.command(gripper=GRIPPER_CLOSED)
    world.step(SCENARIO_STEPS - 400)


def _digest(tier=Tier.REALISTIC, seed=7, scenario=_scenario, with_camera=False) -> str:
    world = World(tier=tier, seed=seed, sensors=_light_sensors(with_camera))
    digest = StreamDigest()
    world.bus.subscribe("*", digest)
    scenario(world)
    if with_camera:
        world.capture("front_rgb")
    return digest.hexdigest()


def _light_sensors(with_camera=False):
    from pandawork.world import CameraSensor, TouchSensor
    sensors = [JointStateSensor(), WristFTSensor(), TouchSensor()]
    return sensors + [CameraSensor("front", "rgb", (160, 120))] if with_camera else sensors


@pytest.fixture
def world():
    w = World(seed=0, sensors=_light_sensors())
    yield w
    w.close()


def test_periodic_messages_describe_the_state_they_were_read_from(world):
    world.step(10)
    msgs = world.bus.window("/joints", 1.0)
    assert [m.header.step for m in msgs] == list(range(10))
    assert msgs[-1].header.sim_time == pytest.approx(9 * world.timestep)
    assert world.clock.step == 10


def test_sensor_rate_controls_publication():
    slow = World(seed=0, sensors=[JointStateSensor(rate_hz=100.0)])
    slow.step(20)
    assert [m.header.step for m in slow.bus.window("/joints", 1.0)] == [0, 5, 10, 15]


def test_time_only_moves_inside_step(world):
    world.step(5)
    before = (world.clock.step, world.data.time)
    world.bus.latest("/joints")
    world.command(gripper=GRIPPER_CLOSED)
    assert (world.clock.step, world.data.time) == before


def test_holding_still_with_gravity_compensation(world):
    q0 = world.data.qpos[world.qadr].copy()
    world.step(500)
    assert np.abs(world.data.qpos[world.qadr] - q0).max() < 1e-4


def test_privileged_is_exact_and_realistic_is_noisy():
    exact = World(tier=Tier.PRIVILEGED, seed=3, sensors=_light_sensors())
    noisy = World(tier=Tier.REALISTIC, seed=3, sensors=_light_sensors())
    for w in (exact, noisy):
        w.step(50)
    # Same physics in both worlds; only the readings differ.
    assert np.array_equal(exact.data.qpos, noisy.data.qpos)
    q_exact = exact.bus.latest("/joints").data["q"]
    q_noisy = noisy.bus.latest("/joints").data["q"]
    assert not np.array_equal(q_exact, q_noisy)
    assert np.abs(q_noisy - q_exact).max() < 1e-3  # noise is small (std 1e-4 rad)
    assert noisy.bus.latest("/touch").data["left"] >= 0.0  # clipped at zero


def test_same_seed_gives_identical_stream():
    assert _digest(seed=7) == _digest(seed=7)


def test_different_seed_changes_only_the_noise():
    assert _digest(seed=7) != _digest(seed=8)
    assert _digest(Tier.PRIVILEGED, seed=7) == _digest(Tier.PRIVILEGED, seed=8)


def test_same_seed_identical_stream_including_camera(cameras):
    assert _digest(seed=7, with_camera=True) == _digest(seed=7, with_camera=True)


def test_reset_restores_the_initial_stream():
    world = World(tier=Tier.REALISTIC, seed=7, sensors=_light_sensors())
    first, second = StreamDigest(), StreamDigest()
    sub = world.bus.subscribe("*", first)
    _scenario(world)
    world.bus.unsubscribe(sub)
    world.reset(seed=7)
    world.bus.subscribe("*", second)
    _scenario(world)
    assert first.hexdigest() == second.hexdigest()


def test_fork_is_independent_and_reproduces_the_original():
    world = World(tier=Tier.REALISTIC, seed=5, sensors=_light_sensors())
    world.step(200)
    twin = world.fork()
    state = (world.clock.step, world.data.qpos.copy())

    twin.command(gripper=GRIPPER_CLOSED)
    twin.step(300)
    assert world.clock.step == state[0] and np.array_equal(world.data.qpos, state[1])
    assert twin.bus.latest("/joints").header.step > world.bus.latest("/joints").header.step

    # Stepping a fresh fork exactly like the original yields an identical stream.
    a, b = world.fork(), world.fork()
    da, db = StreamDigest(), StreamDigest()
    a.bus.subscribe("*", da)
    b.bus.subscribe("*", db)
    for w in (a, b):
        w.command(gripper=GRIPPER_CLOSED)
        w.step(300)
    assert da.hexdigest() == db.hexdigest()


def test_fork_has_no_subscribers_but_keeps_history():
    world = World(seed=0, sensors=_light_sensors())
    seen = []
    world.bus.subscribe("*", seen.append)
    world.step(10)
    twin = world.fork()
    n = len(seen)
    twin.step(10)
    assert len(seen) == n  # predictions never reach the original's logger
    assert twin.bus.latest("/joints").header.seq == 19  # history and sequence numbers continue


def test_camera_capture_is_on_demand_and_stamped_now(cameras):
    world = World(seed=0)
    world.step(3)
    msg = world.capture("wrist_rgb")
    assert msg.header.step == 3 and msg.header.frame_id == "wrist_camera"
    assert msg.data["image"].shape == (480, 640, 3)
    assert msg.data["fovy"] == pytest.approx(75.0)
    assert world.bus.latest("/cam/front/rgb") is None  # nothing renders unless asked
    world.close()

"""Motion skills: which grasp yaw is used, and how smoothly straight-line moves run."""
import numpy as np

from pandawork.core.robot import yaw_quat
from pandawork.skills import motion
from pandawork.skills.motion import solve_grasp_ik, wrap_angle
from pandawork.skills.sensing import robot_state
from pandawork.tasks import make_episode
from pandawork.tasks.base import call
from pandawork.tiers import Tier

ABOVE = np.array([0.5, 0.0, 0.2])


def test_half_turn_twin_is_used_when_it_needs_less_wrist_rotation():
    world = make_episode("reach", seed=0).world
    state = robot_state(world)
    now, quat = state.tcp_yaw, state.tcp_quat
    *_, used = solve_grasp_ik(world, ABOVE, now + np.radians(170), quat)
    assert abs(wrap_angle(used - (now - np.radians(10)))) < 1e-9  # same grasp, 10 deg of turn
    *_, used = solve_grasp_ik(world, ABOVE, now + np.radians(30), quat)
    assert abs(wrap_angle(used - (now + np.radians(30)))) < 1e-9  # the requested yaw itself
    *_, used = solve_grasp_ik(world, ABOVE, now + np.radians(95), quat)
    assert abs(wrap_angle(used - (now + np.radians(95)))) < 1e-9  # near-tie: requested wins


def test_straight_move_caps_joint_speed():
    """Regression: pulling the hand in toward the base, a straight line needs ever faster
    joints (2.1 rad/s here uncapped); such jolts read on the wrist like a collision."""
    world = make_episode("reach", seed=0).world
    motion.move_tcp(world, np.array([0.5, 0.0, 0.3]), 0.0)
    peak = []
    world.step_hooks.append(lambda w: peak.append(np.abs(w.data.qvel[:7]).max()))
    outcome = motion.move_tcp(world, np.array([0.2, 0.0, 0.3]), 0.0, speed=0.3)
    assert outcome.status == "reached", outcome
    assert max(peak) < 1.1 * motion.MAX_JOINT_SPEED


def test_tracking_keeps_joints_inside_their_limits():
    world = make_episode("reach", seed=0).world
    belief = world.belief
    q = world.q_cmd.copy()
    for yaw in np.radians(np.arange(0, 181, 5)):  # turn the hand until the wrist runs out
        q, _ = belief.track(q, ABOVE, yaw_quat(yaw))
        assert np.all(q >= belief.q_lo) and np.all(q <= belief.q_hi)
    hard_lo, hard_hi = world.plant.jnt_range[:7].T
    assert np.all(belief.q_lo > hard_lo) and np.all(belief.q_hi < hard_hi)  # soft limits


def test_fixture_states_only_in_the_privileged_tier():
    ep = make_episode("open_drawer", seed=0)
    (result,) = call(ep.dispatcher, ("get_fixture_states", {}))
    assert "drawer_opening_cm" in result["content"][0]["text"]
    (result,) = call(make_episode("reach", seed=0).dispatcher, ("get_fixture_states", {}))
    assert "no mechanisms" in result["content"][0]["text"]
    realistic = make_episode("open_drawer", tier=Tier.REALISTIC, seed=0)
    assert "get_fixture_states" not in {t.name for t in realistic.dispatcher.registry.available(Tier.REALISTIC)}


def test_pointing_forward_and_moving_relative_keep_the_orientation():
    from pandawork.core.robot import pointing_of, quat_yaw
    from pandawork.tasks.base import call
    ep = make_episode("press_key", seed=0)  # starts pointing forward, in front of the keypad
    world = ep.world
    assert pointing_of(robot_state(world).tcp_quat) == "forward"
    (result,) = call(ep.dispatcher, ("move_to", dict(x=0.42, y=0.05, z=0.76, pointing="forward",
                                                     yaw_deg=90)))
    assert "status: reached" in result["content"][0]["text"]
    state = robot_state(world)
    assert pointing_of(state.tcp_quat) == "forward"
    assert abs(np.degrees(quat_yaw(state.tcp_quat, "forward")) - 90) < 2
    (result,) = call(ep.dispatcher, ("move_relative", {"dz": -0.03}))
    state = robot_state(world)
    assert pointing_of(state.tcp_quat) == "forward" and abs(state.tcp_pos[2] - 0.73) < 0.003


def test_yaw_round_trips_for_both_pointings():
    from pandawork.core.robot import pointing_of, quat_yaw
    for pointing in ("down", "forward"):
        for yaw in np.radians([-150, -45, 0, 30, 90, 179]):
            quat = yaw_quat(yaw, pointing)
            assert pointing_of(quat) == pointing
            assert abs(wrap_angle(quat_yaw(quat, pointing) - yaw)) < 1e-9

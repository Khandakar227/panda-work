"""Robot tools through the real dispatcher on the demo scene."""
import json

import numpy as np
import pytest

from pandawork.core.robot import ARM_JOINT_LIMITS, ARM_JOINTS
from pandawork.core.scene import DEMO_OBJECTS, build_scene
from pandawork.tiers import Tier
from pandawork.tools import (
    Budget, BudgetLimits, Dispatcher, ToolCall, ToolContext, default_registry,
)
from pandawork.world import StreamDigest, World

P, R, V = Tier.PRIVILEGED, Tier.REALISTIC, Tier.VISION_ONLY


def make(tier=P, seed=0) -> Dispatcher:
    self_model, plant = build_scene(DEMO_OBJECTS)
    world = World(plant, self_model, tier=tier, seed=seed)
    ctx = ToolContext(world=world, budget=Budget(BudgetLimits(max_calls=None), world.clock))
    return Dispatcher(default_registry(), ctx)


def call(d: Dispatcher, name: str, **args) -> tuple[dict, str]:
    """Run one tool; returns (structured result data, text the model saw)."""
    result, content = d.run(ToolCall("c", name, args))
    text = "\n".join(c["text"] for c in content if c["type"] == "text")
    return {**result.data, "_ok": result.ok, "_images": result.images}, text


def cube(d: Dispatcher, name="red_cube") -> np.ndarray:
    return d.ctx.world.object_pose(name)[0]


# ---- registry ------------------------------------------------------------------------------

def test_joint_limit_constants_match_the_model():
    _, plant = build_scene()
    for name, (lo, hi) in zip(ARM_JOINTS, ARM_JOINT_LIMITS):
        assert np.allclose(plant.joint(name).range, [lo, hi], atol=1e-4)


def test_tools_per_tier():
    reg = default_registry()
    names = {t: {tool.name for tool in reg.available(t)} for t in Tier}
    assert len(names[P]) == 16 and len(names[R]) == 13 and len(names[V]) == 9
    assert names[P] - names[R] == {"get_object_poses", "get_fixture_states", "simulate"}
    assert names[V] == {"get_robot_state", "look", "move_to", "move_relative", "gripper", "wait",
                        "check_reachable", "get_task", "done"}
    json.dumps(reg.schemas(P))  # every schema is serialisable


# ---- perceive ------------------------------------------------------------------------------

def test_robot_state_per_tier():
    p, _ = call(make(P), "get_robot_state")
    assert np.allclose(p["fingertip"], p["fingertip_true"], atol=1e-3)
    r, _ = call(make(R), "get_robot_state")
    assert "fingertip" in r and "fingertip_true" not in r
    v, _ = call(make(V), "get_robot_state")
    assert set(k for k in v if not k.startswith("_")) == {"gripper_open_cm", "pointing"}
    assert v["pointing"] == "down"


def test_wrist_ft_reads_zero_at_rest_and_the_held_load():
    d = make(R)
    rest, _ = call(d, "read_wrist_ft")
    assert rest["magnitude_N"] < 0.5
    c = cube(d)
    for name, args in [("move_to", dict(x=c[0], y=c[1], z=c[2] + 0.1)), ("gripper", {"action": "open"}),
                       ("move_to", dict(x=c[0], y=c[1], z=c[2])), ("gripper", {"action": "close"}),
                       ("move_to", dict(x=c[0], y=c[1], z=c[2] + 0.15))]:
        call(d, name, **args)
    held, _ = call(d, "read_wrist_ft")
    assert held["held_load_kg"] == pytest.approx(0.1, abs=0.03)  # the cube weighs 100 g
    window, _ = call(d, "read_wrist_ft", window_s=0.5)
    assert window["samples"] > 100


def test_contacts_by_name_only_when_privileged():
    for tier in (P, R):
        d = make(tier)
        c = cube(d)
        call(d, "move_to", x=c[0], y=c[1], z=c[2] + 0.1)
        call(d, "move_to", x=c[0], y=c[1], z=c[2])
        call(d, "gripper", action="close")
        data, text = call(d, "get_contacts")
        if tier is P:
            assert "left_finger - red_cube" in text and "right_finger - red_cube" in text
        else:
            assert data["left_pad_N"] > 1 and data["right_pad_N"] > 1 and "red_cube" not in text


def test_look_sizes_crop_and_tier_modes(cameras):
    d = make(P)
    data, text = call(d, "look", camera="front")
    assert data["_images"][0].shape == (240, 320, 3) and "front camera, rgb, 320x240" in text
    data, _ = call(d, "look", camera="top", size="medium", crop=[0.25, 0.25, 0.75, 0.75])
    assert data["_images"][0].shape == (240, 320, 3)  # half of 640x480
    _, text = call(d, "look", camera="front", mode="depth")
    assert "white =" in text
    _, text = call(d, "look", camera="front", mode="seg")
    assert "red_cube = " in text
    _, text = call(make(R), "look", camera="front", mode="seg")
    assert "[mode_not_available]" in text
    _, text = call(make(V), "look", camera="front", mode="depth")
    assert "[mode_not_available]" in text


def test_object_poses_are_exact():
    _, text = call(make(P), "get_object_poses")
    assert "blue_cube: center [0.45, -0.15, 0.025], yaw 30.0 deg, size 5.0 x 5.0 x 5.0 cm" in text


def test_detection_matches_truth_within_noise(cameras):
    d = make(R)
    call(d, "move_joints", joints_deg=[90, 0, 0, -90, 0, 90, -45])  # arm out of the top view
    _, text = call(d, "detect_objects", camera="top")
    line = next(l for l in text.splitlines() if l.startswith("blue_cube"))
    centre = json.loads(line.split("top centre ")[1].split("]")[0] + "]")
    assert np.allclose(centre[:2], [0.45, -0.15], atol=0.015)
    assert "not detected" not in text


# ---- act -----------------------------------------------------------------------------------

def test_move_to_reports_target_and_achievement():
    data, text = call(make(P), "move_to", x=0.45, y=0.1, z=0.3, yaw_deg=30)
    assert data["status"] == "reached" and data["error_mm"] < 2
    assert data["achieved_yaw_deg"] == pytest.approx(30, abs=1)
    assert np.allclose(data["achieved"], [0.45, 0.1, 0.3], atol=0.002)
    assert "target: [0.45, 0.1, 0.3]" in text and "stop_reason: reached" in text


def test_vision_only_reports_carry_no_coordinates():
    data, _ = call(make(V), "move_to", x=0.45, y=0.1, z=0.3)
    assert data["status"] == "reached" and "achieved" not in data and "error_mm" not in data


def test_unreachable_target_fails_without_moving():
    d = make(P)
    q0 = d.ctx.world.data.qpos[d.ctx.world.qadr].copy()
    data, _ = call(d, "move_to", x=1.3, y=0.0, z=0.3)
    assert data["status"] == "failed" and data["stop_reason"] == "unreachable" and "hint" in data
    assert np.allclose(d.ctx.world.data.qpos[d.ctx.world.qadr], q0)


def test_moving_into_the_table_stops_on_contact():
    for tier in (P, R):
        data, _ = call(make(tier), "move_to", x=0.35, y=0.3, z=-0.05)
        assert data["status"] == "partial" and data["stop_reason"] == "contact"
        assert "sensed" in data
        if tier is P:
            assert any("table" in t for t in data["touching"])


def test_move_relative_in_gripper_frame_moves_along_the_fingers():
    d = make(P)
    before, _ = call(d, "get_robot_state")
    data, _ = call(d, "move_relative", dz=0.05, frame="gripper")
    assert data["status"] == "reached"
    after, _ = call(d, "get_robot_state")
    assert after["fingertip"][2] == pytest.approx(before["fingertip"][2] - 0.05, abs=0.002)


def test_pick_and_place_with_tools():
    d = make(R)
    c = cube(d)
    turns = [
        [("move_to", dict(x=c[0], y=c[1], z=0.15)), ("gripper", {"action": "open"})],
        [("move_to", dict(x=c[0], y=c[1], z=c[2])), ("gripper", {"action": "close"})],
        [("move_to", dict(x=0.4, y=0.25, z=0.15, yaw_deg=45)),
         ("move_to", dict(x=0.4, y=0.25, z=0.0, yaw_deg=45, speed=0.05))],
        [("gripper", {"action": "open"}), ("move_relative", {"dz": 0.1})],
    ]
    for turn in turns:
        d.run_batch([ToolCall(f"c{i}", name, args) for i, (name, args) in enumerate(turn)])
    assert np.allclose(cube(d)[:2], [0.4, 0.25], atol=0.01)
    assert cube(d)[2] == pytest.approx(0.025, abs=0.003)  # resting on the table


def test_gripper_reports_holding():
    d = make(V)
    data, _ = call(d, "gripper", action="close")
    assert data["holding"] is False and "touch_N" not in data  # no touch sensing in vision_only
    d = make(P)
    c = cube(d)
    call(d, "move_to", x=c[0], y=c[1], z=c[2] + 0.1)
    call(d, "move_to", x=c[0], y=c[1], z=c[2])
    data, _ = call(d, "gripper", action="close")
    assert data["holding"] is True and data["open_cm"] == pytest.approx(5.0, abs=0.3)


def test_failed_action_skips_the_rest_of_the_turn():
    d = make(P)
    blocks = d.run_batch([ToolCall("a", "move_to", {"x": 1.3, "y": 0, "z": 0.3}),
                          ToolCall("b", "gripper", {"action": "close"})])
    assert "[skipped]" in blocks[1]["content"][0]["text"]


def test_move_joints_clips_to_limits():
    data, _ = call(make(P), "move_joints", joints_deg=[0, 0, 0, -200, 0, 90, -45])
    assert data["stop_reason"] in ("joint_limit", "contact")
    assert data["status"] == "partial"


# ---- predict and meta ----------------------------------------------------------------------

def test_check_reachable():
    assert call(make(P), "check_reachable", x=0.5, y=0.0, z=0.2)[0]["reachable"] is True
    assert call(make(P), "check_reachable", x=1.3, y=0.0, z=0.2)[0]["reachable"] is False


def test_simulate_predicts_without_touching_the_real_world():
    d = make(P)
    c = cube(d)
    t0, cube0 = d.ctx.world.clock.sim_time, cube(d).copy()
    plan = [{"tool": "move_to", "args": {"x": c[0], "y": c[1], "z": 0.15}},
            {"tool": "gripper", "args": {"action": "open"}},
            {"tool": "move_to", "args": {"x": c[0], "y": c[1], "z": c[2]}},
            {"tool": "gripper", "args": {"action": "close"}},
            {"tool": "move_to", "args": {"x": c[0], "y": c[1], "z": 0.2}}]
    data, text = call(d, "simulate", actions=plan)
    assert data["_ok"], text
    predicted = json.loads(text.split("red_cube ")[1].split("]")[0] + "]")
    assert predicted[2] > 0.15  # the cube would be lifted
    assert d.ctx.world.clock.sim_time == t0 and np.allclose(cube(d), cube0)
    _, text = call(d, "simulate", actions=[{"tool": "get_robot_state", "args": {}}])
    assert "[not_simulatable]" in text


def test_task_and_done():
    d = make(P)
    data, _ = call(d, "get_task")
    assert data["task"] == "no task is loaded" and "red_cube" in data["objects"]
    call(d, "done", success=True, explanation="nothing to do")
    assert d.ctx.state["done"] and d.ctx.state["claimed_success"] is True
    _, text = call(d, "get_robot_state")
    assert "[denied]" in text


def test_tool_episode_is_deterministic(cameras):
    def run():
        d = make(R, seed=3)
        digest = StreamDigest()
        d.ctx.world.bus.subscribe("*", digest)
        d.run_batch([ToolCall("a", "move_to", {"x": 0.45, "y": 0.1, "z": 0.2}),
                     ToolCall("b", "detect_objects", {"camera": "front"}),
                     ToolCall("c", "read_wrist_ft", {})])
        return digest.hexdigest()
    assert run() == run()

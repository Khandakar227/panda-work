"""Faults: the plant differs from the robot's self-model, and the tools keep reporting the belief."""
import numpy as np
import pytest

from pandawork.core.faults import FAULTS, NO_FAULT, EncoderOffset, fault_from_meta
from pandawork.skills.sensing import robot_state, true_tcp
from pandawork.tasks import make_episode
from pandawork.tasks.base import call

TARGET = dict(x=0.5, y=0.1, z=0.2)


def _move(ep, **kw):
    (result,) = call(ep.dispatcher, ("move_to", {**TARGET, **kw}))
    return result["content"][0]["text"]


def test_no_fault_is_the_default_and_leaves_the_episode_unchanged():
    ep = make_episode("reach", seed=0)
    assert ep.world.fault is NO_FAULT
    assert make_episode("reach", seed=0, fault=NO_FAULT).world.plant is ep.world.plant  # cached


def test_encoder_offset_makes_reached_reports_false():
    """The controller servos the encoder to the target, so the belief lands on the target while
    the body is somewhere else: the tool report says reached, the hand is centimetres away."""
    ep = make_episode("reach", seed=0, fault=FAULTS["encoder_j1_5deg"])
    text = _move(ep)
    assert "status: reached" in text and "error_mm" in text
    believed, truth = robot_state(ep.world).tcp_pos, true_tcp(ep.world)[0]
    assert np.linalg.norm(believed - [0.5, 0.1, 0.2]) < 0.003  # the belief is on target
    assert np.linalg.norm(truth - believed) > 0.02               # the body is not


def test_encoder_offset_is_invisible_in_realistic_readings():
    from pandawork.tiers import Tier
    ep = make_episode("reach", tier=Tier.REALISTIC, seed=0, fault=FAULTS["encoder_j1_5deg"])
    _move(ep)
    (state,) = call(ep.dispatcher, ("get_robot_state", {}))
    text = state["content"][0]["text"]
    assert "fingertip: [0.5, 0.1, 0.2" in text and "true" not in text  # the belief, nothing else


def test_fingertip_offset_moves_the_body_not_the_belief():
    ep = make_episode("reach", seed=0, fault=FAULTS["fingertip_2cm_long"])
    _move(ep)
    believed, truth = robot_state(ep.world).tcp_pos, true_tcp(ep.world)[0]
    assert abs(np.linalg.norm(truth - believed) - 0.02) < 0.002
    assert truth[2] < believed[2]  # longer fingers reach lower when pointing down


def test_short_gripper_is_revealed_by_the_gripper_report():
    ep = make_episode("reach", seed=0, fault=FAULTS["gripper_5cm"])
    (result,) = call(ep.dispatcher, ("gripper", {"action": "open"}))
    assert "open_cm: 5" in result["content"][0]["text"]


def test_stiff_joint_is_revealed_by_a_partial_move():
    ep = make_episode("reach", seed=0, fault=FAULTS["stiff_j4"])
    ok = make_episode("reach", seed=0)
    far = dict(x=0.3, y=0.0, z=0.5)  # high and close: the elbow must fold to -133 deg
    (good,) = call(ok.dispatcher, ("move_to", far))
    (bad,) = call(ep.dispatcher, ("move_to", far))
    assert "status: reached" in good["content"][0]["text"]
    assert "status: partial" in bad["content"][0]["text"]


def test_oracle_under_a_hidden_fault_fails_honestly():
    """The oracle trusts the tools: with the hand 3 cm off it places the cube wrong, and the
    world check says so."""
    ep = make_episode("pick_place", seed=0, fault=FAULTS["encoder_j1_5deg"])
    ep.task.oracle(ep.dispatcher)
    assert not ep.evaluate().success


def test_faults_round_trip_through_meta_and_replay(tmp_path):
    from pandawork.recording import EpisodeRecorder, load, replay
    ep = make_episode("reach", seed=1, fault=FAULTS["encoder_j1_5deg"])
    rec = EpisodeRecorder(ep, tmp_path / "ep")
    _move(ep)
    rec.finish()
    log = load(tmp_path / "ep")
    assert log.meta["fault"]["name"] == "encoder_j1_5deg"
    assert fault_from_meta(log.meta) == FAULTS["encoder_j1_5deg"]
    assert replay(log).identical


def test_fault_presets_are_frozen_and_named():
    assert FAULTS["none"] is NO_FAULT
    f = FAULTS["encoder_j1_5deg"]
    assert isinstance(f, EncoderOffset) and f.name == "encoder_j1_5deg"
    with pytest.raises((AttributeError, TypeError)):
        f.joint = 3  # type: ignore[misc]
    assert fault_from_meta({}) is NO_FAULT  # old logs have no fault entry


def test_simulate_predicts_the_believed_robot_not_the_faulty_body():
    """simulate must not become a fault oracle: it runs the plan on a twin built from the
    nominal plant in the believed state, so under a hidden fault it predicts exactly what it
    would predict with no fault."""
    plan = [{"tool": "move_to", "args": {"x": 0.45, "y": 0.0, "z": 0.1, "speed": 0.3}},
            {"tool": "gripper", "args": {"action": "close"}}]
    texts = []
    for fault in (NO_FAULT, FAULTS["encoder_j1_5deg"], FAULTS["fingertip_2cm_long"]):
        ep = make_episode("pick_place", seed=0, fault=fault)
        before = ep.world.data.qpos.copy()
        (result,) = call(ep.dispatcher, ("simulate", {"actions": plan}))
        assert np.array_equal(ep.world.data.qpos, before)  # the real robot did not move
        texts.append(result["content"][0]["text"])
    assert texts[0] == texts[1] == texts[2], texts


def test_detections_are_interpreted_with_the_believed_camera_pose():
    """Images are real, but where a camera is (and so where a pixel points) comes from the
    self-model. A wrist-camera fault, or an encoder offset moving the wrist, skews detections
    from the wrist camera; a fixed camera is unaffected."""
    from pandawork.skills.perception import detect_objects, true_object_poses
    def errors(fault, camera):
        ep = make_episode("pick_place", seed=0, fault=fault)
        _move(ep, x=0.5, y=0.0, z=0.35)  # look down at the scene from above
        truth = {p.name: p.position[:2] for p in true_object_poses(ep.world)}
        found, _ = detect_objects(ep.world, camera)
        # The cube only: the arm (really standing elsewhere) can part-hide the flat pad.
        (cube,) = [d for d in found if d.name == "red_cube"]
        return float(np.linalg.norm(cube.center_xy - truth["red_cube"]))
    assert errors(NO_FAULT, "wrist") < 0.01
    assert errors(FAULTS["wrist_cam_10deg"], "wrist") > 0.02
    assert errors(FAULTS["encoder_j1_5deg"], "wrist") > 0.02
    assert errors(FAULTS["encoder_j1_5deg"], "top") < 0.01


@pytest.mark.parametrize("tier", ["realistic", "vision_only"])
def test_leak_audit_no_unprivileged_tool_reveals_the_true_hand(tier):
    """Under a hidden fault, every tool a non-privileged tier offers must describe the believed
    robot. The true fingertip position must not appear in any result text."""
    from pandawork.tiers import Tier
    from pandawork.tools.robot_tools import _m
    ep = make_episode("pick_place", tier=Tier(tier), seed=0, fault=FAULTS["encoder_j1_5deg"])
    _move(ep, x=0.5, y=0.0, z=0.3)
    believed, truth = robot_state(ep.world).tcp_pos, true_tcp(ep.world)[0]
    assert np.linalg.norm(truth - believed) > 0.02
    probes = {"get_robot_state": {}, "read_wrist_ft": {}, "get_contacts": {}, "get_task": {},
              "look": {"camera": "wrist"}, "detect_objects": {"camera": "wrist"},
              "check_reachable": {"x": 0.5, "y": 0.0, "z": 0.3}, "wait": {"seconds": 0.1},
              "move_to": {"x": 0.5, "y": 0.0, "z": 0.3},
              "move_relative": {"dz": -0.02}, "gripper": {"action": "close"},
              "move_joints": {"joints_deg": [0, -45, 0, -135, 0, 90, 45]}}
    texts = []
    for tool in ep.dispatcher.registry.available(Tier(tier)):
        if tool.name == "done":
            continue
        (result,) = call(ep.dispatcher, (tool.name, probes[tool.name]))
        texts.append("\n".join(b["text"] for b in result["content"] if b["type"] == "text"))
    text = "\n".join(texts)
    tx, ty, tz = _m(truth)
    assert f"{tx}, {ty}" not in text and f"{ty}, {tz}" not in text, text  # no true position
    assert "fingertip_true" not in text
    assert ("fingertip: [" in text) == (tier == "realistic")  # the belief, where coordinates show

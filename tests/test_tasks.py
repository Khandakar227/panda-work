"""Tasks: seeded layouts, honest success checks, and oracles that solve them through the tools."""
import numpy as np
import pytest

from pandawork.tasks import TASKS, make_episode
from pandawork.tasks.base import call
from pandawork.tiers import Tier
from pandawork.tools import ToolCall


def layout(ep) -> dict:
    world = ep.world
    return {name: np.round(world.object_pose(name)[0], 4).tolist() for name in world.objects}


@pytest.mark.parametrize("task", list(TASKS))
def test_layout_is_reproducible_per_seed_and_varies_across_seeds(task):
    a, b, c = make_episode(task, seed=4), make_episode(task, seed=4), make_episode(task, seed=5)
    assert layout(a) == layout(b) and a.task.instruction == b.task.instruction
    assert layout(a) != layout(c)


@pytest.mark.parametrize("task", list(TASKS))
def test_doing_nothing_fails(task):
    assert not make_episode(task, seed=0).evaluate().success


@pytest.mark.parametrize("task", list(TASKS))
@pytest.mark.parametrize("seed", [0, 1])
def test_oracle_solves_the_task_through_tools(task, seed):
    ep = make_episode(task, seed=seed)
    ep.task.oracle(ep.dispatcher)
    result = ep.evaluate()
    assert result.success, result.summary


def test_instruction_reaches_the_agent():
    ep = make_episode("pick_place", seed=2)
    result, _ = ep.dispatcher.run(ToolCall("c", "get_task", {}))
    assert result.data["task"] == ep.task.instruction and "green_pad" in result.data["objects"]


def test_budget_follows_the_task():
    ep = make_episode("reach", seed=0)
    assert ep.dispatcher.ctx.budget.limits.max_sim_seconds == TASKS["reach"].max_seconds


def test_pressing_the_wrong_button_fails():
    ep = make_episode("press_button", seed=0)
    target = ep.task.target
    ep.task.target = next(b for b in ep.task.BUTTONS if b != target)  # oracle presses the other one
    ep.task.oracle(ep.dispatcher)
    ep.task.target = target
    assert not ep.evaluate().success


def test_simulated_press_does_not_count():
    ep = make_episode("press_button", seed=0)
    cap = ep.world.object_pose(ep.task.target)[0]
    blocks = call(ep.dispatcher, ("simulate", {"actions": [
        {"tool": "gripper", "args": {"action": "close"}},
        {"tool": "move_to", "args": {"x": cap[0], "y": cap[1], "z": cap[2] + 0.08}},
        {"tool": "move_to", "args": {"x": cap[0], "y": cap[1], "z": cap[2] - 0.02, "speed": 0.05}}]}))
    assert "is_error" not in blocks[0]
    assert not ep.task.pressed(ep.task.target) and not ep.evaluate().success


def test_holding_the_cube_over_the_pad_is_not_placing_it():
    ep = make_episode("pick_place", seed=1)
    world, task = ep.world, ep.task
    cube, pad = world.object_pose(task.target)[0], world.object_pose("green_pad")[0]
    call(ep.dispatcher, ("gripper", {"action": "open"}),
         ("move_to", dict(x=cube[0], y=cube[1], z=0.15)))
    call(ep.dispatcher, ("move_to", dict(x=cube[0], y=cube[1], z=0.025)), ("gripper", {"action": "close"}),
         ("move_to", dict(x=pad[0], y=pad[1], z=0.15)))
    result = ep.evaluate()
    assert not result.success and not result.metrics["resting"]


def test_hovering_over_the_wrong_cube_fails():
    ep = make_episode("reach", seed=3)
    other = next(n for n in ep.world.objects if n != ep.task.target)
    p = ep.world.object_pose(other)[0]
    call(ep.dispatcher, ("move_to", dict(x=p[0], y=p[1], z=p[2] + 0.075)))
    assert not ep.evaluate().success


def test_realistic_episode_hides_privileged_tools():
    ep = make_episode("open_drawer", tier=Tier.REALISTIC, seed=0)
    names = {t.name for t in ep.dispatcher.registry.available(ep.world.tier)}
    assert "get_object_poses" not in names and "detect_objects" in names


def test_key_presses_count_once_until_the_key_comes_back_out():
    from pandawork.core.scene import KEY_PRESSED, KEY_RELEASED
    ep = make_episode("press_key", seed=0)
    task, data = ep.task, ep.world.data
    adr = task.joints[5]
    for travel in (KEY_PRESSED + 0.002, KEY_PRESSED - 0.002, KEY_PRESSED + 0.002,  # held: one press
                   KEY_RELEASED - 0.001, KEY_PRESSED + 0.001):                    # out, in: a second
        data.qpos[adr] = travel
        task.monitor(ep.world)
    assert task.typed == [5, 5]


def test_keys_wear_their_own_labels_and_shuffle_per_seed():
    ep = make_episode("type_code", seed=1)
    model = ep.world.plant
    for digit in range(1, 10):
        assert model.geom_matid[model.geom(f"obj_key_{digit}").id] == model.material(f"key_label_{digit}").id
    layouts = {tuple(make_episode("type_code", seed=s).task.layout) for s in range(4)}
    assert len(layouts) == 4
    assert make_episode("press_key", seed=2).task.layout == list(range(1, 10))  # phone layout


def test_type_code_can_be_given_its_code():
    from pandawork.tasks.library import TypeCodeTask
    assert make_episode("type_code", seed=0).task.code == "7917"  # drawn from the seed
    ep = make_episode(TypeCodeTask(code="2468"), seed=0)
    assert ep.task.code == "2468" and "2468" in ep.task.instruction
    assert ep.task.layout == make_episode("type_code", seed=0).task.layout  # same keypad
    ep.task.oracle(ep.dispatcher)
    assert ep.evaluate().success, ep.evaluate().summary
    with pytest.raises(ValueError):
        TypeCodeTask(code="1230")  # the keypad has no 0

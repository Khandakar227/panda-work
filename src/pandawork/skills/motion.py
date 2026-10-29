"""Robot motions with an inner closed loop: the skills behind the ACT tools.

Every motion updates the joint-servo targets at 50 Hz and checks the robot's own sensors each
control period: it stops early on an unexpected contact and finishes with an Outcome comparing
what was asked with what was achieved. Only self-knowledge is used (joint readings + self-model,
wrist force/torque, fingertip touch), so the loop is legitimate in every access tier.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..core.ik import pose_error
from ..core.robot import GRIPPER_MAX_WIDTH, READY_FORWARD_DEG, gripper_ctrl, quat_yaw, yaw_quat
from ..world import World
from .sensing import TOUCH_CONTACT_N, external_force, latest, robot_state, touch

CONTROL_STEPS = 10          # physics steps per control period: 20 ms at the 2 ms timestep
WATCH_EVERY = 2             # contact check every 2 physics steps (4 ms)
CONTACT_FORCE_N = 8.0       # growth of external wrist force treated as an unexpected contact
                            # (free motion peaks ~1-2 N; releasing a held object's table
                            # preload while lifting reaches ~4 N)
CONTACT_PERSIST = 2         # consecutive checks (8 ms) the force must stay above it
ROT_SPEED = np.radians(90)  # yaw rate during Cartesian moves (rad/s)
JOINT_SPEED = np.radians(60)
MAX_JOINT_SPEED = 1.0       # rad/s, cap on any joint during a straight-line move (the Panda
                            # allows 2.2-2.6): near a singularity a small hand step needs a
                            # large joint step, and that jolt reads on the wrist like a collision
MIN_DURATION = 0.3          # s, shortest motion
SETTLE_TIMEOUT = 1.5        # s allowed after the path ends to come to rest on target
GRASP_MIN_WIDTH = 0.004     # m; fingers closed tighter than this hold nothing
ANGLE_TOLERANCE = np.radians(2.0)

Check = Callable[[], None] | None  # called every control period (e.g. the tool's deadline)


@dataclass
class Outcome:
    status: str        # "reached" | "partial" | "failed"
    stop_reason: str   # "reached" | "contact" | "unreachable" | "timeout" | "joint_limit"
    duration: float    # sim seconds spent
    details: dict = field(default_factory=dict)
    hint: str | None = None


def min_jerk(tau: float) -> float:
    tau = min(max(tau, 0.0), 1.0)
    return tau ** 3 * (10 - 15 * tau + 6 * tau ** 2)


def slerp(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(dot)
    return (np.sin((1 - s) * theta) * q0 + np.sin(s * theta) * q1) / np.sin(theta)


def quat_angle(q0: np.ndarray, q1: np.ndarray) -> float:
    return float(2 * np.arccos(min(1.0, abs(float(np.dot(q0, q1))))))


class ContactWatcher:
    """Flags an unexpected contact: the external wrist force grows more than CONTACT_FORCE_N
    above its magnitude at the start of the motion (so a held object's weight is not a contact),
    or, while the fingers are open and empty, a fingertip pad starts touching something."""

    def __init__(self, world: World):
        self.world = world
        self.baseline = external_force(world)
        # Pad touches only mean "bumped into something" while the fingers are apart and empty;
        # with the gripper closed (e.g. pressing a button) contact is intended.
        open_width = latest(world, "/joints").data["gripper_width"]
        self.pads_free = max(touch(world)) < TOUCH_CONTACT_N and open_width > 0.02
        self.over = 0  # consecutive checks above the force threshold

    def check(self) -> dict | None:
        force = external_force(self.world)
        # A collision adds force; releasing a load (lifting off after a press, or lifting a held
        # object off the table) removes it. Only growth of the external force counts.
        change = float(np.linalg.norm(force) - np.linalg.norm(self.baseline))
        self.over = self.over + 1 if change > CONTACT_FORCE_N else 0
        if self.over >= CONTACT_PERSIST:
            return {"sensed": "wrist force", "force_change_N": round(change, 1),
                    "external_force_N": np.round(force, 1)}
        if self.pads_free:
            left, right = touch(self.world)
            if max(left, right) > TOUCH_CONTACT_N:
                return {"sensed": "fingertip touch", "touch_N": [round(left, 2), round(right, 2)]}
        return None


def _follow(world: World, q_next: np.ndarray, watcher: "ContactWatcher | None",
            check: Check) -> dict | None:
    """One control period: ramp the servo target linearly to q_next over every physics step
    (target jumps make the stiff servo jerk, which the wrist sensor reads as ~20 N spikes),
    watching for contact every WATCH_EVERY steps. Returns what was sensed, if anything."""
    q_prev = world.q_cmd.copy()
    for i in range(1, CONTROL_STEPS + 1):
        world.command(q=q_prev + (i / CONTROL_STEPS) * (q_next - q_prev))
        world.step()
        if watcher and i % WATCH_EVERY == 0 and (sensed := watcher.check()):
            return sensed
    if check:
        check()
    return None


def _stop_here(world: World) -> None:
    """Hold the arm where it is (after a contact) instead of pushing on toward the target."""
    world.command(q=robot_state(world).q)
    world.step(25)


def solve_ik(world: World, pos: np.ndarray, quat: np.ndarray,
             restarts: bool = True) -> tuple[np.ndarray, bool, bool, float]:
    """(q, converged, from_current, closest position error in m). Tries the current posture
    first. With `restarts`, then the current posture with the last wrist joint turned half a
    turn (when the hand must turn far, the IK otherwise drives that joint into its limit), the
    rest posture and the ready-to-point-forward posture, keeping the converged solution
    nearest the current posture (the shortest joint-space path); only if none converges, a
    few seeded random postures."""
    belief = world.belief
    q_now = world.q_cmd
    best_q, best_err = q_now, np.inf

    def attempt(q0):
        nonlocal best_q, best_err
        q, ok = belief.ik(q0, pos, quat, iters=300)
        err = 0.0 if ok else float(np.linalg.norm(pose_error(*belief.fk(q), pos, quat)[:3]))
        if err < best_err:
            best_q, best_err = q, err
        return q, ok

    q, ok = attempt(q_now)
    if ok or not restarts:
        return q, ok, ok, best_err
    turned = [q_now.copy() for _ in range(2)]
    turned[0][6] += np.pi
    turned[1][6] -= np.pi
    seeds = [t for t in turned if belief.q_lo[6] <= t[6] <= belief.q_hi[6]] + [
        belief.q_rest, np.radians(READY_FORWARD_DEG)]  # the IK rarely finds pointing forward from down
    found = [q for q, ok in map(attempt, seeds) if ok]
    if found:
        return min(found, key=lambda q: float(np.abs(q - q_now).max())), True, False, 0.0
    rng = world.rng("ik")
    for _ in range(4):
        q, ok = attempt(rng.uniform(belief.q_lo, belief.q_hi))
        if ok:
            return q, True, False, 0.0
    return best_q, False, False, best_err


def wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def solve_grasp_ik(world: World, pos: np.ndarray, yaw: float, current_quat: np.ndarray | None = None,
                   pointing: str = "down"):
    """IK for a gripper pointing down (or forward) at `yaw` or at yaw + 180 deg. The parallel
    gripper is symmetric under a half turn, so both hold an object the same way; the one needing
    less rotation from `current_quat` is tried first (no pointless half-turn spins), and the
    other is the fallback when the first is beyond the wrist's joint limit.
    Returns (q, converged, from_current, closest_error, yaw)."""
    requested = wrap_angle(yaw)
    candidates = [requested, wrap_angle(yaw + np.pi)]
    if current_quat is not None:  # ties (within 10 deg) go to the requested yaw
        candidates.sort(key=lambda c: quat_angle(current_quat, yaw_quat(c, pointing))
                        - (np.radians(10) if c == requested else 0.0))
    # A solution reachable from the current posture allows a straight-line path; any other
    # needs a joint-space path, which can swing the hand low through the scene. So both yaws
    # are tried from the current posture before either is tried with restarts, and then the
    # solution nearest the current posture wins (the shortest swing).
    best = None
    for restarts in (False, True):
        solved = []
        for candidate in candidates:
            q, ok, straight, closest = solve_ik(world, pos, yaw_quat(candidate, pointing), restarts)
            if ok and straight:
                return q, True, True, 0.0, candidate
            if ok:
                solved.append((q, True, False, 0.0, candidate))
            elif best is None or closest < best[3]:
                best = (q, False, False, closest, candidate)
        if solved:
            return min(solved, key=lambda r: float(np.abs(r[0] - world.q_cmd).max()))
    return best


def move_tcp(world: World, pos, yaw: float, speed: float = 0.15, tolerance: float = 0.002,
             stop_on_contact: bool = True, check: Check = None, pointing: str = "down") -> Outcome:
    """Move the fingertip centre to `pos` (m), gripper pointing down (or forward, along +x),
    rotated `yaw` rad (or its half-turn twin when only that one is within the joint limits).
    Follows a straight line when the target is reachable from the current posture, otherwise a
    smooth joint-space path."""
    pos = np.asarray(pos, dtype=float)
    start = world.clock.sim_time
    q_goal, ok, straight, closest, used_yaw = solve_grasp_ik(world, pos, yaw,
                                                             robot_state(world).tcp_quat, pointing)
    if not ok:
        hint = ("the position is reachable, but not with this yaw or its half-turn twin; try "
                "another yaw" if closest < 0.005 else
                f"the closest reachable pose is {closest * 1000:.0f} mm away; try a point nearer "
                "the robot")
        return Outcome("failed", "unreachable", 0.0,
                       {"target": {"position": pos, "yaw": wrap_angle(yaw), "pointing": pointing},
                        "closest_error_m": closest}, hint=hint)
    flipped = abs(wrap_angle(used_yaw - yaw)) > np.pi / 2
    yaw, quat = used_yaw, yaw_quat(used_yaw, pointing)
    target = {"position": pos, "yaw": yaw, "pointing": pointing}

    state = robot_state(world)
    p0, quat0 = state.tcp_pos, state.tcp_quat
    duration = max(np.linalg.norm(pos - p0) / speed, quat_angle(quat0, quat) / ROT_SPEED,
                   MIN_DURATION) if straight else _joint_duration(world.q_cmd, q_goal)
    watcher = ContactWatcher(world) if stop_on_contact else None
    q, q_start, t, dt = world.q_cmd.copy(), world.q_cmd.copy(), 0.0, CONTROL_STEPS * world.timestep
    max_step = MAX_JOINT_SPEED * dt
    while t < duration:
        if straight:
            # Keep tracking the same IK branch to the end: the arm is redundant (7 joints), so
            # switching to the separately solved q_goal would jerk it through its null space.
            # Where a period's step along the line would move a joint faster than
            # MAX_JOINT_SPEED, take a shorter step: the hand slows down there instead.
            step = dt
            for _ in range(4):
                t_next = min(t + step, duration)
                s = min_jerk(t_next / duration)
                q_next, _ = world.belief.track(q, p0 + s * (pos - p0), slerp(quat0, quat, s),
                                               iters=20 if t_next < duration else 200)
                jump = float(np.abs(q_next - q).max())
                if jump <= 1.05 * max_step:
                    break
                step *= 0.9 * max_step / jump
            t, q = t_next, q_next
        else:
            t = min(t + dt, duration)
            q = q_start + min_jerk(t / duration) * (q_goal - q_start)
        if sensed := _follow(world, q, watcher, check):
            _stop_here(world)
            return _tcp_outcome(world, "partial", "contact", start, target, sensed,
                                "stopped early: something was in the way", flipped)

    while world.clock.sim_time - start < duration + SETTLE_TIMEOUT:
        state = robot_state(world)
        err = pose_error(state.tcp_pos, state.tcp_quat, pos, quat)
        moving = np.abs(latest(world, "/joints").data["dq"]).max() > 0.02
        if np.linalg.norm(err[:3]) < tolerance and np.linalg.norm(err[3:]) < ANGLE_TOLERANCE and not moving:
            return _tcp_outcome(world, "reached", "reached", start, target, flipped=flipped)
        if sensed := _follow(world, q, watcher, check):
            _stop_here(world)
            return _tcp_outcome(world, "partial", "contact", start, target, sensed,
                                "stopped early: something was in the way", flipped)
    return _tcp_outcome(world, "partial", "timeout", start, target,
                        hint="did not settle within tolerance; something may be pushing back",
                        flipped=flipped)


def _tcp_outcome(world, status, reason, start, target, sensed=None, hint=None,
                 flipped: bool = False) -> Outcome:
    state = robot_state(world)
    pointing = target["pointing"]
    err = pose_error(state.tcp_pos, state.tcp_quat, target["position"], yaw_quat(target["yaw"], pointing))
    details = {"target": target, "achieved": {"position": state.tcp_pos,
                                              "yaw": quat_yaw(state.tcp_quat, pointing)},
               "error_m": float(np.linalg.norm(err[:3])), "error_rad": float(np.linalg.norm(err[3:]))}
    if sensed:
        details["contact"] = sensed
    if flipped:
        note = "turned to the half-turn twin of the requested yaw (same grasp, less wrist rotation)"
        hint = f"{hint}; {note}" if hint else note
    return Outcome(status, reason, world.clock.sim_time - start, details, hint)


def _joint_duration(q0, q1) -> float:
    return max(float(np.abs(np.asarray(q1) - q0).max()) / JOINT_SPEED, MIN_DURATION)


def move_joints(world: World, q_target, stop_on_contact: bool = True, check: Check = None) -> Outcome:
    """Smooth joint-space motion to `q_target` (rad, 7 arm joints), clipped to the joint limits."""
    belief = world.belief
    requested = np.asarray(q_target, dtype=float)
    q_goal = np.clip(requested, belief.q_lo, belief.q_hi)
    clipped = np.abs(q_goal - requested).max() > np.radians(0.5)
    start, q_start = world.clock.sim_time, world.q_cmd.copy()
    duration, t, dt = _joint_duration(q_start, q_goal), 0.0, CONTROL_STEPS * world.timestep
    watcher = ContactWatcher(world) if stop_on_contact else None

    def outcome(status, reason, sensed=None, hint=None):
        achieved = robot_state(world).q
        details = {"target": requested, "achieved": achieved,
                   "max_error_rad": float(np.abs(achieved - requested).max())}
        if sensed:
            details["contact"] = sensed
        return Outcome(status, reason, world.clock.sim_time - start, details, hint)

    while world.clock.sim_time - start < duration + SETTLE_TIMEOUT:
        t = min(t + dt, duration)
        if sensed := _follow(world, q_start + min_jerk(t / duration) * (q_goal - q_start),
                             watcher, check):
            _stop_here(world)
            return outcome("partial", "contact", sensed, "stopped early: something was in the way")
        if t >= duration and np.abs(robot_state(world).q - q_goal).max() < np.radians(0.5):
            if clipped:
                return outcome("partial", "joint_limit", hint="some targets were beyond the "
                               "joint limits and were clipped")
            return outcome("reached", "reached")
    return outcome("partial", "timeout", hint="did not settle on the target")


def move_gripper(world: World, width: float, check: Check = None) -> Outcome:
    """Open (or close) the fingers to `width` m and wait until they stop. Closing on an object
    leaves them resting on it; `grasped` reports whether something is held between them."""
    width = float(np.clip(width, 0.0, GRIPPER_MAX_WIDTH))
    closing = width < latest(world, "/joints").data["gripper_width"]
    world.command(gripper=gripper_ctrl(width))
    start = world.clock.sim_time
    history: list[float] = []
    while world.clock.sim_time - start < 1.5:
        world.step(CONTROL_STEPS)
        if check:
            check()
        history.append(latest(world, "/joints").data["gripper_width"])
        if len(history) > 5 and world.clock.sim_time - start > 0.1 and np.ptp(history[-6:]) < 5e-4:
            break
    measured = history[-1]
    left, right = touch(world)
    grasped = closing and measured > GRASP_MIN_WIDTH and min(left, right) > TOUCH_CONTACT_N
    details = {"target_width_m": width, "width_m": measured, "grasped": grasped,
               "touch_N": [round(left, 2), round(right, 2)]}
    blocked = abs(measured - width) > 0.003
    if closing and not grasped and width <= GRASP_MIN_WIDTH:
        return Outcome("reached", "reached", world.clock.sim_time - start, details,
                       "the fingers closed on nothing")
    if blocked and not grasped:
        return Outcome("partial", "contact", world.clock.sim_time - start, details,
                       "the fingers were blocked before reaching the requested width")
    return Outcome("reached", "reached", world.clock.sim_time - start, details)


def wait(world: World, seconds: float, check: Check = None) -> Outcome:
    start = world.clock.sim_time
    while world.clock.sim_time - start < seconds - 1e-9:
        world.step(min(CONTROL_STEPS, max(1, round((seconds - (world.clock.sim_time - start))
                                                   / world.timestep))))
        if check:
            check()
    return Outcome("reached", "reached", world.clock.sim_time - start)

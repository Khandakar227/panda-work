# Adding a robot

pandawork ships one arm, the Franka Panda from MuJoCo Menagerie. The robot is not an abstract
interface yet: it is a block of constants and three builder functions at the top of
`src/pandawork/core/robot.py`, and the rest of the harness imports those names. This page says
what is Panda-specific, how to swap it, what the rest of the code assumes, and how to prove the
swap worked.

## What is Panda-specific today

Everything between the `Robot description` and `end of robot description` markers in
`core/robot.py`. The comment on each line names what depends on it. In short:

| constant | meaning | who uses it |
|---|---|---|
| `PANDA_DIR`, `PANDA_SCENE` | the MJCF and meshes | `panda_spec()` |
| `ARM_JOINTS` | the seven arm joint names | control, encoders, faults, IK |
| `ARM_JOINT_LIMITS` | joint ranges in radians | `check_reachable`, the IK's soft limits |
| `FINGER_JOINTS` | the two finger joints | gripper width, finger state |
| `TCP_SITE`, `TCP_OFFSET` | the fingertip-centre site and where it sits on the hand | IK, every motion report, faults |
| `GRIPPER_OPEN`, `GRIPPER_CLOSED`, `GRIPPER_MAX_WIDTH`, `GRIPPER_STIFFNESS_SCALE` | the finger actuator's command range, the opening it produces, a servo gain multiplier | gripper tool, skills, `panda_spec()` |
| `DOWN_QUAT`, `FORWARD_QUAT`, `POINTING`, `READY_FORWARD_DEG` | the two hand orientations the tools offer and the joint posture for the forward one | motion skills, the move tools |
| `FT_SITE`, `WRIST_FORCE`, `WRIST_TORQUE`, `TOUCH_SENSORS` | wrist force/torque and finger touch sensor names | sensors, `read_wrist_ft`, the grasp check |
| `CAMERAS` | four cameras by name, with parent body, pose and field of view | rendering, the `look` tool, the system prompt |
| `TABLE_*` | the tabletop the scene builds | scenes, tasks |

Three functions build the model from those values: `_add_sensors` (encoders, joint torques,
wrist F/T, touch pads), `_add_cameras`, and `panda_spec`, which loads the MJCF, adds the table,
the TCP site, the sensors and cameras, and raises the finger servo gain. Both the plant and the
self-model are compiled from the same spec (`build_models`), and `reset_home` puts the arm in
the MJCF's `home` keyframe.

## Steps

1. **Drop in the model.** Put the MJCF and meshes under `assets/<robot>/` and point
   `PANDA_SCENE` at its scene file. Keep the licence file of the source next to it.
2. **Name the joints.** Set `ARM_JOINTS`, `ARM_JOINT_LIMITS` (copy the ranges from the MJCF;
   `tests/test_robot_tools.py` checks they match the compiled model) and `FINGER_JOINTS`.
3. **Give it a tool centre point.** Either add a site named `tcp` at the fingertip centre in
   the MJCF, or keep `panda_spec`'s code that adds one and set `TCP_OFFSET` to the distance from
   the hand body origin. The body the site hangs from is `hand` in `panda_spec`; rename it there.
4. **Sensors.** `_add_sensors` expects a joint position and velocity sensor per joint, a joint
   actuator force per arm joint, a wrist force/torque site on the hand, and a touch site on each
   finger body (`left_finger`, `right_finger`). Rename the bodies there, or stub what the arm does
   not have: the realistic tier reads whatever `world/sensors.py` finds by these names, and a
   missing sensor is a change to that module, not to the tools.
5. **Gripper.** The Panda has one position actuator driving both fingers (`actuator8`).
   Set the command range and `GRIPPER_MAX_WIDTH`, and remove or retune the
   `GRIPPER_STIFFNESS_SCALE` lines in `panda_spec` for a different servo.
6. **Postures.** `reset_home` uses the MJCF `home` keyframe; add one. Set `READY_FORWARD_DEG`
   to a joint posture that points the hand forward, or drop the `forward` entry from
   `POINTING` if the arm cannot.
7. **Cameras.** Re-declare the four entries of `CAMERAS`; `wrist` hangs from a body of the
   arm, the other three from `world`. The system prompt lists them by name, so keep the names
   unless you also edit `agent/prompt.py`.

## What the rest of the code assumes

- **A 7-DoF arm.** `core/ik.py` is damped-least-squares IK over `ARM_JOINTS` with soft joint
  limits; it works for six joints too, but the forward posture and the keypad task were tuned
  for the Panda's reach. `control.py` indexes `data.ctrl[:7]` for the arm and `ctrl[7]` for the
  gripper; change those slices with the actuator layout.
- **A parallel two-finger gripper.** The `gripper` tool reports a width in metres and decides
  "grasped" from width plus the two touch pads. A suction cup or a three-finger hand needs a
  different tool, not a different constant.
- **Gravity compensation from the self-model.** `apply_control` adds the self-model's gravity
  torques to the plant. Any arm the MJCF models with correct masses gets this for free.
- **Faults are written against these names.** `core/faults.py` offsets an encoder on
  `ARM_JOINTS[i]`, tilts a camera in `CAMERAS`, limits a joint range, and caps the gripper
  width. They keep working as long as the names resolve.

## Acceptance check

Run the `reach` oracle over a few seeds and the two test files that exercise motion end to end:

```bash
python scripts/oracle_eval.py --tasks reach --seeds 5   # expect 5/5
python -m pytest tests/test_robot_tools.py tests/test_reach.py tests/test_motion.py -q
```

A wrong TCP site shows up here as a reach error of centimetres, not millimetres. Then run the
whole suite; the task tests (`tests/test_tasks.py`) will tell you which scene layouts no longer
fit the new arm's reach.

## Future work

A `RobotDescription` dataclass carrying these constants, passed to `World`, the skills and the
tools, so that two arms can coexist in one process. The constants block is already the shape of
that object; the work is threading it through the imports.

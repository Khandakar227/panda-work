# Faults

A fault is a way the robot's body can differ from what the robot believes about itself. It is
the harness's instrument for one kind of experiment: what does an agent do when the tools it
relies on describe a robot that is not quite the one moving?

## How a fault works

The harness keeps two MuJoCo models (`core/robot.py:build_models`): the **plant**, which is
simulated, and the **self-model**, which the controller and every tool compute with. A fault
patches the plant only. Encoders read the plant through the fault's `read_encoders` hook, the
self-model never changes, so "fingertip at (x, y, z)" and "reached, error 0.6 mm" keep describing
the belief while the hand is somewhere else.

A fault (`core/faults.py`) has up to three hooks:

| hook | when | used by |
|---|---|---|
| `patch_spec(spec)` | before the plant is compiled | longer fingers, a tilted camera, a reduced joint range, a jammed gripper |
| `read_encoders(q_true) -> q_reported` | every encoder read | encoder offsets |
| `describe()` and the `hidden` flag | the fault table, the viewer panel, the episode's `meta.json` | everyone |

`visibility` is `hidden` when the tool reports conceal the fault and `revealed` when they show it.

## The presets

| preset | what changes | what the tools say | |
|---|---|---|---|
| `encoder_j1_2deg`, `encoder_j1_5deg`, `encoder_j2_2deg`, `encoder_j4_3deg` | an encoder reads high; the arm stops short of every target | "reached, error < 1 mm"; the hand is 1 to 5 cm away | hidden |
| `fingertip_1cm_long`, `fingertip_2cm_long`, `fingertip_15mm_side` | the fingers are longer or offset sideways | "reached"; grasps close on air or hit the table early | hidden |
| `wrist_cam_5deg`, `wrist_cam_10deg`, `top_cam_2deg` | a camera is pitched | images look normal; positions estimated from that camera are off by centimetres (top: about 5 cm, and `top` is the default detection camera) | hidden |
| `gripper_5cm` | the fingers open only to 5 cm | the gripper tool reports the small opening | revealed |
| `stiff_j4` | the elbow lost 50° at the low end of its range | moves that need it come back `partial` | revealed |

`python scripts/faults.py` prints this table with full descriptions;
`python scripts/faults.py --watch encoder_j1_5deg --task pick_place` shows the oracle misled by
one, with a green sphere where the robot believes its fingertip is and a red one where it is.

## Guarantees

- The realistic and vision-only tiers never see plant truth. `tests/test_faults.py` runs every
  tool under every fault and asserts that nothing in the output depends on the plant.
- `simulate` runs on a twin of the *believed* body, so planning cannot detect the fault either.
- `detect_objects` back-projects pixels through the believed camera pose; a tilted camera
  therefore misplaces detections exactly as it would on a real robot.
- The privileged tier is told the truth on purpose: it is the "if the agent knew" control.
- The fault is written to `meta.json` and re-applied on replay, so a recorded episode
  reproduces exactly.

## Running experiments with faults

```bash
python scripts/oracle_eval.py --tasks pick_place --seeds 5 --fault encoder_j1_5deg   # the never-verifying baseline
python scripts/run_agent.py pick_place --tier realistic --seed 0 --fault encoder_j1_5deg --max-calls 60
python scripts/batch.py run faults --faults none encoder_j1_5deg top_cam_2deg gripper_5cm \
       --tasks pick_place press_button --tiers privileged realistic vision_only --seeds 0-4 --max-calls 60
```

Faulty episodes need a larger call budget than clean ones (about three times the calls).
`--exclude TASK:FAULT` on `batch.py` skips cells a fault makes physically impossible, for
instance a stiff elbow and a drawer handle it cannot reach; otherwise those cells measure reach,
not behaviour.

**Prompt variants.** `--warn mild` adds one sentence to the system prompt saying the robot is
not ideal and its calibration can drift; `--warn full` adds an instruction to verify with the
cameras before trusting a motion report or claiming success (`agent/prompt.py:WARNINGS`). The gap
between the two separates knowing from being told what to do. Batches take `--warn both` or
`--warn all` to run the variants as extra arms.

## Adding a preset

1. Subclass `Fault` in `core/faults.py` or instantiate an existing class with new parameters
   (`EncoderOffset("encoder_j1_3deg", joint=0, offset_deg=3.0)`,
   `CameraTilt("wrist_cam_7deg", camera="wrist", tilt_deg=7.0)`).
2. Give it a `name`, set the class's `hidden` flag (`visibility` is derived from it), and write a
   `describe()` an operator would understand.
3. Add it to the `FAULTS` table. It is now a `--fault` choice everywhere.
4. Run `python -m pytest tests/test_faults.py -q`: the leak audit covers every entry in the table.

A new *kind* of fault (say, a slipping gripper) needs a `patch_spec` that edits the plant's
`MjSpec`, and perhaps a new hook; keep the self-model untouched and the audit will tell you
whether any tool gives it away.

# GO-M8010-6 Policy Deployment Bundle

## Files

- `robot_deploy.py` — main deployment script. Startup calibration, per-joint
  invert, 5°-step safety abort + motor release, policy identity verification,
  full logging. Run this on the robot.
- `export_policy_for_deployment.py` — run on your training machine to turn
  an rsl_rl checkpoint into `policy.pt` + `policy.meta.json` (the sidecar
  robot_deploy.py checks against before it will run anything).
- `robot_config.json` — example config: control rate, max step, action
  scale, per-joint motor id / invert / kp / kd.
- `mock_unitree_actuator_sdk.py` — software motor simulator (double-
  integrator physics) so you can run robot_deploy.py with zero hardware
  attached. Verified working end-to-end in testing.
- `isaac_sim_unitree_backend.py` — alternate backend that drives your real
  QMINI articulation live in Isaac Sim for dynamics-accurate testing.
  Written from your Isaac Lab code but NOT executed/verified (no GPU/Isaac
  Sim available in the environment I built this in) — expect to debug a
  few API details on your machine. See docstring at the top for known
  approximations (kp/kd not applied per-command, joint-name mapping to
  verify, stepping timing).

## Quick start (no hardware, sanity-test the pipeline)

    mkdir -p mock_sdk
    cp mock_unitree_actuator_sdk.py mock_sdk/unitree_actuator_sdk.py

    python3 export_policy_for_deployment.py \
        --checkpoint /path/to/your/rsl_rl_checkpoint.pt \
        --output-dir ./deploy_bundle \
        --joint-order hip knee ankle \
        --action-scale 0.15

    PYTHONPATH=mock_sdk python3 robot_deploy.py \
        --config robot_config.json \
        --policy ./deploy_bundle/policy.pt \
        --policy-meta ./deploy_bundle/policy.meta.json \
        --port MOCK

## On the real robot

Edit `UNITREE_SDK_LIB_PATH` at the top of `robot_deploy.py` to point at your
built `unitree_actuator_sdk/lib`, edit `robot_config.json` for your real
motor IDs/gains, then run the same command with `--port /dev/ttyUSB0` (or
whatever your actual serial device is) instead of the mock PYTHONPATH.

## Before you trust any of this on real motors

- Compare `MotionReference` in `robot_deploy.py` against your actual
  `MotionPlayer` — it's a linear-interpolation placeholder and only
  matches training if your real one also interpolates linearly.
- Verify the gear ratio (`queryGearRatio()` / 6.33 fallback) against your
  SDK version and motor datasheet.
- Verify `unitree_actuator_sdk` attribute names (`data.temp`, `data.merror`,
  etc.) against your installed SDK version — all of that is isolated in
  `MotorBus` in `robot_deploy.py`.
- Do the startup confirmation step for real: check the printed raw/calib
  degrees and invert flags actually match the robot's physical pose before
  typing `y`.

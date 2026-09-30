# PhantomOS Quantum

PhantomOS Quantum is a **safety-first spatial interaction engine** for hands-free desktop control. It turns camera/landmark observations into normalized spatial intents and dispatches them through native operating-system APIs when explicitly enabled.

> **Important:** This repository is an extensible production foundation, not a claim of physically impossible latency. End-to-end latency depends on the camera, model, OS compositor, permissions, and hardware. The predictor reduces perceived latency; it cannot remove sensor or OS latency. Native dispatch is disabled by default (`dry_run=true`).

## Design goals

- deterministic, timestamped pipeline with bounded queues and back-pressure;
- predictive hand trajectories (constant-acceleration Kalman + second-order projection);
- adaptive One Euro filtering for low stationary jitter without motion drag;
- native dispatch adapters for Windows `SendInput`, macOS CoreGraphics, and Linux `/dev/uinput`;
- multimodal gaze/hand/voice fusion with explicit confidence and safety gates;
- a platform-neutral gesture state machine, 3D depth model, kinematic physics, biometric unlock challenge, and telemetry;
- optional OpenCV, MediaPipe/ONNX Runtime, PyQt6, and audio integrations that do not prevent headless testing.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -e '.[dev]'
python -m phantomos_quantum --dry-run --synthetic
pytest
```

Install optional integrations as needed:

```bash
pip install -e '.[vision,ui,voice]'
```

The default runner never injects OS input. To use real input, configure `dry_run = false`, review the platform permissions, and start with a narrow allow-list. The optional C++ ABI can be built with `cmake -S native -B build/native && cmake --build build/native --config Release`. Linux uinput normally requires membership in the `input` group; macOS requires Accessibility/Input Monitoring permissions; Windows may require the process to run at the appropriate integrity level.

## Architecture

```
FrameSource -> capture queue -> inference -> spatial prediction/filtering
           -> gaze/voice/gesture fusion -> safety policy -> native dispatcher
           -> telemetry + optional translucent HUD
```

`Pipeline` owns five asynchronous stages (capture, inference, physics, fusion, dispatch). Each stage has a bounded queue and records phase timing. Hardware-specific integrations are adapters: a missing optional dependency produces a clear error or a safe no-op instead of breaking import-time behavior.

## Package map

- `models.py`: immutable sensor, landmark, intent, and telemetry models.
- `filters.py`: Kalman, Taylor, and adaptive One Euro prediction/filtering.
- `camera.py`: OpenCV capture and deterministic synthetic source.
- `inference.py`: MediaPipe hand adapter plus normalized observation interfaces.
- `spatial.py`: depth normalization and spring/momentum physics.
- `gestures.py`: debounced gesture recognition and drag/scroll state.
- `fusion.py`: gaze + hand + local voice command resolution.
- `native.py`: safe dry-run dispatcher and native OS backends.
- `native/`: optional C++17 shared-library C ABI for hardened deployments.
- `security.py`: hand geometry profile and air-passcode challenge.
- `interactions.py`: window sweeps, shortcuts, copy/paste, push, and radial dial primitives.
- `keyboard.py`: normalized dwell/pinch air-keyboard layout.
- `workspace.py` / `display.py`: auditable window intents and multi-monitor coordinates.
- `audio.py`: optional local mechanical key feedback.
- `hud.py`: optional PyQt6 always-on-top telemetry overlay.
- `orchestrator.py`: application wiring, safety policy, and lifecycle.

## Safety model

The engine treats sensor input as untrusted. Confidence thresholds, dwell/debounce windows, an unlock challenge, and a global kill switch are applied before dispatch. `NativeDispatcher` is dry-run by default and emits a structured audit record for every would-be action. Do not enable unrestricted automation until the gesture mappings and permissions have been tested in a disposable user session.

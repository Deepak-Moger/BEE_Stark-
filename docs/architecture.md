# PhantomOS Quantum architecture

## Runtime boundaries

1. **Capture** owns a single camera handle and configures a one-frame buffer. It emits timestamped `SensorFrame` objects.
2. **Inference** runs hand landmarks and optional face landmarks. `MediaPipeHandsBackend` is the reference adapter; `OnnxInferenceBackend` negotiates TensorRT, DirectML, CUDA, CoreML, WebGPU, and CPU providers.
3. **Physics** predicts each landmark using a constant-acceleration Kalman state and a second-order Taylor projection. The adaptive One Euro stage changes cutoff with velocity. A 15ms horizon is a default, not a latency guarantee.
4. **Fusion** converts gaze target, precise hand point, and final local speech transcript into `Intent` objects. Confidence and timestamps survive every boundary.
5. **Dispatch** applies allow-lists, confidence, unlock, and kill-switch policy before calling direct OS APIs.

Every queue is bounded and newest data wins. Stale camera frames are discarded rather than increasing latency. Each worker catches failures, records them in `Pipeline.errors`, and stops the graph; callers should surface that state to an operator.

## Latency budget

The engine measures camera, inference, prediction/physics, filter, and dispatch phases. It cannot promise a sub-10ms end-to-end path: a webcam exposure, USB transfer, model execution, OS scheduling, compositor, and application processing are outside the Python process. Use `TelemetrySnapshot` to measure on the target device and tune model resolution/provider/queue depth from evidence.

## Native event ABI

`native.py` contains direct `ctypes` platform adapters. `native/phantom_input.cpp` is an optional C++17 shared-library ABI for hardened deployments. Neither path uses PyAutoGUI or pynput. Real input is disabled by default, and the compiled library is not auto-selected.

## Extension points

- Implement an ONNX decoder for a hand/gaze model and pass it to `OnnxInferenceBackend`.
- Implement a voice decoder for Whisper-tiny ONNX and pass it to `WhisperOnnxRecognizer`.
- Add a compositor adapter for `WorkspaceManager` actions on each host OS.
- Register a platform action handler with `NativeDispatcher` for window/system operations after validating permissions and audit requirements.
- Feed `TelemetrySnapshot` and landmark batches to `PhantomHud` from the host UI event loop.

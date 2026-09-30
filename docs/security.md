# Security and operational controls

PhantomOS Quantum is an accessibility/automation engine, not an authentication replacement. Camera biometrics are a convenience signal and must not be treated as equivalent to a hardware-backed OS login or MFA.

## Defaults

- native input: **dry-run**;
- unlock: required by `EngineConfig`;
- system/window actions: disabled;
- action allow-list: pointer, mouse buttons, scroll, key, and text only;
- queues: bounded, with stale frames dropped;
- `NativeDispatcher.enabled`: an in-process kill switch.

## Production checklist

1. Run in a disposable user profile and keep a physical keyboard/mouse connected.
2. Enroll a geometry profile locally; never transmit raw landmarks. Rotate the profile if the camera or environment changes.
3. Keep `require_unlock=true` and add only the actions needed for the operator's workflow.
4. Validate Accessibility/Input Monitoring permissions on macOS and uinput group/device ACLs on Linux. Do not run the process as root solely to obtain uinput access.
5. Enable OS-level audit logging and review `DryRunBackend.events` before enabling native input.
6. Provide an external hotkey or hardware stop control; do not rely only on a gesture if tracking is degraded.
7. Treat voice transcripts and selected-window context as sensitive data. Keep speech recognition local and avoid logging transcript contents by default.

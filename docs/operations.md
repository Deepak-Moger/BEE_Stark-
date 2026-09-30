# Operations runbook

## Headless smoke test

```bash
pip install -e '.[dev]'
python -m phantomos_quantum --synthetic --seconds 5
pytest
```

## Camera run

```bash
pip install -e '.[vision]'
PHANTOMOS_DRY_RUN=true python -m phantomos_quantum
```

Start in dry-run mode. Inspect the telemetry bar and audit events. A failure in an optional vision backend should be visible in the pipeline error list rather than silently enabling input.

## Live input

Build the optional C++ library or select the Python direct adapter for the host platform, then set an explicit allow-list. Live input should be a separate deployment profile, not a code change:

```text
PHANTOMOS_DRY_RUN=false
PHANTOMOS_NATIVE_BACKEND=auto
PHANTOMOS_REQUIRE_UNLOCK=true
PHANTOMOS_ALLOW_SYSTEM_ACTIONS=false
PHANTOMOS_ALLOWED_ACTIONS=pointer_move,left_click,right_click,scroll,key,text
```

Use a physical emergency stop. `PhantomEngine.lock()` closes the logical workspace gate; `NativeDispatcher.enabled = False` is the immediate software kill switch.

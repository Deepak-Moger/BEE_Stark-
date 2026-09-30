from __future__ import annotations

from phantomos_quantum.models import Intent, IntentKind, SpatialPoint
from phantomos_quantum.native import DryRunBackend, NativeDispatcher


def test_dry_run_dispatch_records_normalized_pointer() -> None:
    backend = DryRunBackend()
    dispatcher = NativeDispatcher(backend, 1000, 500, frozenset({"pointer_move", "key"}))
    result = dispatcher.dispatch(Intent(IntentKind.POINTER_MOVE, "pointer_move",
                                        point=SpatialPoint(0.5, 0.25), confidence=1.0))
    assert result.accepted and result.dry_run
    assert backend.events[-1].payload == {"x": 500, "y": 125}


def test_key_shortcut_is_explicitly_auditable() -> None:
    backend = DryRunBackend()
    dispatcher = NativeDispatcher(backend, 1000, 500, frozenset({"key"}))
    result = dispatcher.dispatch(Intent(IntentKind.KEY, "c", confidence=1.0,
                                        metadata={"shortcut": ("ctrl",), "pressed": True}))
    assert result.accepted
    assert [(e.action, e.payload) for e in backend.events] == [
        ("key", {"key": "ctrl", "pressed": True}),
        ("key", {"key": "c", "pressed": True}),
        ("key", {"key": "c", "pressed": False}),
        ("key", {"key": "ctrl", "pressed": False}),
    ]

from __future__ import annotations

from phantomos_quantum.models import Intent, IntentKind, SpatialPoint
from phantomos_quantum.native import DryRunBackend, NativeDispatcher


def test_required_unlock_blocks_even_dry_run_dispatch() -> None:
    dispatcher = NativeDispatcher(DryRunBackend(), 100, 100, frozenset({"pointer_move"}), require_unlock=True)
    result = dispatcher.dispatch(Intent(IntentKind.POINTER_MOVE, "pointer_move",
                                        point=SpatialPoint(0.5, 0.5), confidence=1.0))
    assert not result.accepted
    assert result.reason == "workspace locked"

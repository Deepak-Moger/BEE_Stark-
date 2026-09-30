from __future__ import annotations

from phantomos_quantum.fusion import MultimodalFusion
from phantomos_quantum.gestures import GestureEvent
from phantomos_quantum.models import GestureKind, SpatialPoint


def test_click_release_becomes_button_up() -> None:
    fusion = MultimodalFusion()
    event = GestureEvent(GestureKind.LEFT_CLICK, SpatialPoint(0.3, 0.4), 1.0,
                         metadata={"phase": "up"})
    intent = fusion.hand_intent(event)
    assert intent is not None
    assert intent.metadata["button"] == "left"
    assert intent.metadata["pressed"] is False


def test_advanced_system_intents_are_not_pointer_events() -> None:
    fusion = MultimodalFusion()
    event = GestureEvent(GestureKind.WINDOW_SNAP_RIGHT, None, 0.9,
                         metadata={"direction": "right"})
    intent = fusion.hand_intent(event)
    assert intent is not None
    assert intent.action == "window_snap_right"

from __future__ import annotations

from phantomos_quantum.keyboard import AirKeyboard


def test_keyboard_dwell_is_one_shot() -> None:
    keyboard = AirKeyboard(dwell_ms=100)
    assert keyboard.hover(0.06, 0.63, now=1.0) is None
    assert keyboard.hover(0.06, 0.63, now=1.11) == "Q"
    assert keyboard.hover(0.06, 0.63, now=1.12) is None
    assert AirKeyboard.text_for("SPACE") == " "

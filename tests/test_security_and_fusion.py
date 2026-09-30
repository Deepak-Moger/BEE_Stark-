from __future__ import annotations

import time

from phantomos_quantum.fusion import VoiceIntentResolver
from phantomos_quantum.models import VoiceCommand
from phantomos_quantum.security import AirPasscode


def test_air_passcode_requires_ordered_sequence() -> None:
    passcode = AirPasscode()
    passcode.begin()
    assert not passcode.observe("pinch", 1.0)
    assert not passcode.observe("circle", 1.0)
    assert not passcode.observe("pinch", 1.0)
    assert passcode.observe("push", 1.0)


def test_voice_resolver_requires_final_confident_transcript() -> None:
    resolver = VoiceIntentResolver()
    assert resolver.resolve(VoiceCommand("close this", 0.4, is_final=True), None) is None
    intent = resolver.resolve(VoiceCommand("close this", 0.95, is_final=True), None)
    assert intent is not None
    assert intent.action == "close_window"

"""Spatial biometric profile and challenge-response unlock state."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import math
import secrets
from typing import Iterable

from .models import HandObservation, SpatialPoint


@dataclass(frozen=True, slots=True)
class HandProfile:
    ratios: tuple[float, ...]
    palm_scale: float
    digest: str


class HandBiometric:
    """Create a privacy-preserving geometry template; raw frames are not retained."""

    _pairs = ((4, 8), (8, 12), (12, 16), (16, 20), (0, 9), (5, 17))

    def profile(self, hand: HandObservation) -> HandProfile:
        if len(hand.landmarks) != 21:
            raise ValueError("a 21-landmark hand is required")
        wrist, middle = hand.landmarks[0], hand.landmarks[9]
        palm_scale = max(math.hypot(wrist.x - middle.x, wrist.y - middle.y), 1e-6)
        ratios = tuple(round(math.sqrt((hand.landmarks[a].x - hand.landmarks[b].x) ** 2 +
                                     (hand.landmarks[a].y - hand.landmarks[b].y) ** 2) / palm_scale, 5)
                       for a, b in self._pairs)
        digest = hashlib.sha256(",".join(map(str, ratios)).encode()).hexdigest()
        return HandProfile(ratios, palm_scale, digest)

    def similarity(self, expected: HandProfile, observed: HandProfile) -> float:
        if len(expected.ratios) != len(observed.ratios):
            return 0.0
        error = sum(abs(a - b) for a, b in zip(expected.ratios, observed.ratios)) / len(expected.ratios)
        return max(0.0, min(1.0, math.exp(-3.0 * error)))


@dataclass(slots=True)
class AirPasscode:
    """Challenge sequence based on coarse 3-D gesture waypoints."""

    waypoints: tuple[str, ...] = ("circle", "pinch", "push")
    tolerance: float = 0.28
    nonce: bytes = b""
    index: int = 0
    unlocked: bool = False

    def __post_init__(self) -> None:
        if not self.waypoints:
            raise ValueError("an air passcode needs at least one waypoint")
        if not 0.0 < self.tolerance <= 1.0:
            raise ValueError("tolerance must be in (0, 1]")

    def begin(self) -> None:
        self.nonce = secrets.token_bytes(16)
        self.index = 0
        self.unlocked = False

    def observe(self, gesture: str, biometric_score: float, minimum_score: float = 0.78) -> bool:
        if not self.nonce:
            self.begin()
        if biometric_score < minimum_score or self.unlocked:
            return self.unlocked
        if gesture == self.waypoints[self.index]:
            self.index += 1
            if self.index == len(self.waypoints):
                self.unlocked = True
        else:
            # A wrong gesture resets the sequence rather than allowing partial replay.
            self.index = 0
        return self.unlocked

    def lock(self) -> None:
        self.unlocked = False
        self.index = 0
        self.nonce = b""

    def challenge_fingerprint(self) -> str:
        return hmac.new(self.nonce or b"locked", b"phantomos-air-passcode", hashlib.sha256).hexdigest()[:16]

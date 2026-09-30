"""3-D depth normalization and low-cost kinematic physics."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .models import HandObservation, SpatialPoint


def distance(a: SpatialPoint, b: SpatialPoint) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


@dataclass(frozen=True, slots=True)
class DepthSample:
    normalized_depth: float
    layer: int
    hand_scale: float
    confidence: float


class DepthEngine:
    """Map relative MediaPipe z and apparent hand size into stable depth layers."""

    def __init__(self, layer_count: int = 5, near: float = -0.35, far: float = 0.25) -> None:
        self.layer_count, self.near, self.far = layer_count, near, far

    def sample(self, hand: HandObservation) -> DepthSample:
        if len(hand.landmarks) != 21:
            return DepthSample(0.0, 0, 0.0, 0.0)
        wrist, middle = hand.landmarks[0], hand.landmarks[9]
        hand_scale = math.hypot(wrist.x - middle.x, wrist.y - middle.y)
        # z is relative and MediaPipe's hand scale makes it less camera-distance-sensitive.
        raw = middle.z / max(hand_scale, 1e-4)
        normalized = max(0.0, min(1.0, (raw - self.near) / (self.far - self.near)))
        layer = min(self.layer_count - 1, int(normalized * self.layer_count))
        return DepthSample(normalized, layer, hand_scale, hand.confidence)

    def pointer(self, hand: HandObservation) -> SpatialPoint | None:
        tip = hand.index_tip
        if tip is None:
            return None
        depth = self.sample(hand)
        return SpatialPoint(tip.x, tip.y, depth.normalized_depth, hand.confidence)


@dataclass(slots=True)
class SpringDamper:
    """Semi-implicit spring + friction integrator for spatial UI interactions."""

    stiffness: float = 180.0
    damping: float = 24.0
    mass: float = 1.0
    friction: float = 0.85
    value: float = 0.0
    velocity: float = 0.0

    def step(self, target: float, dt: float) -> float:
        dt = max(1e-5, min(dt, 0.05))
        force = self.stiffness * (target - self.value) - self.damping * self.velocity
        self.velocity += force / max(self.mass, 1e-6) * dt
        self.velocity *= math.exp(-self.friction * dt)
        self.value += self.velocity * dt
        return self.value


@dataclass(slots=True)
class InertialScroller:
    friction: float = 8.0
    max_speed: float = 5.0
    position: float = 0.0
    velocity: float = 0.0

    def impulse(self, velocity: float) -> None:
        self.velocity = max(-self.max_speed, min(self.max_speed, self.velocity + velocity))

    def step(self, dt: float) -> float:
        dt = max(0.0, min(dt, 0.1))
        self.position += self.velocity * dt
        self.velocity *= math.exp(-self.friction * dt)
        if abs(self.velocity) < 1e-4:
            self.velocity = 0.0
        return self.position

"""Numerical predictors. No model, GPU, or NumPy dependency is needed here."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

from .models import SpatialPoint


def _finite(*values: float) -> None:
    if not all(math.isfinite(v) for v in values):
        raise ValueError("filter inputs must be finite")


@dataclass(slots=True)
class Kalman1D:
    """Position/velocity/acceleration Kalman model with white-jerk process noise.

    F = [[1, dt, dt²/2], [0, 1, dt], [0, 0, 1]], H = [1, 0, 0].
    Projection is a second-order Taylor expansion, not a neural prediction.
    """
    process_noise: float = 3.0
    measurement_noise: float = 0.0001
    position: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0
    initialized: bool = False
    covariance: list[list[float]] = field(default_factory=lambda: [
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
    ])

    def __post_init__(self) -> None:
        _finite(self.process_noise, self.measurement_noise)
        if self.process_noise < 0 or self.measurement_noise <= 0:
            raise ValueError("invalid Kalman noise parameters")

    def update(self, measurement: float, dt: float) -> float:
        _finite(measurement, dt)
        if dt <= 0:
            raise ValueError("dt must be positive")
        if not self.initialized:
            self.position = measurement
            self.initialized = True
            return measurement
        dt = min(dt, 0.25)
        f = [[1.0, dt, dt * dt / 2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]]
        self.position += self.velocity * dt + self.acceleration * dt * dt / 2
        self.velocity += self.acceleration * dt
        fp = [[sum(f[i][k] * self.covariance[k][j] for k in range(3))
               for j in range(3)] for i in range(3)]
        q = [[dt**5 / 20, dt**4 / 8, dt**3 / 6],
             [dt**4 / 8, dt**3 / 3, dt**2 / 2],
             [dt**3 / 6, dt**2 / 2, dt]]
        predicted = [[sum(fp[i][k] * f[j][k] for k in range(3)) + self.process_noise * q[i][j]
                      for j in range(3)] for i in range(3)]
        innovation = measurement - self.position
        s = predicted[0][0] + self.measurement_noise
        gain = [predicted[i][0] / s for i in range(3)]
        self.position += gain[0] * innovation
        self.velocity += gain[1] * innovation
        self.acceleration += gain[2] * innovation
        # Use the *predicted* covariance for every term, not a partially updated P.
        self.covariance = [[predicted[i][j] - gain[i] * predicted[0][j]
                            for j in range(3)] for i in range(3)]
        for i in range(3):
            self.covariance[i][i] = max(self.covariance[i][i], 1e-15)
        return self.position

    def project(self, horizon_s: float) -> float:
        _finite(horizon_s)
        if horizon_s < 0:
            raise ValueError("prediction horizon cannot be negative")
        return self.position + self.velocity * horizon_s + self.acceleration * horizon_s**2 / 2


@dataclass(slots=True)
class OneEuroFilter:
    min_cutoff: float = 1.5
    beta: float = 4.0
    derivative_cutoff: float = 1.0
    value: float = 0.0
    derivative: float = 0.0
    initialized: bool = False
    _raw: float = 0.0

    def __post_init__(self) -> None:
        _finite(self.min_cutoff, self.beta, self.derivative_cutoff)
        if self.min_cutoff <= 0 or self.beta < 0 or self.derivative_cutoff <= 0:
            raise ValueError("invalid One Euro parameters")

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff * dt))

    def filter(self, value: float, dt: float) -> float:
        _finite(value, dt)
        if dt <= 0:
            raise ValueError("dt must be positive")
        if not self.initialized:
            self.value = self._raw = value
            self.initialized = True
            return value
        raw_derivative = (value - self._raw) / dt
        self._raw = value
        self.derivative += self._alpha(self.derivative_cutoff, dt) * (raw_derivative - self.derivative)
        cutoff = self.min_cutoff + self.beta * abs(self.derivative)
        self.value += self._alpha(cutoff, dt) * (value - self.value)
        return self.value


class TrajectoryPredictor:
    def __init__(self, horizon_ms: float = 15.0, max_projection: float = 0.08) -> None:
        _finite(horizon_ms, max_projection)
        if not 0 <= horizon_ms <= 100 or max_projection <= 0:
            raise ValueError("invalid prediction parameters")
        self.horizon_s = horizon_ms / 1000.0
        self.max_projection = max_projection
        self.reset()

    def update(self, point: SpatialPoint, timestamp: float) -> SpatialPoint:
        _finite(point.x, point.y, point.z, timestamp)
        if self._last_timestamp is not None:
            if timestamp <= self._last_timestamp:
                raise ValueError("timestamps must increase monotonically")
            if timestamp - self._last_timestamp > 0.25:
                self.reset()  # Do not extrapolate across tracking loss.
        dt = timestamp - self._last_timestamp if self._last_timestamp is not None else 1 / 60
        self._last_timestamp = timestamp
        output = []
        for kalman, smooth, value in zip(self._kalman, self._filters, (point.x, point.y, point.z)):
            kalman.update(value, dt)
            filtered = smooth.filter(kalman.position, dt)
            lead = kalman.project(self.horizon_s) - kalman.position
            output.append(filtered + max(-self.max_projection, min(self.max_projection, lead)))
        return SpatialPoint(*output, confidence=point.confidence)

    def reset(self) -> None:
        self._kalman = [Kalman1D() for _ in range(3)]
        self._filters = [OneEuroFilter() for _ in range(3)]
        self._last_timestamp: float | None = None


def second_order_project(samples: Iterable[tuple[float, float]], horizon_s: float) -> float:
    points = list(samples)[-3:]
    _finite(horizon_s)
    if not points or horizon_s < 0:
        raise ValueError("samples and a nonnegative horizon are required")
    for timestamp, value in points:
        _finite(timestamp, value)
    if len(points) == 1:
        return points[0][1]
    t1, x1 = points[-1]
    t0, x0 = points[-2]
    dt = t1 - t0
    if dt <= 0:
        raise ValueError("timestamps must increase")
    velocity = (x1 - x0) / dt
    acceleration = 0.0
    if len(points) == 3:
        t_prev, x_prev = points[0]
        dt_prev = t0 - t_prev
        if dt_prev <= 0:
            raise ValueError("timestamps must increase")
        acceleration = 2 * (velocity - (x0 - x_prev) / dt_prev) / (dt + dt_prev)
        # Secant velocity belongs at the interval midpoint; advance it to t1.
        velocity += acceleration * dt / 2
    return x1 + velocity * horizon_s + acceleration * horizon_s**2 / 2

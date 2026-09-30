from __future__ import annotations

import pytest

from phantomos_quantum.filters import OneEuroFilter, TrajectoryPredictor, second_order_project
from phantomos_quantum.models import SpatialPoint


def test_second_order_projection_tracks_linear_motion() -> None:
    assert second_order_project([(0.0, 0.0), (0.1, 1.0)], 0.01) == pytest.approx(1.1)


def test_predictor_rejects_nonmonotonic_timestamps() -> None:
    predictor = TrajectoryPredictor()
    predictor.update(SpatialPoint(0.1, 0.2, 0.0), 1.0)
    with pytest.raises(ValueError):
        predictor.update(SpatialPoint(0.2, 0.2, 0.0), 1.0)


def test_one_euro_reduces_stationary_noise() -> None:
    filter_ = OneEuroFilter()
    values = [filter_.filter(0.5 + (0.01 if i % 2 else -0.01), 1 / 60) for i in range(100)]
    assert max(values[-20:]) - min(values[-20:]) <= 0.025

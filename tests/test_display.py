from __future__ import annotations

from phantomos_quantum.display import Monitor, MultiMonitorLayout
from phantomos_quantum.models import SpatialPoint


def test_virtual_desktop_supports_negative_monitor_origin() -> None:
    layout = MultiMonitorLayout((Monitor(-1920, 0, 1920, 1080), Monitor(0, 0, 1920, 1080)))
    assert layout.map(SpatialPoint(0.0, 0.5)) == (-1920, 540)
    assert layout.map(SpatialPoint(1.0, 0.5)) == (1919, 540)

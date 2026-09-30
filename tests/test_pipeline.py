from __future__ import annotations

import time

from phantomos_quantum.camera import SyntheticFrameSource
from phantomos_quantum.inference import NullInference
from phantomos_quantum.native import DryRunBackend, NativeDispatcher
from phantomos_quantum.pipeline import LatestQueue, Pipeline


def test_latest_queue_drops_oldest_item() -> None:
    values = LatestQueue[int](1)
    values.put_latest(1)
    values.put_latest(2)
    assert values.dropped == 1
    assert values.get() == 2
    values.task_done()


def test_five_stage_pipeline_starts_and_stops_headlessly() -> None:
    backend = DryRunBackend()
    dispatcher = NativeDispatcher(backend, 640, 480, frozenset({"pointer_move"}))
    pipeline = Pipeline(SyntheticFrameSource(fps=120), NullInference(), dispatcher, queue_size=2)
    pipeline.start()
    time.sleep(0.08)
    pipeline.stop()
    assert not pipeline.errors
    assert pipeline.telemetry.snapshot().total_ms >= 0

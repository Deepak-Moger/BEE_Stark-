"""Command-line entry point for safe demos and hardware runs."""

from __future__ import annotations

import argparse
import signal
import threading
import time
from dataclasses import replace

from .camera import SyntheticFrameSource
from .config import EngineConfig
from .inference import NullInference
from .orchestrator import PhantomEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PhantomOS Quantum spatial interaction engine")
    parser.add_argument("--synthetic", action="store_true", help="run without opening a camera")
    parser.add_argument("--dry-run", action="store_true", default=True, help="record events, never inject OS input")
    parser.add_argument("--live-input", action="store_true", help="enable native OS input (explicit opt-in)")
    parser.add_argument("--seconds", type=float, default=0.0, help="stop after N seconds; 0 runs until interrupted")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = EngineConfig.from_env()
    if args.live_input:
        config = replace(config, dry_run=False)
    source = SyntheticFrameSource(fps=config.target_fps) if args.synthetic else None
    inference = NullInference() if args.synthetic else None
    engine = PhantomEngine(config, source=source, inference=inference,
                           on_telemetry=lambda snapshot: print(engine.pipeline.telemetry.format_bar(), end="\r"))
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    engine.start()
    started = time.monotonic()
    try:
        while not stop.is_set() and (args.seconds <= 0 or time.monotonic() - started < args.seconds):
            time.sleep(0.1)
    finally:
        engine.stop()
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

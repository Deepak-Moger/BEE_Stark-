"""PhantomOS Quantum spatial interaction engine.

The public API intentionally stays small. Optional integrations are imported lazily so
that headless CI and machines without camera/UI dependencies can use the core engine.
"""

from .config import EngineConfig
from .models import (
    GestureKind,
    HandObservation,
    Intent,
    IntentKind,
    Landmark3D,
    SensorFrame,
)
from .orchestrator import PhantomEngine

__all__ = [
    "EngineConfig",
    "GestureKind",
    "HandObservation",
    "Intent",
    "IntentKind",
    "Landmark3D",
    "PhantomEngine",
    "SensorFrame",
]

__version__ = "0.1.0"

"""High-level lifecycle, safety policy, and integration wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .camera import FrameSource, OpenCVFrameSource, SyntheticFrameSource
from .config import EngineConfig
from .errors import DependencyUnavailable, SafetyLockError
from .inference import InferenceBackend, MediaPipeHandsBackend, NullInference
from .models import HandObservation, VoiceCommand
from .native import NativeDispatcher, create_native_backend
from .pipeline import Pipeline
from .security import AirPasscode, HandBiometric, HandProfile


@dataclass(slots=True)
class SafetyPolicy:
    require_unlock: bool = True
    allow_system_actions: bool = False
    unlocked: bool = False

    def __post_init__(self) -> None:
        if not self.require_unlock:
            self.unlocked = True

    def lock(self) -> None:
        self.unlocked = False

    def assert_action_allowed(self, action: str) -> None:
        if self.require_unlock and not self.unlocked:
            raise SafetyLockError("workspace is locked")
        if action in {"close_window", "move_second_monitor", "open_code_editor", "summarize_selection"} and not self.allow_system_actions:
            raise SafetyLockError("system actions are disabled by policy")


class PhantomEngine:
    """Public engine facade.

    Construction is safe: no camera is opened and no OS event is emitted until
    `start()` is called, and `dry_run=True` remains the default.
    """

    def __init__(self, config: EngineConfig | None = None,
                 source: FrameSource | None = None,
                 inference: InferenceBackend | None = None,
                 on_telemetry: Callable[[object], None] | None = None) -> None:
        self.config = config or EngineConfig.from_env()
        self.policy = SafetyPolicy(self.config.require_unlock, self.config.allow_system_actions)
        self.biometric = HandBiometric()
        self.passcode = AirPasscode()
        self.profile_template: HandProfile | None = None
        source = source or OpenCVFrameSource(self.config.camera_index, self.config.target_fps)
        if inference is None:
            try:
                inference = MediaPipeHandsBackend(min_confidence=self.config.min_hand_confidence)
            except DependencyUnavailable:
                # Optional vision dependencies are not a reason for a headless
                # dry-run process to crash; the pipeline remains inspectable.
                inference = NullInference()
        backend = create_native_backend(self.config.native_backend, self.config.dry_run)
        self.dispatcher = NativeDispatcher(
            backend, self.config.screen_width, self.config.screen_height,
            self.config.allowed_actions, self.config.min_intent_confidence,
            require_unlock=self.config.require_unlock,
        )
        self.dispatcher.unlocked = self.policy.unlocked
        self.hud = None
        if self.config.enable_hud:
            from .hud import PhantomHud
            self.hud = PhantomHud(self.config.screen_width, self.config.screen_height)

        def telemetry_callback(snapshot: object) -> None:
            if self.hud is not None:
                self.hud.update_telemetry(snapshot)  # type: ignore[arg-type]
            if on_telemetry is not None:
                on_telemetry(snapshot)

        self.pipeline = Pipeline(
            source, inference, self.dispatcher, self.config.queue_size,
            self.config.prediction_horizon_ms, self.config.min_hand_confidence,
            telemetry_callback, self.lock,
        )

    @property
    def unlocked(self) -> bool:
        return self.policy.unlocked

    def enroll_hand(self, hand: HandObservation) -> HandProfile:
        profile = self.biometric.profile(hand)
        self.profile_template = profile
        return profile

    def unlock_gesture(self, gesture: str, hand: HandObservation) -> bool:
        if self.profile_template is None:
            raise SafetyLockError("enroll a hand profile before unlocking")
        score = self.biometric.similarity(self.profile_template, self.biometric.profile(hand))
        unlocked = self.passcode.observe(gesture, score)
        if unlocked:
            self.policy.unlocked = True
            self.dispatcher.unlocked = True
        return unlocked

    def lock(self) -> None:
        self.passcode.lock()
        self.policy.lock()
        self.dispatcher.unlocked = False

    def submit_voice(self, command: VoiceCommand) -> None:
        self.pipeline.submit_voice(command)

    def start(self) -> None:
        if self.hud is not None:
            self.hud.show()
        self.pipeline.start()

    def stop(self) -> None:
        self.pipeline.stop()
        if self.hud is not None:
            self.hud.close()

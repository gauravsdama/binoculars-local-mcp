"""Detector lifecycle and serialized inference ownership."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .config import RuntimeConfig
from .detector import Analysis, Binoculars
from .integrity import ModelVerification, verify_model_directory
from .profiles import ModelProfile, get_profile


@dataclass(frozen=True)
class RuntimeStatus:
    observer: ModelVerification
    performer: ModelVerification
    detector_loaded: bool
    active_device: str | None

    @property
    def ready(self) -> bool:
        return self.observer.verified and self.performer.verified


@dataclass(frozen=True)
class RuntimeAnalysis:
    analysis: Analysis
    threshold: float
    profile: ModelProfile
    device: str
    characters_analyzed: int


class DetectorRuntime:
    """Own one immutable configuration, detector instance, and inference lock."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.profile = get_profile(config.profile_name)
        self._detector: Binoculars | None = None
        self._lock = threading.Lock()

    def status(self) -> RuntimeStatus:
        observer, performer = self._verify_models()
        detector = self._detector
        return RuntimeStatus(
            observer=observer,
            performer=performer,
            detector_loaded=detector is not None,
            active_device=detector.device if detector is not None else None,
        )

    def analyze(self, text: str, mode: str) -> RuntimeAnalysis:
        if not self._lock.acquire(timeout=self.config.lock_timeout_seconds):
            raise TimeoutError(
                f"Detector remained busy for {self.config.lock_timeout_seconds:g} seconds"
            )
        try:
            detector = self._load_detector()
            analysis = detector.analyze(text)
            if not isinstance(analysis, Analysis):
                raise RuntimeError("Single-text analysis unexpectedly returned a batch")
            return RuntimeAnalysis(
                analysis=analysis,
                threshold=detector.threshold_for_mode(mode),
                profile=self.profile,
                device=detector.device,
                characters_analyzed=len(text),
            )
        finally:
            self._lock.release()

    def _verify_models(self) -> tuple[ModelVerification, ModelVerification]:
        return (
            verify_model_directory(self.config.observer_path, self.profile.observer_files),
            verify_model_directory(self.config.performer_path, self.profile.performer_files),
        )

    def _load_detector(self) -> Binoculars:
        if self._detector is not None:
            return self._detector
        if self.config.observer_path is None or self.config.performer_path is None:
            raise RuntimeError(
                "Local models are not configured. Set BINOCULARS_OBSERVER_MODEL and "
                "BINOCULARS_PERFORMER_MODEL to compatible local model directories."
            )
        observer, performer = self._verify_models()
        if not observer.verified or not performer.verified:
            details = "; ".join(
                message
                for message in (
                    f"observer: {observer.error}" if observer.error else "",
                    f"performer: {performer.error}" if performer.error else "",
                )
                if message
            )
            raise RuntimeError(f"Local model verification failed ({details})")
        self._detector = Binoculars(
            self.config.observer_path,
            self.config.performer_path,
            accuracy_threshold=self.profile.accuracy_threshold,
            low_fpr_threshold=self.profile.low_fpr_threshold,
            max_token_observed=self.profile.max_tokens,
            device=self.config.device,
        )
        return self._detector

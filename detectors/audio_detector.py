"""Audio threat detector using YAMNet classifications."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from security_ai_system.audio.audio_sources import load_audio_file
from security_ai_system.audio.yamnet_classifier import (
    YAMNET_SAMPLE_RATE,
    AudioClassification,
    YamNetClassifier,
    YamNetClassifierConfig,
    prepare_waveform,
)
from security_ai_system.detectors.base import AudioDetector as BaseAudioDetector
from security_ai_system.detectors.base import DetectorMetadata
from security_ai_system.utils.types import (
    Detection,
    DetectorKind,
    DetectorResult,
    RuntimeConfig,
)


@dataclass(frozen=True)
class AudioThreatDetectorConfig:
    """Configuration for audio threat detection."""

    sample_rate: int = YAMNET_SAMPLE_RATE
    classifier_config: YamNetClassifierConfig = field(default_factory=YamNetClassifierConfig)
    alert_cooldown_sec: float = 1.0


class AudioThreatDetector(BaseAudioDetector):
    """Detect screams, gunshots, glass breaking, and explosions in audio chunks."""

    metadata = DetectorMetadata(
        name="Audio Threat Detector",
        kind=DetectorKind.AUDIO,
        description="YAMNet audio classifier mapped to security threat labels.",
    )

    def __init__(
        self,
        camera_id: str,
        runtime: RuntimeConfig | None = None,
        config: AudioThreatDetectorConfig | None = None,
        classifier: YamNetClassifier | None = None,
    ) -> None:
        super().__init__(camera_id=camera_id, runtime=runtime)
        self.config = config or AudioThreatDetectorConfig()
        self.sample_rate = self.config.sample_rate
        self.classifier = classifier or YamNetClassifier(self.config.classifier_config)
        self._last_alert_at_by_label: dict[str, float] = {}

    def load(self) -> None:
        """Load YAMNet classifier resources."""

        if not self.classifier.is_loaded:
            self.classifier.load()
        self._loaded = True

    def predict(self, data: Any, timestamp: float | None = None) -> DetectorResult:
        """Run audio threat detection on one waveform chunk.

        Accepted inputs:
        - 1-D mono numpy-like waveform at 16 kHz;
        - 2-D waveform with channels, which is mixed down;
        - `(waveform, sample_rate)` tuple, resampled when needed;
        - file path string/Path for short demo clips.
        """

        self._require_loaded()
        started = time.perf_counter()
        now = timestamp or time.time()

        try:
            waveform = self._prepare_input(data)
            classifications = self.classifier.classify(waveform)
            detections, alerts = self._classifications_to_detections(classifications, now)
        except Exception as exc:
            return DetectorResult(
                detector=DetectorKind.AUDIO,
                camera_id=self.camera_id,
                timestamp=now,
                processing_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
            )

        return DetectorResult(
            detector=DetectorKind.AUDIO,
            camera_id=self.camera_id,
            timestamp=now,
            detections=detections,
            alerts=alerts,
            processing_ms=(time.perf_counter() - started) * 1000,
        )

    def _prepare_input(self, data: Any) -> np.ndarray:
        if isinstance(data, (str, Path)):
            return load_audio_file(data, sample_rate=self.sample_rate)

        sample_rate = self.sample_rate
        waveform = data
        if isinstance(data, tuple) and len(data) == 2:
            waveform, sample_rate = data

        prepared = prepare_waveform(np.asarray(waveform))
        if sample_rate != self.sample_rate:
            import librosa

            prepared = librosa.resample(
                prepared,
                orig_sr=int(sample_rate),
                target_sr=self.sample_rate,
            )
            prepared = prepare_waveform(prepared)
        return prepared

    def _classifications_to_detections(
        self,
        classifications: list[AudioClassification],
        timestamp: float,
    ) -> tuple[list[Detection], list[Detection]]:
        detections: list[Detection] = []
        alerts: list[Detection] = []

        for classification in classifications:
            detection = Detection(
                label=classification.display_label,
                confidence=classification.confidence,
                severity=classification.severity,
                threat_score=classification.threat_score,
                metadata={
                    "camera_id": self.camera_id,
                    "source_label": classification.source_label,
                    "threat_type": classification.threat_type,
                    "sample_rate": self.sample_rate,
                },
            )
            detections.append(detection)

            last_alert_at = self._last_alert_at_by_label.get(classification.display_label, 0.0)
            if timestamp - last_alert_at >= self.config.alert_cooldown_sec:
                self._last_alert_at_by_label[classification.display_label] = timestamp
                alerts.append(detection)

        return detections, alerts


AudioDetector = AudioThreatDetector

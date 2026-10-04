"""YAMNet audio classifier wrapper.

YAMNet is a pretrained TensorFlow model trained on AudioSet. It emits scores for
AudioSet classes such as "Screaming", "Gunshot, gunfire", "Glass", and
"Explosion". This module maps those real model labels into project threat labels.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from security_ai_system.utils.types import ModelPathConfig, Severity


YAMNET_SAMPLE_RATE = 16000

SCREAM = "scream"
GUNSHOT = "gunshot"
GLASS_BREAK = "glass_break"
EXPLOSION = "explosion"


DEFAULT_AUDIO_THREAT_ALIASES: dict[str, tuple[str, ...]] = {
    SCREAM: ("scream", "screaming", "shout", "yell", "crying"),
    GUNSHOT: ("gunshot", "gunfire"),
    GLASS_BREAK: ("glass", "glass breaking", "breaking glass", "shatter", "smash"),
    EXPLOSION: ("explosion", "blast"),
}


@dataclass(frozen=True)
class AudioThreatDefinition:
    """Metadata for a project audio threat class."""

    threat_type: str
    display_label: str
    score: int
    severity: Severity


AUDIO_THREAT_DEFINITIONS: dict[str, AudioThreatDefinition] = {
    SCREAM: AudioThreatDefinition(SCREAM, "SCREAM DETECTED", 60, Severity.HIGH),
    GUNSHOT: AudioThreatDefinition(GUNSHOT, "GUNSHOT DETECTED", 120, Severity.CRITICAL),
    GLASS_BREAK: AudioThreatDefinition(GLASS_BREAK, "GLASS BREAK DETECTED", 70, Severity.HIGH),
    EXPLOSION: AudioThreatDefinition(EXPLOSION, "EXPLOSION DETECTED", 100, Severity.CRITICAL),
}


@dataclass(frozen=True)
class AudioClassification:
    """One audio threat classification."""

    threat_type: str
    display_label: str
    source_label: str
    confidence: float
    threat_score: int
    severity: Severity


@dataclass(frozen=True)
class YamNetClassifierConfig:
    """Configuration for the YAMNet classifier."""

    model_paths: ModelPathConfig = ModelPathConfig()
    confidence_threshold: float = 0.20
    aggregate: str = "max"
    threat_aliases: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: DEFAULT_AUDIO_THREAT_ALIASES.copy()
    )


class YamNetClassifier:
    """Load YAMNet and classify waveform chunks."""

    def __init__(self, config: YamNetClassifierConfig | None = None) -> None:
        self.config = config or YamNetClassifierConfig()
        self._model: Any | None = None
        self._class_names: list[str] = []

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    def load(self) -> None:
        """Load YAMNet from TensorFlow Hub and read its AudioSet class map."""

        try:
            import tensorflow_hub as hub
        except ImportError as exc:
            raise ImportError(
                "tensorflow-hub is required for YAMNet. "
                "Install it with `pip install tensorflow-hub`."
            ) from exc

        self._model = hub.load(self.config.model_paths.yamnet_model_handle)
        self._class_names = self._load_class_names_from_model(self._model)

    def classify(self, waveform: np.ndarray) -> list[AudioClassification]:
        """Classify a mono 16 kHz waveform chunk."""

        if self._model is None:
            raise RuntimeError("YamNetClassifier.load() must be called before classify().")

        prepared = prepare_waveform(waveform)
        scores, _, _ = self._model(prepared)
        scores_np = self._to_numpy(scores)
        if scores_np.ndim == 1:
            class_scores = scores_np
        elif self.config.aggregate == "mean":
            class_scores = scores_np.mean(axis=0)
        else:
            class_scores = scores_np.max(axis=0)

        return self.match_threats(class_scores, self._class_names)

    def match_threats(
        self,
        class_scores: np.ndarray,
        class_names: list[str],
    ) -> list[AudioClassification]:
        """Map raw AudioSet scores to configured threat labels."""

        matches: list[AudioClassification] = []
        for threat_type, aliases in self.config.threat_aliases.items():
            best_label = ""
            best_score = 0.0

            for idx, class_name in enumerate(class_names):
                if idx >= len(class_scores):
                    break
                if self._label_matches_alias(class_name, aliases):
                    score = float(class_scores[idx])
                    if score > best_score:
                        best_score = score
                        best_label = class_name

            if best_score >= self.config.confidence_threshold:
                definition = AUDIO_THREAT_DEFINITIONS[threat_type]
                matches.append(
                    AudioClassification(
                        threat_type=definition.threat_type,
                        display_label=definition.display_label,
                        source_label=best_label,
                        confidence=best_score,
                        threat_score=definition.score,
                        severity=definition.severity,
                    )
                )

        matches.sort(key=lambda item: item.confidence, reverse=True)
        return matches

    def _load_class_names_from_model(self, model: Any) -> list[str]:
        """Read YAMNet class names from the model's class_map_path asset."""

        if not hasattr(model, "class_map_path"):
            raise RuntimeError("Loaded YAMNet model does not expose class_map_path().")

        class_map_path = model.class_map_path().numpy().decode("utf-8")
        class_names: list[str] = []
        with open(class_map_path, newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                display_name = row.get("display_name")
                if display_name:
                    class_names.append(display_name)

        if not class_names:
            raise RuntimeError("Could not read YAMNet class names from class map.")
        return class_names

    @staticmethod
    def _label_matches_alias(label: str, aliases: tuple[str, ...]) -> bool:
        normalized_label = normalize_label(label)
        return any(normalize_label(alias) in normalized_label for alias in aliases)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)


def normalize_label(label: str) -> str:
    """Normalize labels for robust alias matching."""

    normalized = label.strip().lower().replace("-", " ").replace("_", " ")
    normalized = re.sub(r"[,/()]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized)


def prepare_waveform(waveform: np.ndarray) -> np.ndarray:
    """Return a mono float32 waveform suitable for YAMNet."""

    array = np.asarray(waveform)
    if array.ndim == 2:
        array = array.mean(axis=1)
    elif array.ndim != 1:
        raise ValueError("Audio waveform must be 1-D mono or 2-D channels.")

    array = array.astype(np.float32, copy=False)
    if array.size == 0:
        raise ValueError("Audio waveform is empty.")

    peak = float(np.max(np.abs(array)))
    if peak > 1.0:
        array = array / peak

    return np.clip(array, -1.0, 1.0).astype(np.float32, copy=False)

"""Unit tests for audio threat detection helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import numpy as np  # noqa: E402
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None
    _NUMPY_AVAILABLE = False

if _NUMPY_AVAILABLE:
    from security_ai_system.audio.audio_sources import iter_audio_chunks  # noqa: E402
    from security_ai_system.audio.yamnet_classifier import (  # noqa: E402
        GUNSHOT,
        SCREAM,
        AudioClassification,
        YamNetClassifier,
        YamNetClassifierConfig,
        normalize_label,
        prepare_waveform,
    )
    from security_ai_system.detectors.audio_detector import (  # noqa: E402
        AudioThreatDetector,
        AudioThreatDetectorConfig,
    )
    from security_ai_system.utils.types import Severity  # noqa: E402
else:
    GUNSHOT = "gunshot"
    SCREAM = "scream"


class FakeClassifier:
    def __init__(self) -> None:
        self.loaded = False

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    def load(self) -> None:
        self.loaded = True

    def classify(self, waveform: np.ndarray) -> list[AudioClassification]:
        return [
            AudioClassification(
                threat_type=GUNSHOT,
                display_label="GUNSHOT DETECTED",
                source_label="Gunshot, gunfire",
                confidence=0.91,
                threat_score=120,
                severity=Severity.CRITICAL,
            )
        ]


@unittest.skipUnless(_NUMPY_AVAILABLE, "numpy not installed")
class AudioHelperTests(unittest.TestCase):
    def test_prepare_waveform_mixes_stereo_and_normalizes(self) -> None:
        stereo = np.array([[2.0, 0.0], [-2.0, 0.0]], dtype=np.float32)
        prepared = prepare_waveform(stereo)

        self.assertEqual(prepared.dtype, np.float32)
        self.assertEqual(prepared.shape, (2,))
        self.assertLessEqual(float(np.max(np.abs(prepared))), 1.0)

    def test_iter_audio_chunks_pads_last_chunk(self) -> None:
        waveform = np.ones(24000, dtype=np.float32)
        chunks = list(iter_audio_chunks(waveform, sample_rate=16000, chunk_seconds=1.0))

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].shape, (16000,))
        self.assertEqual(chunks[1].shape, (16000,))

    def test_normalize_label_handles_punctuation(self) -> None:
        self.assertEqual(normalize_label("Gunshot, gunfire"), "gunshot gunfire")


@unittest.skipUnless(_NUMPY_AVAILABLE, "numpy not installed")
class YamNetClassifierMappingTests(unittest.TestCase):
    def test_raw_yamnet_scores_map_to_project_threats(self) -> None:
        classifier = YamNetClassifier(
            YamNetClassifierConfig(confidence_threshold=0.3)
        )
        class_names = ["Speech", "Screaming", "Gunshot, gunfire"]
        scores = np.array([0.1, 0.72, 0.88], dtype=np.float32)

        matches = classifier.match_threats(scores, class_names)
        labels = [match.display_label for match in matches]

        self.assertEqual(labels[0], "GUNSHOT DETECTED")
        self.assertIn("SCREAM DETECTED", labels)
        self.assertEqual(matches[0].threat_score, 120)

    def test_scores_below_threshold_are_ignored(self) -> None:
        classifier = YamNetClassifier(
            YamNetClassifierConfig(confidence_threshold=0.8)
        )
        matches = classifier.match_threats(
            np.array([0.3], dtype=np.float32),
            ["Screaming"],
        )

        self.assertEqual(matches, [])


@unittest.skipUnless(_NUMPY_AVAILABLE, "numpy not installed")
class AudioThreatDetectorTests(unittest.TestCase):
    def test_detector_converts_classification_to_alert(self) -> None:
        detector = AudioThreatDetector(
            camera_id="mic-1",
            config=AudioThreatDetectorConfig(alert_cooldown_sec=1.0),
            classifier=FakeClassifier(),
        )
        detector.load()
        result = detector.predict(np.zeros(16000, dtype=np.float32), timestamp=10.0)

        self.assertIsNone(result.error)
        self.assertEqual(result.alerts[0].label, "GUNSHOT DETECTED")
        self.assertEqual(result.alerts[0].threat_score, 120)

    def test_alert_cooldown_keeps_detection_but_suppresses_repeat_alert(self) -> None:
        detector = AudioThreatDetector(
            camera_id="mic-1",
            config=AudioThreatDetectorConfig(alert_cooldown_sec=5.0),
            classifier=FakeClassifier(),
        )
        detector.load()
        first = detector.predict(np.zeros(16000, dtype=np.float32), timestamp=10.0)
        second = detector.predict(np.zeros(16000, dtype=np.float32), timestamp=11.0)

        self.assertEqual(len(first.alerts), 1)
        self.assertEqual(len(second.detections), 1)
        self.assertEqual(second.alerts, [])


if __name__ == "__main__":
    unittest.main()


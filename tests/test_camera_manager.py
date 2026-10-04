"""Unit tests for threaded multi-camera management."""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from security_ai_system.cameras.camera_manager import (  # noqa: E402
    CameraManager,
    CameraSourceConfig,
    CameraSourceType,
    FrameReader,
    infer_source_type,
)
from security_ai_system.detectors.base import DetectorMetadata, VisionDetector  # noqa: E402
from security_ai_system.utils.types import (  # noqa: E402
    Detection,
    DetectorKind,
    DetectorResult,
)


class FakeReader(FrameReader):
    """Deterministic reader for camera worker tests."""

    def __init__(self, frames: list[Any]) -> None:
        self.frames = list(frames)
        self.index = 0
        self.opened = False
        self.released = False
        self.lock = threading.Lock()

    def open(self) -> None:
        self.opened = True

    def read(self) -> tuple[bool, Any | None]:
        with self.lock:
            if self.index >= len(self.frames):
                return False, None
            frame = self.frames[self.index]
            self.index += 1
            return True, frame

    def restart(self) -> bool:
        return False

    def release(self) -> None:
        self.released = True


class FailingReader(FakeReader):
    def open(self) -> None:
        raise RuntimeError("source unavailable")


class FakeDetector(VisionDetector):
    metadata = DetectorMetadata(
        name="Fake Vision Detector",
        kind=DetectorKind.BAG,
        description="Test detector.",
    )

    def __init__(self, camera_id: str) -> None:
        super().__init__(camera_id=camera_id)
        self.load_count = 0
        self.predict_count = 0

    def load(self) -> None:
        self.load_count += 1
        self._loaded = True

    def predict(self, data: Any, timestamp: float | None = None) -> DetectorResult:
        self._require_loaded()
        self.predict_count += 1
        alert = Detection(
            label="TEST ALERT",
            confidence=1.0,
            threat_score=5,
        )
        return DetectorResult(
            detector=DetectorKind.BAG,
            camera_id=self.camera_id,
            timestamp=timestamp or 0.0,
            detections=[alert],
            alerts=[alert],
            annotated_frame=f"{data}-annotated",
        )


class CameraManagerTests(unittest.TestCase):
    def test_source_type_inference(self) -> None:
        self.assertEqual(infer_source_type(0), CameraSourceType.WEBCAM)
        self.assertEqual(infer_source_type("1"), CameraSourceType.WEBCAM)
        self.assertEqual(infer_source_type("rtsp://camera/live"), CameraSourceType.RTSP)
        self.assertEqual(infer_source_type("https://example.com/feed"), CameraSourceType.HTTP)
        self.assertEqual(infer_source_type("demo.mp4"), CameraSourceType.VIDEO_FILE)

    def test_worker_processes_frames_with_detector_pipeline(self) -> None:
        reader = FakeReader(["frame-1", "frame-2"])
        detector = FakeDetector(camera_id="cam-1")
        manager = CameraManager()
        worker = manager.add_camera(
            CameraSourceConfig(
                camera_id="cam-1",
                source=0,
                queue_size=1,
                result_queue_size=2,
                read_failures_before_reconnect=100,
            ),
            detectors=[detector],
            reader_factory=lambda config: reader,
        )

        worker.start()
        result = worker.read_result(timeout=2.0)
        worker.stop()

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.camera_id, "cam-1")
        self.assertEqual(result.frame, "frame-1-annotated")
        self.assertEqual(result.alerts[0].label, "TEST ALERT")
        self.assertEqual(detector.load_count, 1)
        self.assertGreaterEqual(detector.predict_count, 1)
        self.assertTrue(reader.opened)
        self.assertTrue(reader.released)

    def test_manager_runs_multiple_cameras_independently(self) -> None:
        readers = {
            "cam-a": FakeReader(["a1"]),
            "cam-b": FakeReader(["b1"]),
        }

        def reader_factory(config: CameraSourceConfig) -> FakeReader:
            return readers[config.camera_id]

        manager = CameraManager()
        manager.add_camera(
            CameraSourceConfig("cam-a", "demo_a.mp4", CameraSourceType.VIDEO_FILE),
            detectors=[FakeDetector("cam-a")],
            reader_factory=reader_factory,
        )
        manager.add_camera(
            CameraSourceConfig("cam-b", "demo_b.mp4", CameraSourceType.VIDEO_FILE),
            detectors=[FakeDetector("cam-b")],
            reader_factory=reader_factory,
        )

        manager.start_all()
        result_a = manager.get_worker("cam-a").read_result(timeout=2.0)
        result_b = manager.get_worker("cam-b").read_result(timeout=2.0)
        latest = manager.latest_results()
        manager.stop_all()

        self.assertIsNotNone(result_a)
        self.assertIsNotNone(result_b)
        self.assertIn("cam-a", latest)
        self.assertIn("cam-b", latest)
        self.assertEqual(latest["cam-a"].frame, "a1-annotated")
        self.assertEqual(latest["cam-b"].frame, "b1-annotated")

    def test_duplicate_camera_id_is_rejected(self) -> None:
        manager = CameraManager()
        config = CameraSourceConfig("cam-1", 0)
        manager.add_camera(config)

        with self.assertRaises(ValueError):
            manager.add_camera(config)

    def test_worker_reports_startup_failure(self) -> None:
        manager = CameraManager()
        worker = manager.add_camera(
            CameraSourceConfig("cam-failing", 0),
            reader_factory=lambda config: FailingReader([]),
        )

        worker.start()

        self.assertFalse(worker.wait_until_started(timeout=1.0))
        self.assertEqual(worker.status().last_error, "source unavailable")

        manager.remove_camera("cam-failing")
        with self.assertRaises(KeyError):
            manager.get_worker("cam-failing")


if __name__ == "__main__":
    unittest.main()


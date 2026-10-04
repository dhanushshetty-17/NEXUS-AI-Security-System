"""Threaded multi-camera source management.

This module supports webcam indexes, video files, and RTSP/HTTP CCTV streams via
OpenCV. Each camera source runs independently in its own worker thread and can
optionally run a detector pipeline on every frame.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol

from security_ai_system.detectors.base import VisionDetector
from security_ai_system.utils.types import Detection, DetectorResult


class CameraSourceType(str, Enum):
    """Supported visual camera source types."""

    WEBCAM = "webcam"
    VIDEO_FILE = "video_file"
    RTSP = "rtsp"
    HTTP = "http"


@dataclass(frozen=True)
class CameraSourceConfig:
    """Configuration for one camera/video source."""

    camera_id: str
    source: int | str | Path
    source_type: CameraSourceType = CameraSourceType.WEBCAM
    display_name: str | None = None
    width: int | None = None
    height: int | None = None
    target_fps: float | None = None
    queue_size: int = 2
    result_queue_size: int = 4
    reconnect_delay_sec: float = 2.0
    loop_video: bool = True
    read_failures_before_reconnect: int = 10

    @property
    def label(self) -> str:
        return self.display_name or self.camera_id


@dataclass
class FramePacket:
    """One frame emitted by a camera worker."""

    camera_id: str
    timestamp: float
    frame: Any
    frame_index: int
    fps: float


@dataclass
class CameraPipelineResult:
    """Processed output for one frame and one camera source."""

    camera_id: str
    timestamp: float
    frame: Any
    frame_index: int
    fps: float
    detector_results: list[DetectorResult] = field(default_factory=list)
    alerts: list[Detection] = field(default_factory=list)
    processing_ms: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class CameraStatus:
    """Thread-safe snapshot of a camera worker's status."""

    camera_id: str
    source: str
    running: bool
    connected: bool
    models_ready: bool
    fps: float
    frame_count: int
    last_error: str | None = None


class FrameReader(Protocol):
    """Protocol implemented by real and fake frame readers."""

    def open(self) -> None:
        """Open the underlying frame source."""

    def read(self) -> tuple[bool, Any | None]:
        """Read one frame."""

    def restart(self) -> bool:
        """Try to restart the source after end-of-file or read failure."""

    def release(self) -> None:
        """Release the frame source."""


class OpenCVCameraReader:
    """OpenCV VideoCapture-backed frame reader."""

    def __init__(self, config: CameraSourceConfig) -> None:
        self.config = config
        self._capture: Any | None = None

    def open(self) -> None:
        """Open webcam, video file, RTSP, or HTTP source."""

        import cv2

        source = self._opencv_source(self.config.source, self.config.source_type)
        capture = cv2.VideoCapture(source)
        if self.config.width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        if self.config.height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open camera source: {self.config.source}")
        self._capture = capture

    def read(self) -> tuple[bool, Any | None]:
        if self._capture is None:
            raise RuntimeError("OpenCVCameraReader.open() must be called before read().")
        return self._capture.read()

    def restart(self) -> bool:
        """Restart a stream or rewind a video file."""

        import cv2

        if self._capture is None:
            return False
        if self.config.source_type == CameraSourceType.VIDEO_FILE and self.config.loop_video:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return True

        self.release()
        try:
            self.open()
        except RuntimeError:
            return False
        return True

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    @staticmethod
    def _opencv_source(source: int | str | Path, source_type: CameraSourceType) -> int | str:
        if source_type == CameraSourceType.WEBCAM:
            if isinstance(source, int):
                return source
            if str(source).isdigit():
                return int(str(source))
        return str(source)


class CameraWorker:
    """Threaded camera source and per-frame detector pipeline."""

    def __init__(
        self,
        config: CameraSourceConfig,
        detectors: list[VisionDetector] | None = None,
        reader_factory: Callable[[CameraSourceConfig], FrameReader] | None = None,
    ) -> None:
        self.config = config
        self.detectors = detectors or []
        self.reader_factory = reader_factory or OpenCVCameraReader
        self.frame_queue: queue.Queue[FramePacket] = queue.Queue(maxsize=config.queue_size)
        self.result_queue: queue.Queue[CameraPipelineResult] = queue.Queue(
            maxsize=config.result_queue_size
        )
        self._stop_event = threading.Event()
        self._startup_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._detector_load_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._reader: FrameReader | None = None
        self._latest_frame: FramePacket | None = None
        self._latest_result: CameraPipelineResult | None = None
        self._running = False
        self._connected = False
        self._models_ready = False
        self._fps = 0.0
        self._frame_count = 0
        self._last_error: str | None = None

    def start(self, load_detectors: bool = True) -> None:
        """Start the camera worker thread."""

        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._startup_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"CameraWorker-{self.config.camera_id}",
            kwargs={"load_detectors": load_detectors},
            daemon=True,
        )
        self._thread.start()

    def wait_until_started(self, timeout: float = 5.0) -> bool:
        """Wait for source initialization and report whether it connected."""

        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        if not self._startup_event.wait(timeout=timeout):
            return False
        with self._lock:
            return self._connected

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the camera worker and release resources."""

        self._stop_event.set()
        self._startup_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if self._detector_load_thread is not None:
            self._detector_load_thread.join(timeout=timeout)
        if self._reader is not None and not self._thread.is_alive():
            self._reader.release()
            self._reader = None
        for detector in self.detectors:
            detector.close()
        with self._lock:
            self._running = False
            self._connected = False
            self._models_ready = False

    def read_frame(self, timeout: float = 0.0) -> FramePacket | None:
        """Read the next queued raw frame."""

        return self._queue_get(self.frame_queue, timeout)

    def read_result(self, timeout: float = 0.0) -> CameraPipelineResult | None:
        """Read the next queued processed result."""

        return self._queue_get(self.result_queue, timeout)

    def latest_frame(self) -> FramePacket | None:
        with self._lock:
            return self._latest_frame

    def latest_result(self) -> CameraPipelineResult | None:
        with self._lock:
            return self._latest_result

    def status(self) -> CameraStatus:
        with self._lock:
            return CameraStatus(
                camera_id=self.config.camera_id,
                source=str(self.config.source),
                running=self._running,
                connected=self._connected,
                models_ready=self._models_ready,
                fps=self._fps,
                frame_count=self._frame_count,
                last_error=self._last_error,
            )

    def _run(self, load_detectors: bool) -> None:
        with self._lock:
            self._running = True
            self._last_error = None

        try:
            self._reader = self.reader_factory(self.config)
            self._reader.open()
            with self._lock:
                self._connected = True
                self._models_ready = not load_detectors or not self.detectors
        except Exception as exc:
            if self._reader is not None:
                self._reader.release()
                self._reader = None
            with self._lock:
                self._last_error = str(exc)
                self._running = False
                self._connected = False
                self._models_ready = False
            self._startup_event.set()
            return

        self._startup_event.set()

        if load_detectors and self.detectors:
            self._detector_load_thread = threading.Thread(
                target=self._load_detectors_background,
                name=f"DetectorLoader-{self.config.camera_id}",
                daemon=True,
            )
            self._detector_load_thread.start()

        last_frame_time: float | None = None
        read_failures = 0

        try:
            while not self._stop_event.is_set():
                ok, frame = self._safe_read()
                if not ok or frame is None:
                    read_failures += 1
                    if read_failures >= self.config.read_failures_before_reconnect:
                        if not self._restart_reader():
                            self._stop_event.wait(self.config.reconnect_delay_sec)
                        read_failures = 0
                    continue

                read_failures = 0
                timestamp = time.time()
                fps = self._update_fps(timestamp, last_frame_time)
                last_frame_time = timestamp
                self._frame_count += 1

                packet = FramePacket(
                    camera_id=self.config.camera_id,
                    timestamp=timestamp,
                    frame=frame,
                    frame_index=self._frame_count,
                    fps=fps,
                )
                self._publish_frame(packet)
                result = self._process_packet(packet)
                self._publish_result(result)
                self._sleep_for_target_fps(last_frame_time)
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)

        if self._reader is not None:
            self._reader.release()
        with self._lock:
            self._running = False
            self._connected = False
            self._models_ready = False

    def _load_detectors(self) -> None:
        for detector in self.detectors:
            if not detector.is_loaded:
                try:
                    detector.load()
                except Exception as exc:
                    with self._lock:
                        self._last_error = str(exc)

    def _load_detectors_background(self) -> None:
        self._load_detectors()
        with self._lock:
            self._models_ready = all(detector.is_loaded for detector in self.detectors)

    def _safe_read(self) -> tuple[bool, Any | None]:
        try:
            if self._reader is None:
                return False, None
            return self._reader.read()
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
                self._connected = False
            return False, None

    def _restart_reader(self) -> bool:
        if self._reader is None:
            return False
        try:
            restarted = self._reader.restart()
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
                self._connected = False
            return False

        with self._lock:
            self._connected = restarted
            if restarted:
                self._last_error = None
        return restarted

    def _process_packet(self, packet: FramePacket) -> CameraPipelineResult:
        started = time.perf_counter()
        detector_results: list[DetectorResult] = []
        alerts: list[Detection] = []
        display_frame = packet.frame
        error: str | None = None

        for detector in self.detectors:
            if not detector.is_loaded:
                error = "AI models are still loading"
                continue
            try:
                result = detector.predict(display_frame, timestamp=packet.timestamp)
            except Exception as exc:
                error = str(exc)
                continue

            detector_results.append(result)
            alerts.extend(result.alerts)
            if result.annotated_frame is not None:
                display_frame = result.annotated_frame
            if result.error:
                error = result.error

        return CameraPipelineResult(
            camera_id=packet.camera_id,
            timestamp=packet.timestamp,
            frame=display_frame,
            frame_index=packet.frame_index,
            fps=packet.fps,
            detector_results=detector_results,
            alerts=alerts,
            processing_ms=(time.perf_counter() - started) * 1000,
            error=error,
        )

    def _publish_frame(self, packet: FramePacket) -> None:
        with self._lock:
            self._latest_frame = packet
        self._put_latest(self.frame_queue, packet)

    def _publish_result(self, result: CameraPipelineResult) -> None:
        with self._lock:
            self._latest_result = result
        self._put_latest(self.result_queue, result)

    def _update_fps(self, timestamp: float, last_frame_time: float | None) -> float:
        if last_frame_time is None:
            fps = 0.0
        else:
            dt = timestamp - last_frame_time
            fps = 0.0 if dt <= 0 else 1.0 / dt
        with self._lock:
            self._fps = fps
        return fps

    def _sleep_for_target_fps(self, last_frame_time: float) -> None:
        if not self.config.target_fps or self.config.target_fps <= 0:
            return
        frame_period = 1.0 / self.config.target_fps
        elapsed = time.time() - last_frame_time
        remaining = frame_period - elapsed
        if remaining > 0:
            time.sleep(remaining)

    @staticmethod
    def _put_latest(target_queue: queue.Queue[Any], item: Any) -> None:
        try:
            target_queue.put_nowait(item)
        except queue.Full:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                pass
            target_queue.put_nowait(item)

    @staticmethod
    def _queue_get(target_queue: queue.Queue[Any], timeout: float) -> Any | None:
        try:
            if timeout and timeout > 0:
                return target_queue.get(timeout=timeout)
            return target_queue.get_nowait()
        except queue.Empty:
            return None


class CameraManager:
    """Manage multiple independent camera workers."""

    def __init__(self) -> None:
        self._workers: dict[str, CameraWorker] = {}
        self._lock = threading.Lock()

    def add_camera(
        self,
        config: CameraSourceConfig,
        detectors: list[VisionDetector] | None = None,
        reader_factory: Callable[[CameraSourceConfig], FrameReader] | None = None,
    ) -> CameraWorker:
        """Register a camera source and return its worker."""

        with self._lock:
            if config.camera_id in self._workers:
                raise ValueError(f"Camera already exists: {config.camera_id}")
            worker = CameraWorker(config, detectors=detectors, reader_factory=reader_factory)
            self._workers[config.camera_id] = worker
            return worker

    def remove_camera(self, camera_id: str) -> None:
        """Stop and remove a camera worker."""

        with self._lock:
            worker = self._workers.pop(camera_id)
        worker.stop()

    def start_all(self, load_detectors: bool = True) -> None:
        """Start all registered cameras."""

        for worker in self.workers():
            worker.start(load_detectors=load_detectors)

    def stop_all(self) -> None:
        """Stop all registered cameras."""

        for worker in self.workers():
            worker.stop()

    def workers(self) -> list[CameraWorker]:
        with self._lock:
            return list(self._workers.values())

    def get_worker(self, camera_id: str) -> CameraWorker:
        with self._lock:
            return self._workers[camera_id]

    def statuses(self) -> list[CameraStatus]:
        return [worker.status() for worker in self.workers()]

    def latest_results(self) -> dict[str, CameraPipelineResult]:
        results: dict[str, CameraPipelineResult] = {}
        for worker in self.workers():
            result = worker.latest_result()
            if result is not None:
                results[worker.config.camera_id] = result
        return results

    def drain_alerts(self) -> list[Detection]:
        """Collect currently queued alerts from all camera result queues."""

        alerts: list[Detection] = []
        for worker in self.workers():
            while True:
                result = worker.read_result(timeout=0.0)
                if result is None:
                    break
                alerts.extend(result.alerts)
        return alerts


def infer_source_type(source: int | str | Path) -> CameraSourceType:
    """Infer a source type from a webcam index, URL, or file path."""

    if isinstance(source, int):
        return CameraSourceType.WEBCAM

    text = str(source).lower()
    if text.isdigit():
        return CameraSourceType.WEBCAM
    if text.startswith("rtsp://"):
        return CameraSourceType.RTSP
    if text.startswith("http://") or text.startswith("https://"):
        return CameraSourceType.HTTP
    return CameraSourceType.VIDEO_FILE

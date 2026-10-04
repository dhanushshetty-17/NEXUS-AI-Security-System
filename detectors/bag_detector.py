"""Suspicious unattended bag detection.

This detector uses:
- Ultralytics YOLOv8 COCO object detection for `person`, `backpack`,
  `handbag`, and `suitcase`.
- DeepSORT tracking for stable person and bag IDs.
- Rule-based ownership and unattended timing logic.

The implementation is intentionally conservative for demo stability: it avoids
claiming intent and only raises an alert when a tracked bag remains separated
from its associated owner for a configurable timeout.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from security_ai_system.detectors.base import DetectorMetadata, VisionDetector
from security_ai_system.trackers.tracker import (
    DeepSortTracker,
    TrackedObject,
    TrackerDetection,
)
from security_ai_system.utils.drawing import (
    BAG_STYLE,
    PERSON_STYLE,
    SUSPICIOUS_STYLE,
    draw_connection,
    draw_labeled_box,
    draw_status_text,
)
from security_ai_system.utils.types import (
    BoundingBox,
    Detection,
    DetectorKind,
    DetectorResult,
    ModelPathConfig,
    RuntimeConfig,
    Severity,
)


PERSON_LABEL = "person"
BAG_LABELS = {"backpack", "handbag", "suitcase"}
SUPPORTED_LABELS = {PERSON_LABEL, *BAG_LABELS}


@dataclass(frozen=True)
class BagDetectorConfig:
    """Tunable parameters for unattended bag detection."""

    unattended_timeout_sec: float = 10.0
    owner_assignment_max_distance_px: float = 220.0
    owner_absent_distance_px: float = 280.0
    stale_track_timeout_sec: float = 8.0
    snapshot_cooldown_sec: float = 20.0
    model_paths: ModelPathConfig = ModelPathConfig()


@dataclass
class BagState:
    """State retained for a tracked bag across frames."""

    bag_id: str
    owner_track_id: str | None = None
    unattended_since: float | None = None
    suspicious_since: float | None = None
    last_seen: float = 0.0
    last_snapshot_at: float = 0.0
    last_snapshot_path: str | None = None


@dataclass(frozen=True)
class BagStatus:
    """Current frame status for a tracked bag."""

    bag: TrackedObject
    owner: TrackedObject | None
    owner_distance_px: float | None
    unattended_seconds: float
    countdown_seconds: float
    is_suspicious: bool


class SuspiciousBagDetector(VisionDetector):
    """Detect and alert on unattended bags in BGR video frames."""

    metadata = DetectorMetadata(
        name="Suspicious Unattended Bag Detector",
        kind=DetectorKind.BAG,
        description="YOLOv8 + DeepSORT unattended bag ownership detector.",
    )

    def __init__(
        self,
        camera_id: str,
        runtime: RuntimeConfig | None = None,
        config: BagDetectorConfig | None = None,
        tracker: DeepSortTracker | None = None,
    ) -> None:
        super().__init__(camera_id=camera_id, runtime=runtime)
        self.config = config or BagDetectorConfig()
        self.tracker = tracker or DeepSortTracker()
        self._model: Any | None = None
        self._bag_states: dict[str, BagState] = {}

    def load(self) -> None:
        """Load YOLOv8 object detector and DeepSORT tracker."""

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for bag detection. "
                "Install it with `pip install ultralytics`."
            ) from exc

        weights = self._resolve_yolo_weights(self.config.model_paths.yolo_object_weights)
        self._model = YOLO(weights)
        if not self.tracker.is_loaded:
            self.tracker.load()
        self._loaded = True

    def predict(self, data: Any, timestamp: float | None = None) -> DetectorResult:
        """Run unattended bag detection on one OpenCV BGR frame."""

        self._require_loaded()
        started = time.perf_counter()
        now = timestamp or time.time()

        if data is None or not hasattr(data, "shape"):
            return self._empty_result(
                detector=DetectorKind.BAG,
                timestamp=now,
                error="Bag detector expected an OpenCV/Numpy image frame.",
            )

        frame = data
        annotated = frame.copy()

        try:
            tracker_detections = self._detect_supported_objects(frame)
            tracks = self.tracker.update(tracker_detections, frame=frame)
            persons = [track for track in tracks if track.label == PERSON_LABEL]
            bags = [track for track in tracks if track.label in BAG_LABELS]
            statuses = self._update_bag_states(bags, persons, now)
            self._remove_stale_bag_states(now)

            detections = self._build_detections(persons, bags, statuses)
            alerts = self._draw_and_collect_alerts(
                annotated=annotated,
                persons=persons,
                statuses=statuses,
                timestamp=now,
            )
        except Exception as exc:
            return DetectorResult(
                detector=DetectorKind.BAG,
                camera_id=self.camera_id,
                timestamp=now,
                annotated_frame=annotated,
                processing_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
            )

        return DetectorResult(
            detector=DetectorKind.BAG,
            camera_id=self.camera_id,
            timestamp=now,
            detections=detections,
            annotated_frame=annotated,
            alerts=alerts,
            processing_ms=(time.perf_counter() - started) * 1000,
        )

    def reset(self) -> None:
        """Clear retained bag ownership state."""

        self._bag_states.clear()

    def _resolve_yolo_weights(self, configured_path: Path) -> str:
        """Resolve local weights while allowing official YOLOv8 auto-download."""

        path = Path(configured_path)
        if path.exists():
            return str(path)

        official_names = {
            "yolov8n.pt",
            "yolov8s.pt",
            "yolov8m.pt",
            "yolov8l.pt",
            "yolov8x.pt",
        }
        if path.name in official_names:
            return path.name

        raise FileNotFoundError(
            f"YOLO object weights not found: {path}. "
            "Use an official YOLOv8 name such as yolov8n.pt or place weights in models/."
        )

    def _detect_supported_objects(self, frame: Any) -> list[TrackerDetection]:
        """Run YOLOv8 and convert selected classes to tracker detections."""

        if self._model is None:
            raise RuntimeError("YOLO model is not loaded.")

        predict_kwargs: dict[str, Any] = {
            "conf": self.runtime.confidence_threshold,
            "iou": self.runtime.iou_threshold,
            "imgsz": self.runtime.inference_size,
            "max_det": self.runtime.max_detections,
            "augment": self.runtime.test_time_augmentation,
            "verbose": False,
        }
        if self.runtime.device != "auto":
            predict_kwargs["device"] = self.runtime.device

        result = self._model.predict(frame, **predict_kwargs)[0]
        names = result.names
        tracker_detections: list[TrackerDetection] = []

        if result.boxes is None:
            return tracker_detections

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            label = str(names[class_id])
            if label not in SUPPORTED_LABELS:
                continue

            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = BoundingBox(int(x1), int(y1), int(x2), int(y2))
            if bbox.width <= 0 or bbox.height <= 0:
                continue

            tracker_detections.append(
                TrackerDetection(
                    bbox=bbox,
                    confidence=confidence,
                    label=label,
                )
            )

        return tracker_detections

    def _update_bag_states(
        self,
        bags: list[TrackedObject],
        persons: list[TrackedObject],
        timestamp: float,
    ) -> dict[str, BagStatus]:
        """Update ownership and unattended timers for all visible bags."""

        persons_by_id = {person.track_id: person for person in persons}
        statuses: dict[str, BagStatus] = {}

        for bag in bags:
            state = self._bag_states.setdefault(bag.track_id, BagState(bag_id=bag.track_id))
            state.last_seen = timestamp

            owner = persons_by_id.get(state.owner_track_id or "")
            nearest_person, nearest_distance = self._nearest_person(bag, persons)

            if state.owner_track_id is None and nearest_person is not None:
                if nearest_distance <= self.config.owner_assignment_max_distance_px:
                    state.owner_track_id = nearest_person.track_id
                    owner = nearest_person

            owner_distance = self._center_distance(bag, owner) if owner else None
            owner_is_near = (
                owner_distance is not None
                and owner_distance <= self.config.owner_absent_distance_px
            )

            if owner_is_near:
                state.unattended_since = None
                state.suspicious_since = None
            else:
                if state.unattended_since is None:
                    state.unattended_since = timestamp
                unattended_elapsed = timestamp - state.unattended_since
                if unattended_elapsed >= self.config.unattended_timeout_sec:
                    if state.suspicious_since is None:
                        state.suspicious_since = timestamp

            unattended_seconds = (
                0.0 if state.unattended_since is None else timestamp - state.unattended_since
            )
            countdown_seconds = max(
                0.0,
                self.config.unattended_timeout_sec - unattended_seconds,
            )

            statuses[bag.track_id] = BagStatus(
                bag=bag,
                owner=owner,
                owner_distance_px=owner_distance,
                unattended_seconds=unattended_seconds,
                countdown_seconds=countdown_seconds,
                is_suspicious=state.suspicious_since is not None,
            )

        return statuses

    def _remove_stale_bag_states(self, timestamp: float) -> None:
        """Drop bag state for tracks that have not been visible recently."""

        stale_ids = [
            bag_id
            for bag_id, state in self._bag_states.items()
            if timestamp - state.last_seen > self.config.stale_track_timeout_sec
        ]
        for bag_id in stale_ids:
            del self._bag_states[bag_id]

    def _build_detections(
        self,
        persons: list[TrackedObject],
        bags: list[TrackedObject],
        statuses: dict[str, BagStatus],
    ) -> list[Detection]:
        """Convert tracked people and bags to common Detection objects."""

        detections: list[Detection] = []

        for person in persons:
            detections.append(
                Detection(
                    label="PERSON",
                    confidence=person.confidence,
                    bbox=person.bbox,
                    track_id=person.track_id,
                    global_id=person.global_id,
                    metadata={"source_label": person.label},
                )
            )

        for bag in bags:
            status = statuses.get(bag.track_id)
            is_suspicious = bool(status and status.is_suspicious)
            detections.append(
                Detection(
                    label="SUSPICIOUS BAG" if is_suspicious else "BAG",
                    confidence=bag.confidence,
                    bbox=bag.bbox,
                    track_id=bag.track_id,
                    global_id=bag.global_id,
                    severity=Severity.HIGH if is_suspicious else Severity.INFO,
                    threat_score=40 if is_suspicious else 0,
                    metadata={
                        "source_label": bag.label,
                        "owner_track_id": status.owner.track_id if status and status.owner else None,
                        "unattended_seconds": status.unattended_seconds if status else 0.0,
                    },
                )
            )

        return detections

    def _draw_and_collect_alerts(
        self,
        annotated: Any,
        persons: list[TrackedObject],
        statuses: dict[str, BagStatus],
        timestamp: float,
    ) -> list[Detection]:
        """Draw annotations and return active suspicious-bag alerts."""

        for person in persons:
            draw_labeled_box(
                annotated,
                person.bbox,
                f"PERSON ID {person.track_id}",
                PERSON_STYLE,
            )

        alerts: list[Detection] = []

        for status in statuses.values():
            state = self._bag_states[status.bag.track_id]
            if status.owner is not None:
                draw_connection(
                    annotated,
                    status.owner.bbox.center,
                    status.bag.bbox.center,
                )

            if status.is_suspicious:
                label = f"SUSPICIOUS BAG ID {status.bag.track_id}"
                style = SUSPICIOUS_STYLE
                draw_labeled_box(annotated, status.bag.bbox, label, style)
                snapshot_path = self._maybe_save_snapshot(
                    annotated,
                    status.bag.track_id,
                    state,
                    timestamp,
                )
                alerts.append(
                    Detection(
                        label="SUSPICIOUS BAG",
                        confidence=status.bag.confidence,
                        bbox=status.bag.bbox,
                        track_id=status.bag.track_id,
                        global_id=status.bag.global_id,
                        severity=Severity.HIGH,
                        threat_score=40,
                        metadata={
                            "camera_id": self.camera_id,
                            "source_label": status.bag.label,
                            "owner_track_id": status.owner.track_id if status.owner else None,
                            "owner_distance_px": status.owner_distance_px,
                            "unattended_seconds": round(status.unattended_seconds, 2),
                            "snapshot_path": snapshot_path,
                        },
                    )
                )
            else:
                label = f"BAG ID {status.bag.track_id}"
                style = BAG_STYLE
                draw_labeled_box(annotated, status.bag.bbox, label, style)

            if status.countdown_seconds > 0 and status.unattended_seconds > 0:
                draw_status_text(
                    annotated,
                    f"Unattended alert in {status.countdown_seconds:.1f}s",
                    (status.bag.bbox.x1, min(status.bag.bbox.y2 + 20, annotated.shape[0] - 10)),
                    color=(0, 220, 255),
                )

        return alerts

    def _maybe_save_snapshot(
        self,
        annotated: Any,
        bag_id: str,
        state: BagState,
        timestamp: float,
    ) -> str | None:
        """Save suspicious-bag evidence snapshots with cooldown."""

        if not self.runtime.save_snapshots:
            return state.last_snapshot_path

        if timestamp - state.last_snapshot_at < self.config.snapshot_cooldown_sec:
            return state.last_snapshot_path

        import cv2

        snapshot_dir = Path(self.runtime.output_dir) / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        safe_camera_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.camera_id)
        safe_bag_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", bag_id)
        filename = f"{safe_camera_id}_suspicious_bag_{safe_bag_id}_{int(timestamp)}.jpg"
        path = snapshot_dir / filename

        if cv2.imwrite(str(path), annotated):
            state.last_snapshot_at = timestamp
            state.last_snapshot_path = str(path)
        return state.last_snapshot_path

    @staticmethod
    def _nearest_person(
        bag: TrackedObject,
        persons: list[TrackedObject],
    ) -> tuple[TrackedObject | None, float]:
        if not persons:
            return None, math.inf

        nearest = min(persons, key=lambda person: SuspiciousBagDetector._center_distance(bag, person))
        return nearest, SuspiciousBagDetector._center_distance(bag, nearest)

    @staticmethod
    def _center_distance(first: TrackedObject, second: TrackedObject) -> float:
        return math.dist(first.bbox.center, second.bbox.center)


BagDetector = SuspiciousBagDetector

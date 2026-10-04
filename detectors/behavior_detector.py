"""Human behavior detector using YOLOv8 pose plus motion heuristics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from security_ai_system.behavior.motion_analysis import (
    CentroidTrackMatcher,
    MotionAnalyzer,
    MotionFinding,
)
from security_ai_system.behavior.pose_analysis import (
    Keypoint,
    PoseAnalyzer,
    PoseFinding,
    PoseObservation,
)
from security_ai_system.detectors.base import DetectorMetadata, VisionDetector
from security_ai_system.utils.drawing import BEHAVIOR_STYLE, draw_labeled_box
from security_ai_system.utils.types import (
    BoundingBox,
    Detection,
    DetectorKind,
    DetectorResult,
    ModelPathConfig,
    RuntimeConfig,
    Severity,
)


@dataclass(frozen=True)
class BehaviorDetectorConfig:
    """Configuration for pose and motion behavior detection."""

    model_paths: ModelPathConfig = ModelPathConfig()
    min_pose_confidence: float = 0.35
    track_match_distance_px: float = 130.0


class BehaviorDetector(VisionDetector):
    """Detect running, fighting, falling, loitering, and suspicious movement."""

    metadata = DetectorMetadata(
        name="Human Behavior Detector",
        kind=DetectorKind.BEHAVIOR,
        description="YOLOv8 pose with rule-based movement analysis.",
    )

    def __init__(
        self,
        camera_id: str,
        runtime: RuntimeConfig | None = None,
        config: BehaviorDetectorConfig | None = None,
        pose_analyzer: PoseAnalyzer | None = None,
        motion_analyzer: MotionAnalyzer | None = None,
        track_matcher: CentroidTrackMatcher | None = None,
    ) -> None:
        super().__init__(camera_id=camera_id, runtime=runtime)
        self.config = config or BehaviorDetectorConfig()
        self.pose_analyzer = pose_analyzer or PoseAnalyzer()
        self.motion_analyzer = motion_analyzer or MotionAnalyzer()
        self.track_matcher = track_matcher or CentroidTrackMatcher(
            max_distance_px=self.config.track_match_distance_px
        )
        self._model: Any | None = None
        self._alert_cooldowns: dict[tuple[str, str], float] = {}  # (track_id, label) -> last_alert_time
        self._alert_cooldown_sec: float = 10.0

    def load(self) -> None:
        """Load YOLOv8 pose weights."""

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for behavior detection. "
                "Install it with `pip install ultralytics`."
            ) from exc

        weights = self._resolve_pose_weights(self.config.model_paths.yolo_pose_weights)
        self._model = YOLO(weights)
        self._loaded = True

    def predict(self, data: Any, timestamp: float | None = None) -> DetectorResult:
        """Run behavior detection on one OpenCV BGR frame."""

        self._require_loaded()
        started = time.perf_counter()
        now = timestamp or time.time()

        if data is None or not hasattr(data, "shape"):
            return self._empty_result(
                detector=DetectorKind.BEHAVIOR,
                timestamp=now,
                error="Behavior detector expected an OpenCV/Numpy image frame.",
            )

        frame = data
        annotated = frame.copy()

        try:
            observations = self._extract_pose_observations(frame, now)
            detections, alerts = self._analyze_observations(observations, now)
            self._draw_observations(annotated, observations, detections)
        except Exception as exc:
            return DetectorResult(
                detector=DetectorKind.BEHAVIOR,
                camera_id=self.camera_id,
                timestamp=now,
                annotated_frame=annotated,
                processing_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
            )

        return DetectorResult(
            detector=DetectorKind.BEHAVIOR,
            camera_id=self.camera_id,
            timestamp=now,
            detections=detections,
            annotated_frame=annotated,
            alerts=alerts,
            processing_ms=(time.perf_counter() - started) * 1000,
        )

    def _resolve_pose_weights(self, configured_path: Path) -> str:
        """Resolve local pose weights while allowing official auto-download."""

        path = Path(configured_path)
        if path.exists():
            return str(path)

        official_names = {
            "yolov8n-pose.pt",
            "yolov8s-pose.pt",
            "yolov8m-pose.pt",
            "yolov8l-pose.pt",
            "yolov8x-pose.pt",
        }
        if path.name in official_names:
            return path.name

        raise FileNotFoundError(
            f"YOLO pose weights not found: {path}. "
            "Use an official name such as yolov8n-pose.pt or place weights in models/."
        )

    def _extract_pose_observations(self, frame: Any, timestamp: float) -> list[PoseObservation]:
        """Run YOLOv8 pose and convert outputs to PoseObservation objects."""

        if self._model is None:
            raise RuntimeError("YOLO pose model is not loaded.")

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
        if result.boxes is None or result.keypoints is None:
            return []

        boxes_xyxy = self._to_numpy(result.boxes.xyxy)
        box_conf = self._to_numpy(result.boxes.conf)
        keypoint_xy = self._to_numpy(result.keypoints.xy)
        keypoint_conf = self._to_numpy(result.keypoints.conf)

        boxes = [
            BoundingBox(int(x1), int(y1), int(x2), int(y2))
            for x1, y1, x2, y2 in boxes_xyxy.tolist()
        ]
        track_ids = self.track_matcher.assign(boxes, timestamp)

        observations: list[PoseObservation] = []
        for idx, bbox in enumerate(boxes):
            if bbox.width <= 0 or bbox.height <= 0:
                continue
            confidence = float(box_conf[idx])
            if confidence < self.config.min_pose_confidence:
                continue

            keypoints: dict[int, Keypoint] = {}
            for point_idx, (x, y) in enumerate(keypoint_xy[idx].tolist()):
                if keypoint_conf is None:
                    conf = 1.0
                else:
                    conf = float(keypoint_conf[idx][point_idx])
                keypoints[point_idx] = Keypoint(float(x), float(y), conf)

            observations.append(
                PoseObservation(
                    track_id=track_ids[idx],
                    bbox=bbox,
                    confidence=confidence,
                    keypoints=keypoints,
                )
            )

        return observations

    def _analyze_observations(
        self,
        observations: list[PoseObservation],
        timestamp: float,
    ) -> tuple[list[Detection], list[Detection]]:
        pose_findings: dict[str, list[PoseFinding | MotionFinding]] = {
            obs.track_id: self.pose_analyzer.analyze_person(obs) for obs in observations
        }

        interaction_findings = self.pose_analyzer.analyze_interactions(observations)
        for track_id, findings in interaction_findings.items():
            pose_findings.setdefault(track_id, []).extend(findings)

        motion_findings = self.motion_analyzer.update(observations, timestamp)
        for track_id, findings in motion_findings.items():
            pose_findings.setdefault(track_id, []).extend(findings)

        detections: list[Detection] = []
        alerts: list[Detection] = []
        observations_by_id = {obs.track_id: obs for obs in observations}

        # Expire stale cooldown entries
        self._alert_cooldowns = {
            key: t for key, t in self._alert_cooldowns.items()
            if timestamp - t < self._alert_cooldown_sec * 3
        }

        for track_id, findings in pose_findings.items():
            observation = observations_by_id.get(track_id)
            if observation is None:
                continue
            if not findings:
                detections.append(
                    Detection(
                        label="PERSON",
                        confidence=observation.confidence,
                        bbox=observation.bbox,
                        track_id=track_id,
                    )
                )
                continue

            for finding in self._dedupe_findings(findings):
                detection = Detection(
                    label=finding.label,
                    confidence=observation.confidence,
                    bbox=observation.bbox,
                    track_id=track_id,
                    severity=self._severity_for_label(finding.label),
                    threat_score=finding.score,
                    metadata={
                        "camera_id": self.camera_id,
                        "reason": finding.reason,
                    },
                )
                detections.append(detection)

                # Per-person per-label cooldown to suppress repeated alerts
                cooldown_key = (track_id, finding.label)
                last_alert = self._alert_cooldowns.get(cooldown_key, 0.0)
                if timestamp - last_alert >= self._alert_cooldown_sec:
                    alerts.append(detection)
                    self._alert_cooldowns[cooldown_key] = timestamp

        return detections, alerts

    def _draw_observations(
        self,
        annotated: Any,
        observations: list[PoseObservation],
        detections: list[Detection],
    ) -> None:
        labels_by_track: dict[str, list[str]] = {}
        for detection in detections:
            if detection.track_id is None:
                continue
            if detection.label == "PERSON":
                continue
            labels_by_track.setdefault(str(detection.track_id), []).append(detection.label)

        for observation in observations:
            labels = labels_by_track.get(observation.track_id, ["PERSON"])
            label = f"ID {observation.track_id} " + " | ".join(labels[:2])
            draw_labeled_box(annotated, observation.bbox, label, BEHAVIOR_STYLE)

    @staticmethod
    def _dedupe_findings(findings: list[PoseFinding | MotionFinding]) -> list[PoseFinding | MotionFinding]:
        seen: set[str] = set()
        deduped: list[PoseFinding | MotionFinding] = []
        for finding in findings:
            if finding.label in seen:
                continue
            seen.add(finding.label)
            deduped.append(finding)
        return deduped

    @staticmethod
    def _severity_for_label(label: str) -> Severity:
        if label == "FIGHT DETECTED":
            return Severity.HIGH
        if label == "FALL DETECTED":
            return Severity.HIGH
        if label == "RUNNING":
            return Severity.MEDIUM
        return Severity.MEDIUM

    @staticmethod
    def _to_numpy(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return value


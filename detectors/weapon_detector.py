"""Weapon detection using a configured YOLOv8-compatible model.

Important:
    Official YOLOv8 COCO weights do not provide reliable gun or knife classes.
    This detector therefore works best with a custom `.pt` model trained/exported
    for Ultralytics YOLOv8 with classes such as `gun`, `pistol`, `rifle`, and
    `knife`.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from security_ai_system.detectors.base import DetectorMetadata, VisionDetector
from security_ai_system.utils.drawing import WEAPON_STYLE, draw_labeled_box
from security_ai_system.utils.types import (
    BoundingBox,
    Detection,
    DetectorKind,
    DetectorResult,
    ModelPathConfig,
    RuntimeConfig,
    Severity,
)


FIREARM_CANONICAL = "firearm"
KNIFE_CANONICAL = "knife"


DEFAULT_WEAPON_ALIASES: dict[str, str] = {
    "gun": FIREARM_CANONICAL,
    "guns": FIREARM_CANONICAL,
    "pistol": FIREARM_CANONICAL,
    "pistols": FIREARM_CANONICAL,
    "handgun": FIREARM_CANONICAL,
    "rifle": FIREARM_CANONICAL,
    "rifles": FIREARM_CANONICAL,
    "firearm": FIREARM_CANONICAL,
    "knife": KNIFE_CANONICAL,
    "knives": KNIFE_CANONICAL,
    "kitchen knife": KNIFE_CANONICAL,
    "utility knife": KNIFE_CANONICAL,
    "box cutter": KNIFE_CANONICAL,
}


@dataclass(frozen=True)
class WeaponDetectorConfig:
    """Configuration for YOLOv8 weapon detection."""

    model_paths: ModelPathConfig = ModelPathConfig()
    class_aliases: dict[str, str] = field(default_factory=lambda: DEFAULT_WEAPON_ALIASES.copy())
    snapshot_cooldown_sec: float = 10.0
    detection_hold_frames: int = 5
    small_object_tiles: bool = True


class WeaponDetector(VisionDetector):
    """Detect guns/pistols/rifles and knives in OpenCV BGR frames."""

    metadata = DetectorMetadata(
        name="Weapon Detector",
        kind=DetectorKind.WEAPON,
        description="YOLOv8-compatible custom weapon detector.",
    )

    def __init__(
        self,
        camera_id: str,
        runtime: RuntimeConfig | None = None,
        config: WeaponDetectorConfig | None = None,
    ) -> None:
        super().__init__(camera_id=camera_id, runtime=runtime)
        self.config = config or WeaponDetectorConfig()
        self._model: Any | None = None
        self._last_snapshot_at_by_label: dict[str, float] = {}
        self._last_detections: list[Detection] = []
        self._missed_detection_frames = 0

    def load(self) -> None:
        """Load the configured YOLOv8 weapon model."""

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for weapon detection. "
                "Install it with `pip install ultralytics`."
            ) from exc

        weights = self._resolve_weapon_weights()
        self._model = YOLO(str(weights))
        self._loaded = True

    def predict(self, data: Any, timestamp: float | None = None) -> DetectorResult:
        """Run weapon detection on one OpenCV BGR frame."""

        self._require_loaded()
        started = time.perf_counter()
        now = timestamp or time.time()

        if data is None or not hasattr(data, "shape"):
            return self._empty_result(
                detector=DetectorKind.WEAPON,
                timestamp=now,
                error="Weapon detector expected an OpenCV/Numpy image frame.",
            )

        frame = data
        annotated = frame.copy()

        try:
            detections = self._detect_weapons(frame)
            alerts = []
            for detection in detections:
                if detection.bbox is None:
                    continue

                draw_labeled_box(
                    annotated,
                    detection.bbox,
                    f"{detection.label} {detection.confidence:.2f}",
                    WEAPON_STYLE,
                )
                detection.metadata["snapshot_path"] = self._maybe_save_snapshot(
                    annotated,
                    detection.label,
                    now,
                )
                alerts.append(detection)
        except Exception as exc:
            return DetectorResult(
                detector=DetectorKind.WEAPON,
                camera_id=self.camera_id,
                timestamp=now,
                annotated_frame=annotated,
                processing_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
            )

        return DetectorResult(
            detector=DetectorKind.WEAPON,
            camera_id=self.camera_id,
            timestamp=now,
            detections=detections,
            annotated_frame=annotated,
            alerts=alerts,
            processing_ms=(time.perf_counter() - started) * 1000,
        )

    def _resolve_weapon_weights(self) -> Path | str:
        """Return a validated local path to custom weapon weights or an official name for auto-download."""

        path = self.config.model_paths.yolo_weapon_weights
        if path is None:
            raise FileNotFoundError(
                "Weapon detector requires weapon weights. "
                "Set ModelPathConfig(yolo_weapon_weights=Path('models/weapon_yolov8.pt'))."
            )

        resolved = Path(path)
        if resolved.exists():
            if resolved.suffix.lower() != ".pt":
                raise ValueError(
                    f"Expected an Ultralytics YOLOv8 .pt model, got: {resolved.name}"
                )
            return resolved
            
        official_names = {
            "yolov8n.pt",
            "yolov8s.pt",
            "yolov8m.pt",
            "yolov8l.pt",
            "yolov8x.pt",
        }
        if resolved.name in official_names:
            return resolved.name

        raise FileNotFoundError(f"Weapon model weights not found: {resolved}")

    def _detect_weapons(self, frame: Any) -> list[Detection]:
        """Run YOLOv8 inference and convert weapon classes to detections."""

        if self._model is None:
            raise RuntimeError("YOLO weapon model is not loaded.")

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
        detections = self._detections_from_result(result, 0, 0)

        if not detections and self.config.small_object_tiles:
            height, width = frame.shape[:2]
            tile_width = max(1, int(width * 0.6))
            tile_height = max(1, int(height * 0.6))
            step_x = max(1, int(tile_width * 0.65))
            step_y = max(1, int(tile_height * 0.65))
            for top in range(0, max(1, height - tile_height + 1), step_y):
                for left in range(0, max(1, width - tile_width + 1), step_x):
                    tile = frame[top : min(top + tile_height, height), left : min(left + tile_width, width)]
                    tile_result = self._model.predict(tile, **predict_kwargs)[0]
                    detections.extend(self._detections_from_result(tile_result, left, top))
            detections = self._deduplicate_detections(detections)

        return self._stabilize_detections(detections)

    def _detections_from_result(self, result: Any, offset_x: int, offset_y: int) -> list[Detection]:
        """Convert one full-frame or tile inference result to project detections."""

        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            source_label = str(result.names[class_id])
            canonical = self._canonical_weapon_label(source_label)
            if canonical is None:
                continue
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = BoundingBox(
                int(x1) + offset_x,
                int(y1) + offset_y,
                int(x2) + offset_x,
                int(y2) + offset_y,
            )
            if bbox.width <= 0 or bbox.height <= 0:
                continue
            detections.append(
                Detection(
                    label=self._display_label(canonical),
                    confidence=confidence,
                    bbox=bbox,
                    severity=Severity.CRITICAL if canonical == FIREARM_CANONICAL else Severity.HIGH,
                    threat_score=self._threat_score(canonical),
                    metadata={
                        "camera_id": self.camera_id,
                        "source_label": source_label,
                        "canonical_label": canonical,
                        "priority": "high",
                    },
                )
            )
        return detections

    @staticmethod
    def _deduplicate_detections(detections: list[Detection]) -> list[Detection]:
        """Keep the highest-confidence box when overlapping tiles find the same object."""

        selected: list[Detection] = []
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            if detection.bbox is None:
                continue
            if any(
                existing.bbox is not None
                and WeaponDetector._box_iou(detection.bbox, existing.bbox) >= 0.5
                for existing in selected
            ):
                continue
            selected.append(detection)
        return selected

    @staticmethod
    def _box_iou(first: BoundingBox, second: BoundingBox) -> float:
        """Return intersection-over-union for two pixel-space boxes."""

        intersection_width = max(0, min(first.x2, second.x2) - max(first.x1, second.x1))
        intersection_height = max(0, min(first.y2, second.y2) - max(first.y1, second.y1))
        intersection = intersection_width * intersection_height
        union = first.width * first.height + second.width * second.height - intersection
        return intersection / union if union else 0.0

    def _stabilize_detections(self, detections: list[Detection]) -> list[Detection]:
        """Hold a recent detection briefly through transient model misses."""

        if detections:
            self._last_detections = detections
            self._missed_detection_frames = 0
            return detections

        if (
            self._last_detections
            and self._missed_detection_frames < self.config.detection_hold_frames
        ):
            self._missed_detection_frames += 1
            return self._last_detections

        self._last_detections = []
        self._missed_detection_frames = 0
        return detections

    def _canonical_weapon_label(self, label: str) -> str | None:
        normalized = self._normalize_label(label)
        return self.config.class_aliases.get(normalized)

    @staticmethod
    def _normalize_label(label: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", label.strip().lower())
        return re.sub(r"\s+", " ", normalized)

    @staticmethod
    def _display_label(canonical_label: str) -> str:
        if canonical_label == KNIFE_CANONICAL:
            return "KNIFE DETECTED"
        return "GUN DETECTED"

    @staticmethod
    def _threat_score(canonical_label: str) -> int:
        if canonical_label == KNIFE_CANONICAL:
            return 80
        return 100

    def _maybe_save_snapshot(
        self,
        annotated: Any,
        display_label: str,
        timestamp: float,
    ) -> str | None:
        """Save weapon evidence snapshots with per-label cooldown."""

        if not self.runtime.save_snapshots:
            return None

        last_snapshot_at = self._last_snapshot_at_by_label.get(display_label, 0.0)
        if timestamp - last_snapshot_at < self.config.snapshot_cooldown_sec:
            return None

        import cv2

        snapshot_dir = Path(self.runtime.output_dir) / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        safe_camera_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.camera_id)
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", display_label.lower())
        filename = f"{safe_camera_id}_{safe_label}_{int(timestamp)}.jpg"
        path = snapshot_dir / filename

        if cv2.imwrite(str(path), annotated):
            self._last_snapshot_at_by_label[display_label] = timestamp
            return str(path)
        return None

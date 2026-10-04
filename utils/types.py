"""Shared typed data structures used across the surveillance pipeline.

The project intentionally keeps these classes lightweight and dependency-free.
Detector modules can be imported and unit-tested without loading OpenCV,
YOLO, TensorFlow, or a UI framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Severity(str, Enum):
    """Normalized event severity for alerts and dashboard display."""

    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DetectorKind(str, Enum):
    """Detector categories supported by the system."""

    BAG = "bag"
    WEAPON = "weapon"
    BEHAVIOR = "behavior"
    AUDIO = "audio"


@dataclass(frozen=True)
class BoundingBox:
    """Pixel-space bounding box using x1, y1, x2, y2 coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[int, int]:
        return (self.x1 + self.width // 2, self.y1 + self.height // 2)

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass
class Detection:
    """A single model or rule-based detection."""

    label: str
    confidence: float
    bbox: BoundingBox | None = None
    track_id: int | str | None = None
    global_id: str | None = None
    severity: Severity = Severity.INFO
    threat_score: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectorResult:
    """Result returned by any detector for one frame, chunk, or time slice."""

    detector: DetectorKind
    camera_id: str
    timestamp: float
    detections: list[Detection] = field(default_factory=list)
    annotated_frame: Any | None = None
    alerts: list[Detection] = field(default_factory=list)
    processing_ms: float = 0.0
    error: str | None = None

    @property
    def has_alerts(self) -> bool:
        return bool(self.alerts)


@dataclass(frozen=True)
class ModelPathConfig:
    """Model path configuration shared by detector implementations."""

    yolo_object_weights: Path = Path("models/yolov8n.pt")
    yolo_pose_weights: Path = Path("models/yolov8n-pose.pt")
    yolo_weapon_weights: Path | None = None
    yamnet_model_handle: str = "https://tfhub.dev/google/yamnet/1"


@dataclass(frozen=True)
class RuntimeConfig:
    """Common runtime settings used by detectors and camera workers."""

    device: str = "auto"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    inference_size: int = 960
    max_detections: int = 100
    test_time_augmentation: bool = False
    save_snapshots: bool = True
    output_dir: Path = Path("outputs")


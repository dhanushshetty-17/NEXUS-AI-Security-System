from collections import deque
import csv
import json
import logging
import platform
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from security_ai_system.alerts.threat_engine import ThreatScoringEngine, ThreatState
from security_ai_system.cameras.camera_manager import CameraPipelineResult
from security_ai_system.utils.types import Detection, Severity


@dataclass(frozen=True)
class AlertManagerConfig:
    """Configuration for event logging and evidence persistence."""

    output_dir: Path = Path("outputs")
    snapshot_dir_name: str = "snapshots"
    log_dir_name: str = "logs"
    jsonl_name: str = "threat_events.jsonl"
    csv_name: str = "threat_events.csv"
    save_snapshots: bool = True
    event_cooldown_sec: float = 5.0
    alarm_enabled: bool = False


@dataclass(frozen=True)
class AlertEvent:
    """Persisted alert event."""

    event_id: str
    timestamp: float
    iso_time: str
    camera_id: str
    label: str
    severity: Severity
    confidence: float
    threat_score: int
    total_score: int
    threat_level: str
    snapshot_path: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """Create, log, and optionally notify on threat events."""

    def __init__(
        self,
        config: AlertManagerConfig | None = None,
        threat_engine: ThreatScoringEngine | None = None,
        reporter: Any | None = None,
    ) -> None:
        self.config = config or AlertManagerConfig()
        self.threat_engine = threat_engine or ThreatScoringEngine()
        self.output_dir = Path(self.config.output_dir)
        self.snapshot_dir = self.output_dir / self.config.snapshot_dir_name
        self.log_dir = self.output_dir / self.config.log_dir_name
        
        if reporter is not None:
            self.reporter = reporter
        else:
            try:
                from security_ai_system.utils.genai_reporter import GenAIReporter
                self.reporter = GenAIReporter(output_dir=str(self.output_dir / "reports"))
            except Exception:
                self.reporter = None

        self.jsonl_path = self.log_dir / self.config.jsonl_name
        self.csv_path = self.log_dir / self.config.csv_name
        self._last_event_at: dict[tuple[str, str], float] = {}
        self._events: deque[AlertEvent] = deque(maxlen=200)
        self._lock = threading.Lock()
        self._persistence_lock = threading.Lock()
        self._logger = logging.getLogger(__name__)
        self._ensure_dirs()

    @property
    def events(self) -> list[AlertEvent]:
        with self._lock:
            return list(self._events)

    def recent_events(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent events formatted as dicts, latest first."""
        with self._lock:
            items = list(self._events)
        items.reverse()
        return [self._event_to_dict(evt) for evt in items[:limit]]

    def handle_camera_result(self, result: CameraPipelineResult) -> list[AlertEvent]:
        """Persist alerts from one processed camera frame."""

        return self.handle_alerts(
            detections=result.alerts,
            camera_id=result.camera_id,
            timestamp=result.timestamp,
            frame=result.frame,
        )

    def handle_alerts(
        self,
        detections: Iterable[Detection],
        camera_id: str,
        timestamp: float | None = None,
        frame: Any | None = None,
        source: str | None = None,
    ) -> list[AlertEvent]:
        """Score and persist alert detections."""

        now = timestamp or time.time()
        detections_list = list(detections)
        state = self.threat_engine.update_from_detections(
            detections=detections_list,
            camera_id=camera_id,
            timestamp=now,
        )

        events: list[AlertEvent] = []
        for detection in detections_list:
            if self.threat_engine.score_for_detection(detection) <= 0:
                continue
            cooldown_key = (camera_id, detection.label)
            with self._lock:
                last_event_at = self._last_event_at.get(cooldown_key, 0.0)
                if now - last_event_at < self.config.event_cooldown_sec:
                    continue
                self._last_event_at[cooldown_key] = now

            event = self._build_event(
                detection=detection,
                camera_id=camera_id,
                timestamp=now,
                state=state,
                frame=frame,
                source=source,
            )
            try:
                self._write_event(event)
            except OSError:
                self._logger.exception("Failed to persist alert event %s", event.event_id)
                continue
            with self._lock:
                self._events.append(event)
            events.append(event)

        if events and self.config.alarm_enabled:
            self.play_alarm()
        return events

    def current_threat_state(self, timestamp: float | None = None) -> ThreatState:
        """Return the current score and threat level."""

        return self.threat_engine.current_state(timestamp=timestamp)

    def play_alarm(self) -> None:
        """Play a short Windows beep when enabled."""

        if platform.system().lower() != "windows":
            return
        try:
            import winsound

            winsound.Beep(1200, 250)
        except Exception:
            return

    def _build_event(
        self,
        detection: Detection,
        camera_id: str,
        timestamp: float,
        state: ThreatState,
        frame: Any | None,
        source: str | None,
    ) -> AlertEvent:
        score = self.threat_engine.score_for_detection(detection)
        snapshot_path = detection.metadata.get("snapshot_path")
        if snapshot_path is None and self.config.save_snapshots and frame is not None:
            snapshot_path = self._save_snapshot(frame, camera_id, detection.label, timestamp)
            
        if snapshot_path and score >= 30 and self.reporter is not None:
            try:
                self.reporter.generate_report_async(
                    image_path=str(snapshot_path),
                    context={
                        "camera_id": str(detection.metadata.get("camera_id", camera_id)),
                        "label": detection.label,
                        "severity": detection.severity.value,
                    }
                )
            except Exception:
                pass

        return AlertEvent(
            event_id=str(uuid.uuid4()),
            timestamp=timestamp,
            iso_time=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(timestamp)),
            camera_id=str(detection.metadata.get("camera_id", camera_id)),
            label=detection.label,
            severity=detection.severity,
            confidence=float(detection.confidence),
            threat_score=score,
            total_score=state.total_score,
            threat_level=state.level.value,
            snapshot_path=str(snapshot_path) if snapshot_path else None,
            source=source,
            metadata=dict(detection.metadata),
        )

    def _save_snapshot(
        self,
        frame: Any,
        camera_id: str,
        label: str,
        timestamp: float,
    ) -> str | None:
        """Save an evidence image if OpenCV can encode the frame."""

        try:
            import cv2
        except ImportError:
            return None

        safe_camera = self._safe_filename(camera_id)
        safe_label = self._safe_filename(label.lower())
        path = self.snapshot_dir / f"{safe_camera}_{safe_label}_{int(timestamp)}.jpg"
        try:
            if cv2.imwrite(str(path), frame):
                return str(path)
        except Exception:
            return None
        return None

    def _write_event(self, event: AlertEvent) -> None:
        event_dict = self._event_to_dict(event)
        with self._persistence_lock:
            with self.jsonl_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(event_dict, sort_keys=True) + "\n")
                file.flush()
            self._append_csv(event_dict)

    def _append_csv(self, event_dict: dict[str, Any]) -> None:
        fieldnames = [
            "event_id",
            "iso_time",
            "camera_id",
            "label",
            "severity",
            "confidence",
            "threat_score",
            "total_score",
            "threat_level",
            "snapshot_path",
            "source",
        ]
        exists = self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow({key: event_dict.get(key) for key in fieldnames})

    def _ensure_dirs(self) -> None:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _event_to_dict(event: AlertEvent) -> dict[str, Any]:
        data = asdict(event)
        data["severity"] = event.severity.value
        return data

    @staticmethod
    def _safe_filename(value: str) -> str:
        return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)

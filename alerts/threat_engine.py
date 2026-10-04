"""Centralized threat scoring engine."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from security_ai_system.utils.types import Detection, Severity


class ThreatLevel(str, Enum):
    """Dashboard threat level."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


DEFAULT_THREAT_SCORES: dict[str, int] = {
    "SUSPICIOUS BAG": 40,
    "RUNNING": 20,
    "SUSPICIOUS RUNNING": 20,
    "FIGHT DETECTED": 50,
    "FALL DETECTED": 50,
    "SUSPICIOUS MOVEMENT": 20,
    "KNIFE DETECTED": 80,
    "GUN DETECTED": 100,
    "SCREAM DETECTED": 60,
    "GUNSHOT DETECTED": 120,
    "GLASS BREAK DETECTED": 70,
    "EXPLOSION DETECTED": 100,
}


@dataclass(frozen=True)
class ThreatContribution:
    """One alert's contribution to the active threat score."""

    label: str
    score: int
    camera_id: str
    timestamp: float
    severity: Severity
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ThreatState:
    """Current aggregate threat state."""

    total_score: int
    level: ThreatLevel
    contributions: list[ThreatContribution]
    timestamp: float


class ThreatScoringEngine:
    """Aggregate active detector alerts into one threat score and level."""

    def __init__(
        self,
        score_map: dict[str, int] | None = None,
        active_window_sec: float = 10.0,
    ) -> None:
        self.score_map = score_map or DEFAULT_THREAT_SCORES.copy()
        self.active_window_sec = active_window_sec
        self._active: list[ThreatContribution] = []
        self._lock = threading.Lock()

    def update_from_detections(
        self,
        detections: Iterable[Detection],
        camera_id: str,
        timestamp: float | None = None,
    ) -> ThreatState:
        """Add alerts/detections to the active scoring window in a thread-safe manner."""

        now = timestamp or time.time()
        new_items: list[ThreatContribution] = []
        for detection in detections:
            score = self.score_for_detection(detection)
            if score <= 0:
                continue
            new_items.append(
                ThreatContribution(
                    label=detection.label,
                    score=score,
                    camera_id=str(detection.metadata.get("camera_id", camera_id)),
                    timestamp=now,
                    severity=detection.severity,
                    confidence=float(detection.confidence),
                    metadata=dict(detection.metadata),
                )
            )

        with self._lock:
            self._expire_old(now)
            self._active.extend(new_items)
            total = sum(item.score for item in self._active)
            return ThreatState(
                total_score=total,
                level=self.level_for_score(total),
                contributions=list(self._active),
                timestamp=now,
            )

    def current_state(self, timestamp: float | None = None) -> ThreatState:
        """Return the current aggregate state after expiring old events in a thread-safe manner."""

        now = timestamp or time.time()
        with self._lock:
            self._expire_old(now)
            total = sum(item.score for item in self._active)
            return ThreatState(
                total_score=total,
                level=self.level_for_score(total),
                contributions=list(self._active),
                timestamp=now,
            )

    def reset(self) -> None:
        """Clear all active contributions."""

        with self._lock:
            self._active.clear()

    def score_for_detection(self, detection: Detection) -> int:
        """Return detector-provided threat score or configured fallback score."""

        if detection.threat_score > 0:
            return int(detection.threat_score)
        return int(self.score_map.get(detection.label.upper(), 0))

    def _expire_old(self, timestamp: float) -> None:
        cutoff = timestamp - self.active_window_sec
        self._active = [item for item in self._active if item.timestamp >= cutoff]

    @staticmethod
    def level_for_score(score: int) -> ThreatLevel:
        """Map score to LOW/MEDIUM/HIGH/CRITICAL.

        Scores greater than 100 are critical, matching the project requirement.
        """

        if score > 100:
            return ThreatLevel.CRITICAL
        if score >= 80:
            return ThreatLevel.HIGH
        if score >= 40:
            return ThreatLevel.MEDIUM
        return ThreatLevel.LOW


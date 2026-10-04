"""PyQt5 dashboard for the AI surveillance system.

The helper functions in this module are dependency-light and unit-testable. The
actual dashboard imports PyQt5 lazily so non-GUI test environments can still
import the module.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from security_ai_system.alerts import AlertEvent, AlertManager, ThreatLevel
from security_ai_system.cameras import CameraManager, CameraPipelineResult, CameraStatus


class DashboardUnavailableError(RuntimeError):
    """Raised when the dashboard is started without PyQt5 installed."""


@dataclass(frozen=True)
class GridPosition:
    """Row/column placement for one camera tile."""

    index: int
    row: int
    column: int


def compute_grid_positions(count: int) -> list[GridPosition]:
    """Return a balanced grid layout for `count` camera feeds."""

    if count <= 0:
        return []
    columns = math.ceil(math.sqrt(count))
    return [
        GridPosition(index=idx, row=idx // columns, column=idx % columns)
        for idx in range(count)
    ]


def format_fps(fps: float) -> str:
    """Format an FPS value for compact dashboard display."""

    if fps <= 0:
        return "0.0 FPS"
    return f"{fps:.1f} FPS"


def threat_level_color(level: str | ThreatLevel) -> str:
    """Return a CSS color for a threat level."""

    value = level.value if isinstance(level, ThreatLevel) else str(level)
    colors = {
        "LOW": "#2e7d32",
        "MEDIUM": "#a66f00",
        "HIGH": "#b3261e",
        "CRITICAL": "#7f0000",
    }
    return colors.get(value.upper(), "#4b5563")


def event_row(event: AlertEvent) -> tuple[str, str, str, str, str]:
    """Return table-safe strings for one alert event."""

    return (
        event.iso_time,
        event.camera_id,
        event.label,
        event.threat_level,
        str(event.threat_score),
    )


def _import_qt() -> tuple[Any, Any, Any]:
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
    except ImportError as exc:
        raise DashboardUnavailableError(
            "PyQt5 is required for the dashboard. Install it with `pip install PyQt5`."
        ) from exc
    return QtCore, QtGui, QtWidgets


def bgr_frame_to_pixmap(frame: Any, target_size: Any | None = None) -> Any:
    """Convert an OpenCV BGR frame to a QPixmap."""

    QtCore, QtGui, _ = _import_qt()
    import cv2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    bytes_per_line = channels * width
    image = QtGui.QImage(
        rgb.data,
        width,
        height,
        bytes_per_line,
        QtGui.QImage.Format_RGB888,
    ).copy()
    pixmap = QtGui.QPixmap.fromImage(image)
    if target_size is not None and target_size.width() > 0 and target_size.height() > 0:
        pixmap = pixmap.scaled(
            target_size,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
    return pixmap


def create_dashboard_window(
    camera_manager: CameraManager,
    alert_manager: AlertManager,
    refresh_interval_ms: int = 100,
) -> Any:
    """Create and return the PyQt5 dashboard window."""

    QtCore, QtGui, QtWidgets = _import_qt()

    class CameraFeedWidget(QtWidgets.QFrame):
        def __init__(self, camera_id: str, parent: Any | None = None) -> None:
            super().__init__(parent)
            self.camera_id = camera_id
            self.setObjectName("cameraTile")
            self.setMinimumSize(320, 220)

            self.title = QtWidgets.QLabel(camera_id)
            self.title.setObjectName("cameraTitle")
            self.status = QtWidgets.QLabel("OFFLINE")
            self.status.setObjectName("cameraStatus")
            self.fps = QtWidgets.QLabel("0.0 FPS")
            self.fps.setObjectName("cameraFps")
            self.image = QtWidgets.QLabel()
            self.image.setObjectName("cameraImage")
            self.image.setAlignment(QtCore.Qt.AlignCenter)
            self.image.setMinimumSize(280, 160)
            self.image.setText("NO SIGNAL")
            self.alert = QtWidgets.QLabel("")
            self.alert.setObjectName("cameraAlert")

            top = QtWidgets.QHBoxLayout()
            top.addWidget(self.title, 1)
            top.addWidget(self.status)
            top.addWidget(self.fps)

            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(8)
            layout.addLayout(top)
            layout.addWidget(self.image, 1)
            layout.addWidget(self.alert)

        def update_status(self, status: CameraStatus | None) -> None:
            if status is None:
                self.status.setText("OFFLINE")
                self.fps.setText("0.0 FPS")
                return
            self.status.setText("LIVE" if status.connected else "OFFLINE")
            self.fps.setText(format_fps(status.fps))
            if status.last_error:
                self.alert.setText(status.last_error[:120])

        def update_result(self, result: CameraPipelineResult | None) -> None:
            if result is None:
                return
            if result.frame is not None and hasattr(result.frame, "shape"):
                pixmap = bgr_frame_to_pixmap(result.frame, self.image.size())
                self.image.setPixmap(pixmap)
            self.fps.setText(format_fps(result.fps))
            if result.alerts:
                labels = ", ".join(alert.label for alert in result.alerts[:2])
                self.alert.setText(labels)
            elif result.error:
                self.alert.setText(result.error[:120])
            else:
                self.alert.setText("")

    class ThreatPanel(QtWidgets.QFrame):
        def __init__(self, parent: Any | None = None) -> None:
            super().__init__(parent)
            self.setObjectName("sidePanel")
            self.score = QtWidgets.QLabel("0")
            self.score.setObjectName("scoreValue")
            self.level = QtWidgets.QLabel("LOW")
            self.level.setObjectName("levelValue")
            self.active = QtWidgets.QLabel("0 Active Alerts")
            self.active.setObjectName("activeAlerts")
            self.events = QtWidgets.QTableWidget(0, 5)
            self.events.setObjectName("eventTable")
            self.events.setHorizontalHeaderLabels(["Time", "Camera", "Alert", "Level", "Score"])
            self.events.horizontalHeader().setStretchLastSection(True)
            self.events.verticalHeader().setVisible(False)
            self.events.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            self.events.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            self.events.setAlternatingRowColors(True)

            header = QtWidgets.QLabel("Threat Monitor")
            header.setObjectName("panelTitle")

            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(14, 14, 14, 14)
            layout.setSpacing(10)
            layout.addWidget(header)
            layout.addWidget(self.level)
            layout.addWidget(self.score)
            layout.addWidget(self.active)
            layout.addWidget(self.events, 1)

        def update_state(self, score: int, level: str, active_alerts: int) -> None:
            self.score.setText(str(score))
            self.level.setText(level)
            self.level.setStyleSheet(f"background: {threat_level_color(level)};")
            self.active.setText(f"{active_alerts} Active Alerts")

        def append_events(self, events: list[AlertEvent]) -> None:
            for event in events:
                row = self.events.rowCount()
                self.events.insertRow(row)
                for col, value in enumerate(event_row(event)):
                    item = QtWidgets.QTableWidgetItem(value)
                    if col == 3:
                        item.setForeground(QtGui.QBrush(QtGui.QColor(threat_level_color(value))))
                    self.events.setItem(row, col, item)
            if events:
                self.events.scrollToBottom()
            while self.events.rowCount() > 250:
                self.events.removeRow(0)

    class SurveillanceDashboard(QtWidgets.QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.camera_manager = camera_manager
            self.alert_manager = alert_manager
            self._processed_frames: set[tuple[str, int]] = set()
            self._feed_widgets: dict[str, CameraFeedWidget] = {}

            self.setWindowTitle("AI Smart Surveillance")
            self.resize(1280, 780)

            central = QtWidgets.QWidget()
            self.setCentralWidget(central)

            self.grid = QtWidgets.QGridLayout()
            self.grid.setContentsMargins(12, 12, 12, 12)
            self.grid.setSpacing(10)

            self.panel = ThreatPanel()
            self.start_button = QtWidgets.QPushButton("Start")
            self.stop_button = QtWidgets.QPushButton("Stop")
            self.clear_button = QtWidgets.QPushButton("Clear")
            self.start_button.clicked.connect(self.start_cameras)
            self.stop_button.clicked.connect(self.stop_cameras)
            self.clear_button.clicked.connect(self.clear_events)

            controls = QtWidgets.QHBoxLayout()
            controls.addWidget(self.start_button)
            controls.addWidget(self.stop_button)
            controls.addWidget(self.clear_button)

            right = QtWidgets.QVBoxLayout()
            right.addWidget(self.panel, 1)
            right.addLayout(controls)

            root = QtWidgets.QHBoxLayout(central)
            root.setContentsMargins(0, 0, 0, 0)
            root.addLayout(self.grid, 3)
            root.addLayout(right, 1)

            self._build_camera_grid()
            self._apply_styles()

            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.refresh)
            self.timer.start(refresh_interval_ms)

        def start_cameras(self) -> None:
            self.camera_manager.start_all()

        def stop_cameras(self) -> None:
            self.camera_manager.stop_all()

        def clear_events(self) -> None:
            self.panel.events.setRowCount(0)

        def refresh(self) -> None:
            statuses = {status.camera_id: status for status in self.camera_manager.statuses()}
            for camera_id, widget in self._feed_widgets.items():
                widget.update_status(statuses.get(camera_id))

            new_events: list[AlertEvent] = []
            for camera_id, result in self.camera_manager.latest_results().items():
                if camera_id not in self._feed_widgets:
                    self._add_feed_widget(camera_id)
                self._feed_widgets[camera_id].update_result(result)
                key = (camera_id, result.frame_index)
                if key in self._processed_frames:
                    continue
                self._processed_frames.add(key)
                new_events.extend(self.alert_manager.handle_camera_result(result))

            self.panel.append_events(new_events)
            state = self.alert_manager.current_threat_state()
            self.panel.update_state(
                score=state.total_score,
                level=state.level.value,
                active_alerts=len(state.contributions),
            )

        def closeEvent(self, event: Any) -> None:
            self.camera_manager.stop_all()
            event.accept()

        def _build_camera_grid(self) -> None:
            for worker in self.camera_manager.workers():
                self._add_feed_widget(worker.config.camera_id)
            if not self._feed_widgets:
                self._add_feed_widget("camera-preview")

        def _add_feed_widget(self, camera_id: str) -> None:
            if camera_id in self._feed_widgets:
                return
            widget = CameraFeedWidget(camera_id)
            self._feed_widgets[camera_id] = widget
            self._reflow_grid()

        def _reflow_grid(self) -> None:
            while self.grid.count():
                item = self.grid.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
            widgets = list(self._feed_widgets.values())
            for position in compute_grid_positions(len(widgets)):
                self.grid.addWidget(widgets[position.index], position.row, position.column)
            for col in range(max(1, math.ceil(math.sqrt(len(widgets))))):
                self.grid.setColumnStretch(col, 1)

        def _apply_styles(self) -> None:
            self.setStyleSheet(
                """
                QMainWindow { background: #f4f6f8; }
                #cameraTile, #sidePanel {
                    background: #ffffff;
                    border: 1px solid #d7dde4;
                    border-radius: 6px;
                }
                #cameraTitle {
                    color: #111827;
                    font-size: 14px;
                    font-weight: 700;
                }
                #cameraStatus, #cameraFps {
                    color: #374151;
                    font-size: 12px;
                    padding: 3px 6px;
                    border: 1px solid #d7dde4;
                    border-radius: 4px;
                }
                #cameraImage {
                    background: #111827;
                    color: #9ca3af;
                    border-radius: 4px;
                    font-size: 13px;
                    font-weight: 600;
                }
                #cameraAlert {
                    color: #b3261e;
                    font-size: 12px;
                    min-height: 18px;
                }
                #panelTitle {
                    color: #111827;
                    font-size: 18px;
                    font-weight: 800;
                }
                #levelValue {
                    color: #ffffff;
                    font-size: 18px;
                    font-weight: 800;
                    padding: 8px;
                    border-radius: 4px;
                }
                #scoreValue {
                    color: #111827;
                    font-size: 42px;
                    font-weight: 900;
                }
                #activeAlerts {
                    color: #374151;
                    font-size: 13px;
                    font-weight: 600;
                }
                QTableWidget {
                    background: #ffffff;
                    color: #111827;
                    border: 1px solid #d7dde4;
                    gridline-color: #e5e7eb;
                    alternate-background-color: #f8fafc;
                }
                QHeaderView::section {
                    background: #edf1f5;
                    color: #111827;
                    padding: 5px;
                    border: 0;
                    font-weight: 700;
                }
                QPushButton {
                    background: #263238;
                    color: white;
                    border: 0;
                    padding: 8px 12px;
                    border-radius: 4px;
                    font-weight: 700;
                }
                QPushButton:hover { background: #37474f; }
                """
            )

    return SurveillanceDashboard()


def run_dashboard(
    camera_manager: CameraManager,
    alert_manager: AlertManager | None = None,
    start_cameras: bool = True,
) -> int:
    """Run the PyQt5 dashboard event loop."""

    _, _, QtWidgets = _import_qt()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = create_dashboard_window(
        camera_manager=camera_manager,
        alert_manager=alert_manager or AlertManager(),
    )
    window.show()
    if start_cameras:
        camera_manager.start_all()
    return app.exec_()


def build_basic_camera_manager(sources: list[str]) -> CameraManager:
    """Build a camera manager populated with vision detectors and ReID."""

    from security_ai_system.cameras import CameraSourceConfig, infer_source_type, CameraSourceType
    from security_ai_system.detectors.bag_detector import BagDetector, BagDetectorConfig
    from security_ai_system.detectors.weapon_detector import WeaponDetector, WeaponDetectorConfig
    from security_ai_system.detectors.behavior_detector import BehaviorDetector, BehaviorDetectorConfig
    from security_ai_system.trackers.tracker import DeepSortTracker
    from security_ai_system.utils.reid_manager import GlobalReIDManager
    from security_ai_system.utils.types import ModelPathConfig
    from pathlib import Path

    manager = CameraManager()
    project_root = Path(__file__).resolve().parents[1]
    if not sources:
        sources = ["0"]
        
    reid_manager = GlobalReIDManager(similarity_threshold=0.75)
    
    for idx, source in enumerate(sources, start=1):
        parsed_source: int | str = int(source) if str(source).isdigit() else source
        camera_id = f"camera-{idx}"
        
        source_type = infer_source_type(parsed_source)
        
        # Share the ReID manager across all camera trackers
        tracker = DeepSortTracker(reid_manager=reid_manager)
        bag_detector = BagDetector(
            camera_id=camera_id,
            tracker=tracker,
            config=BagDetectorConfig(
                model_paths=ModelPathConfig(
                    yolo_object_weights=project_root / "yolov8m.pt",
                )
            ),
        )
        
        # Add WeaponDetector using the standard YOLOv8m model as a fallback for knife detection
        configured_weapon_path = os.getenv("WEAPON_MODEL_PATH", "").strip()
        weapon_weights = (
            Path(configured_weapon_path)
            if configured_weapon_path
            else project_root / "models" / "weapon_yolov8.pt"
        )
        if not weapon_weights.is_absolute():
            weapon_weights = project_root / weapon_weights
        if not weapon_weights.exists():
            # The COCO checkpoint contains a knife class, but no firearm classes.
            # It is a useful fallback for knife detection only.
            weapon_weights = project_root / "yolov8m.pt"
        weapon_detector_config = WeaponDetectorConfig(
            model_paths=ModelPathConfig(yolo_weapon_weights=weapon_weights)
        )
        detectors = [bag_detector]
        if weapon_weights.exists():
            detectors.append(
                WeaponDetector(camera_id=camera_id, config=weapon_detector_config)
            )
        
        if source_type != CameraSourceType.VIDEO_FILE:
            behavior_config = BehaviorDetectorConfig(
                model_paths=ModelPathConfig(
                    yolo_pose_weights=project_root / "yolov8m-pose.pt"
                )
            )
            behavior_detector = BehaviorDetector(camera_id=camera_id, config=behavior_config)
            detectors.append(behavior_detector)
        
        worker = manager.add_camera(
            CameraSourceConfig(
                camera_id=camera_id,
                source=parsed_source,
                source_type=source_type,
                display_name=f"Camera {idx}",
                target_fps=20.0,
            ),
            detectors=detectors
        )
        worker._all_detectors = detectors
    return manager


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for dashboard-only smoke demos."""

    import argparse

    parser = argparse.ArgumentParser(description="Run the surveillance dashboard.")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Camera source. Use multiple --source values for a grid.",
    )
    args = parser.parse_args(argv)
    manager = build_basic_camera_manager(args.source)
    return run_dashboard(manager)


if __name__ == "__main__":
    raise SystemExit(main())

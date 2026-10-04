# Module 7: PyQt5 Dashboard

## What It Shows

`ui/dashboard.py` provides a production-style operator dashboard with:

- multi-camera grid;
- live annotated camera frames;
- FPS per camera;
- camera connection status;
- active camera alerts;
- total threat score;
- LOW / MEDIUM / HIGH / CRITICAL threat level;
- event history table;
- Start / Stop / Clear controls.

The dashboard consumes `CameraManager.latest_results()` and sends frame alerts to
`AlertManager`, so event logging and threat scoring continue to use the central
alert layer.

## Quick Webcam Demo

```powershell
cd security_ai_system
python main.py --dashboard --source 0
```

## Multiple Source Demo

```powershell
python main.py --dashboard --source 0 --source demo_video.mp4 --source rtsp://user:pass@camera/live
```

This launcher starts camera feeds without AI detectors. It is useful for testing
the grid, FPS, status, and event panel wiring.

## Full Detector Dashboard Script

Create a script such as `run_security_dashboard.py`:

```python
from pathlib import Path

from security_ai_system.alerts import AlertManager
from security_ai_system.cameras import CameraManager, CameraSourceConfig, CameraSourceType
from security_ai_system.detectors.bag_detector import SuspiciousBagDetector
from security_ai_system.detectors.behavior_detector import BehaviorDetector
from security_ai_system.detectors.weapon_detector import WeaponDetector, WeaponDetectorConfig
from security_ai_system.ui.dashboard import run_dashboard
from security_ai_system.utils.types import ModelPathConfig

manager = CameraManager()

manager.add_camera(
    CameraSourceConfig("entrance", 0, CameraSourceType.WEBCAM),
    detectors=[SuspiciousBagDetector(camera_id="entrance")],
)

manager.add_camera(
    CameraSourceConfig("lobby", "demo_lobby.mp4", CameraSourceType.VIDEO_FILE),
    detectors=[BehaviorDetector(camera_id="lobby")],
)

manager.add_camera(
    CameraSourceConfig("checkpoint", "demo_weapon.mp4", CameraSourceType.VIDEO_FILE),
    detectors=[
        WeaponDetector(
            camera_id="checkpoint",
            config=WeaponDetectorConfig(
                model_paths=ModelPathConfig(
                    yolo_weapon_weights=Path("models/weapon_yolov8.pt")
                )
            ),
        )
    ],
)

run_dashboard(manager, AlertManager())
```

Run:

```powershell
python run_security_dashboard.py
```

## Presentation Flow

1. Start with `python main.py --dashboard --source 0` to show the live grid.
2. Switch to the full detector script for annotated feeds.
3. Use short prerecorded clips for repeatability.
4. Open `outputs/logs/threat_events.csv` after the demo to show event history.
5. Open `outputs/snapshots/` to show evidence images.

## Unit Tests

The helper tests do not require PyQt5:

```powershell
cd security_ai_system
python -m unittest tests.test_dashboard_helpers
```

## Troubleshooting

- If the dashboard cannot import PyQt5, run `pip install PyQt5`.
- If a webcam tile stays offline, try `--source 1`.
- If RTSP is offline, validate the stream in VLC first.
- If the UI is slow, set `target_fps` lower in `CameraSourceConfig` or use
  smaller YOLO weights.
- If no events appear, confirm detectors are attached to the camera manager.

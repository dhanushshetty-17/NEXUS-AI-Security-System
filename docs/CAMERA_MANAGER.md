# Module 5: Multi-Camera Support

## What It Does

`cameras/camera_manager.py` manages multiple visual sources concurrently:

- webcam indexes such as `0` or `1`;
- prerecorded video files such as `demo.mp4`;
- RTSP CCTV streams such as `rtsp://user:pass@host/stream`;
- HTTP/HTTPS streams supported by OpenCV.

Each camera has its own worker thread and optional detector pipeline. A slow
camera or detector does not block other camera sources.

## Core Classes

- `CameraSourceConfig`: one source definition.
- `CameraWorker`: one threaded source plus detector pipeline.
- `CameraManager`: registers, starts, stops, and queries multiple workers.
- `FramePacket`: raw frame output.
- `CameraPipelineResult`: processed frame plus detector results and alerts.
- `CameraStatus`: current running/connected/FPS status.

## Webcam Example

```python
from security_ai_system.cameras import CameraManager, CameraSourceConfig

manager = CameraManager()
manager.add_camera(CameraSourceConfig(camera_id="webcam-0", source=0))
manager.start_all(load_detectors=False)

worker = manager.get_worker("webcam-0")
result = worker.read_result(timeout=2.0)
print(result.camera_id, result.fps)

manager.stop_all()
```

## Multiple Cameras with Different Pipelines

```python
from pathlib import Path

from security_ai_system.cameras import CameraManager, CameraSourceConfig, CameraSourceType
from security_ai_system.detectors.bag_detector import SuspiciousBagDetector
from security_ai_system.detectors.behavior_detector import BehaviorDetector
from security_ai_system.detectors.weapon_detector import WeaponDetector, WeaponDetectorConfig
from security_ai_system.utils.types import ModelPathConfig

manager = CameraManager()

manager.add_camera(
    CameraSourceConfig("entrance", 0, CameraSourceType.WEBCAM),
    detectors=[SuspiciousBagDetector(camera_id="entrance")],
)

manager.add_camera(
    CameraSourceConfig("parking", "parking_demo.mp4", CameraSourceType.VIDEO_FILE),
    detectors=[BehaviorDetector(camera_id="parking")],
)

manager.add_camera(
    CameraSourceConfig("security-desk", "rtsp://user:pass@host/live", CameraSourceType.RTSP),
    detectors=[
        WeaponDetector(
            camera_id="security-desk",
            config=WeaponDetectorConfig(
                model_paths=ModelPathConfig(
                    yolo_weapon_weights=Path("models/weapon_yolov8.pt")
                )
            ),
        )
    ],
)

manager.start_all()

try:
    while True:
        for camera_id, result in manager.latest_results().items():
            print(camera_id, result.fps, len(result.alerts))
finally:
    manager.stop_all()
```

## Source Type Inference

Use `infer_source_type()` for simple CLI parsing:

```python
from security_ai_system.cameras import CameraSourceConfig, infer_source_type

source = "rtsp://192.168.1.50/live"
config = CameraSourceConfig(
    camera_id="cam-1",
    source=source,
    source_type=infer_source_type(source),
)
```

## Runtime Behavior

- Queues are bounded; when a queue is full, the oldest item is dropped and the
  latest frame/result is kept.
- Video files can loop with `loop_video=True`.
- RTSP/HTTP sources try to reconnect after repeated read failures.
- `target_fps` can be set to reduce CPU usage.
- `latest_results()` is intended for dashboards that render the newest frame per
  camera.

## Unit Tests

The tests use fake readers and fake detectors, so they do not require webcams,
video files, OpenCV, or AI models:

```powershell
cd security_ai_system
python -m unittest tests.test_camera_manager
```

## Troubleshooting

- If a webcam does not open, try source `1` instead of `0`.
- If RTSP fails, test the exact URL in VLC first.
- If video files do not play, confirm OpenCV can read the codec.
- If CPU usage is high, set `target_fps`, use smaller YOLO weights, or assign
  fewer detectors to each camera.

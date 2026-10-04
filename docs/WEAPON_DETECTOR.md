# Module 2: Weapon Detection

## What It Does

`detectors/weapon_detector.py` detects weapons using a configured
YOLOv8-compatible `.pt` model.

Supported model class names include:

- `gun`
- `pistol`
- `rifle`
- `knife`

The detector maps these classes to required alert labels:

- `gun`, `pistol`, `rifle` -> `GUN DETECTED`
- `knife` -> `KNIFE DETECTED`

## Threat Scores

- Gun/pistol/rifle: `100`
- Knife: `80`

Gun-class detections are marked `CRITICAL`; knife detections are marked `HIGH`.
All detections are returned as alerts because weapons are high-priority events.

## Important Model Requirement

Official YOLOv8 COCO weights provide a `knife` class but do not provide reliable
gun detection or small box-knife detection. For production accuracy, use custom
YOLOv8 weapon weights, for example:

```text
models/weapon_yolov8.pt
```

The project does not ship a weapon model because model licensing and class names
vary by dataset. Use a real YOLOv8-compatible `.pt` file trained for these
classes.

## Configuration Example

```python
from pathlib import Path

from security_ai_system.detectors.weapon_detector import (
    WeaponDetector,
    WeaponDetectorConfig,
)
from security_ai_system.utils.types import ModelPathConfig, RuntimeConfig

model_paths = ModelPathConfig(
    yolo_weapon_weights=Path("models/weapon_yolov8.pt")
)
config = WeaponDetectorConfig(model_paths=model_paths)
runtime = RuntimeConfig(confidence_threshold=0.35, device="auto")

detector = WeaponDetector(
    camera_id="camera-weapon-zone",
    runtime=runtime,
    config=config,
)
detector.load()
```

## Webcam Demo Snippet

```python
import cv2
from pathlib import Path

from security_ai_system.detectors.weapon_detector import (
    WeaponDetector,
    WeaponDetectorConfig,
)
from security_ai_system.utils.types import ModelPathConfig

detector = WeaponDetector(
    camera_id="webcam-0",
    config=WeaponDetectorConfig(
        model_paths=ModelPathConfig(
            yolo_weapon_weights=Path("models/weapon_yolov8.pt")
        )
    ),
)
detector.load()

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break

    result = detector.predict(frame)
    cv2.imshow("Weapon Detector", result.annotated_frame)

    for alert in result.alerts:
        print(alert.label, alert.confidence, alert.threat_score)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

## Unit Tests

These tests do not require a weapon model. They validate class mapping, scoring,
and weight-path validation:

```powershell
cd security_ai_system
python -m unittest tests.test_weapon_detector
```

## Troubleshooting

- `Weapon detector requires YOLOv8 weights`: set
  `ModelPathConfig(yolo_weapon_weights=Path("models/weapon_yolov8.pt"))`.
- `Weapon model weights not found`: confirm the file path exists from the current
  working directory.
- `Expected an Ultralytics YOLOv8 .pt model`: use a `.pt` model compatible with
  Ultralytics YOLOv8.
- No detections appear: inspect your model class names. If they differ, pass a
  custom `class_aliases` mapping in `WeaponDetectorConfig`.

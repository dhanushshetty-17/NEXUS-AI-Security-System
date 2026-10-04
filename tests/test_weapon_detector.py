"""Unit tests for weapon detector configuration and label mapping."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from security_ai_system.detectors.weapon_detector import (
    FIREARM_CANONICAL,
    KNIFE_CANONICAL,
    WeaponDetector,
    WeaponDetectorConfig,
)
from security_ai_system.utils.types import ModelPathConfig


class WeaponDetectorTests(unittest.TestCase):
    def test_weapon_aliases_map_to_required_labels_and_scores(self) -> None:
        detector = WeaponDetector(camera_id="cam-test")

        self.assertEqual(detector._canonical_weapon_label("gun"), FIREARM_CANONICAL)
        self.assertEqual(detector._canonical_weapon_label("Pistol"), FIREARM_CANONICAL)
        self.assertEqual(detector._canonical_weapon_label("rifle"), FIREARM_CANONICAL)
        self.assertEqual(detector._canonical_weapon_label("knife"), KNIFE_CANONICAL)
        self.assertEqual(detector._canonical_weapon_label("Kitchen-Knife"), KNIFE_CANONICAL)

        self.assertEqual(detector._display_label(FIREARM_CANONICAL), "GUN DETECTED")
        self.assertEqual(detector._display_label(KNIFE_CANONICAL), "KNIFE DETECTED")
        self.assertEqual(detector._threat_score(FIREARM_CANONICAL), 100)
        self.assertEqual(detector._threat_score(KNIFE_CANONICAL), 80)

    def test_unknown_model_class_is_ignored(self) -> None:
        detector = WeaponDetector(camera_id="cam-test")
        self.assertIsNone(detector._canonical_weapon_label("cell phone"))

    def test_transient_misses_hold_recent_detection(self) -> None:
        detector = WeaponDetector(camera_id="cam-test")
        detection = object()
        detector._last_detections = [detection]  # type: ignore[list-item]

        for _ in range(5):
            self.assertEqual(detector._stabilize_detections([]), [detection])

        self.assertEqual(detector._stabilize_detections([]), [])

    def test_box_iou(self) -> None:
        from security_ai_system.utils.types import BoundingBox

        self.assertAlmostEqual(
            WeaponDetector._box_iou(
                BoundingBox(0, 0, 10, 10),
                BoundingBox(0, 0, 10, 10),
            ),
            1.0,
        )

    def test_missing_weapon_weights_raise_clear_error(self) -> None:
        detector = WeaponDetector(camera_id="cam-test")
        with self.assertRaises(FileNotFoundError):
            detector._resolve_weapon_weights()

    def test_non_pt_weapon_weights_are_rejected(self) -> None:
        fake_weights = Path("models/weapon.onnx")
        config = WeaponDetectorConfig(
            model_paths=ModelPathConfig(yolo_weapon_weights=fake_weights)
        )
        detector = WeaponDetector(camera_id="cam-test", config=config)

        with patch.object(Path, "exists", return_value=True):
            with self.assertRaises(ValueError):
                detector._resolve_weapon_weights()

    def test_existing_pt_weapon_weights_are_accepted(self) -> None:
        fake_weights = Path("models/weapon_yolov8.pt")
        config = WeaponDetectorConfig(
            model_paths=ModelPathConfig(yolo_weapon_weights=fake_weights)
        )
        detector = WeaponDetector(camera_id="cam-test", config=config)

        with patch.object(Path, "exists", return_value=True):
            self.assertEqual(detector._resolve_weapon_weights(), fake_weights)


if __name__ == "__main__":
    unittest.main()

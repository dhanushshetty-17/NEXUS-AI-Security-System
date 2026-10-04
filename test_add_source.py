import os
import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from security_ai_system.cameras.camera_manager import CameraManager, CameraSourceConfig, infer_source_type
from security_ai_system.detectors.bag_detector import BagDetector
from security_ai_system.detectors.weapon_detector import WeaponDetector, WeaponDetectorConfig
from security_ai_system.detectors.behavior_detector import BehaviorDetector, BehaviorDetectorConfig
from security_ai_system.trackers.tracker import DeepSortTracker
from security_ai_system.utils.reid_manager import GlobalReIDManager
from security_ai_system.utils.types import ModelPathConfig

def test_add_source():
    manager = CameraManager()
    source = "video.mp4"
    parsed_source = int(source) if str(source).isdigit() else source
    
    cam_count = 1
    camera_id = f"camera-{cam_count + 1}"
    
    reid_manager = GlobalReIDManager(similarity_threshold=0.75)
    tracker = DeepSortTracker(reid_manager=reid_manager)
    bag_detector = BagDetector(camera_id=camera_id, tracker=tracker)
    
    behavior_config = BehaviorDetectorConfig(
        model_paths=ModelPathConfig(yolo_pose_weights=Path("models/yolov8m-pose.pt"))
    )
    behavior_detector = BehaviorDetector(camera_id=camera_id, config=behavior_config)
    
    weapon_detector_config = WeaponDetectorConfig(
        model_paths=ModelPathConfig(yolo_weapon_weights=Path("models/yolov8m.pt"))
    )
    weapon_detector = WeaponDetector(camera_id=camera_id, config=weapon_detector_config)
    
    config = CameraSourceConfig(
        camera_id=camera_id,
        source=parsed_source,
        source_type=infer_source_type(parsed_source),
        display_name=f"Camera {cam_count + 1}",
        target_fps=20.0,
    )
    
    worker = manager.add_camera(config, detectors=[bag_detector, weapon_detector, behavior_detector])
    worker._all_detectors = [bag_detector, weapon_detector, behavior_detector]
    worker.start()
    
    print("Successfully added source")
    worker.stop()

if __name__ == "__main__":
    test_add_source()

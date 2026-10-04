from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
import json
import os
import re
import threading
import time
from urllib.parse import urlparse

from .auth import credentials_match, create_access_token, get_current_user, get_current_user_optional
from .streamer import generate_mjpeg_stream
from security_ai_system.cameras.camera_manager import CameraManager, CameraSourceConfig, infer_source_type
from dotenv import set_key
import psutil
from dataclasses import replace

_settings_lock = threading.Lock()
_login_lock = threading.Lock()
_login_failures: dict[str, list[float]] = {}
_LOGIN_WINDOW_SECONDS = 300.0
_LOGIN_MAX_FAILURES = 5
_SOURCE_MAX_LENGTH = 2048
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")

# Settings helpers
SETTINGS_FILE = Path(__file__).resolve().parent.parent / "outputs" / "settings.json"

def load_settings():
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "alarm_enabled": True,
        "bag_enabled": True,
        "bag_conf": 50,
        "weapon_enabled": True,
        "weapon_conf": 20,
        "behavior_enabled": True,
        "behavior_conf": 50,
        "audio_enabled": True,
        "audio_conf": 25,
        "gemini_api_key": "",
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "heatmap_enabled": True
    }

def save_settings(settings: dict) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = SETTINGS_FILE.with_suffix(".tmp")
    with _settings_lock:
        with temporary_file.open("w", encoding="utf-8") as file:
            json.dump(settings, file, sort_keys=True)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_file, SETTINGS_FILE)


def parse_confidence(form, name: str, default: int) -> int:
    try:
        value = int(form.get(name, default))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"{name} must be an integer") from exc
    if not 0 <= value <= 100:
        raise HTTPException(status_code=422, detail=f"{name} must be between 0 and 100")
    return value


def validate_source(source: str) -> str | int:
    value = source.strip()
    if not value or len(value) > _SOURCE_MAX_LENGTH or _CONTROL_CHARACTERS.search(value):
        raise HTTPException(status_code=422, detail="Invalid camera source")
    if value.isdigit():
        return int(value)
    parsed = urlparse(value)
    if len(value) >= 2 and value[1] == ":":
        return value
    if parsed.scheme and parsed.scheme not in {"rtsp", "http", "https"}:
        raise HTTPException(status_code=422, detail="Unsupported camera source scheme")
    if parsed.scheme and not parsed.netloc:
        raise HTTPException(status_code=422, detail="Network source must include a host")
    if parsed.scheme:
        allowed_hosts = {
            host.strip().lower()
            for host in os.getenv("ALLOWED_SOURCE_HOSTS", "").split(",")
            if host.strip()
        }
        if not parsed.hostname or parsed.hostname.lower() not in allowed_hosts:
            raise HTTPException(status_code=422, detail="Network source host is not allowlisted")
    return value


def client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def login_allowed(key: str, now: float) -> bool:
    with _login_lock:
        recent = [stamp for stamp in _login_failures.get(key, []) if now - stamp < _LOGIN_WINDOW_SECONDS]
        _login_failures[key] = recent
        return len(recent) < _LOGIN_MAX_FAILURES


def record_login_failure(key: str, now: float) -> None:
    with _login_lock:
        _login_failures.setdefault(key, []).append(now)


def apply_settings_to_pipeline(application: FastAPI) -> None:
    """Apply persisted settings to workers before they process live frames."""

    settings = application.state.settings
    manager = getattr(application.state, "camera_manager", None)
    if manager:
        for worker in manager.workers():
            all_detectors = getattr(worker, "_all_detectors", None)
            if all_detectors is None:
                all_detectors = list(worker.detectors)
                worker._all_detectors = all_detectors

            enabled_detectors = []
            for detector in all_detectors:
                detector_name = type(detector).__name__
                setting_key = {
                    "BagDetector": ("bag_enabled", "bag_conf"),
                    "SuspiciousBagDetector": ("bag_enabled", "bag_conf"),
                    "WeaponDetector": ("weapon_enabled", "weapon_conf"),
                    "BehaviorDetector": ("behavior_enabled", "behavior_conf"),
                }.get(detector_name)
                if setting_key is None or settings.get(setting_key[0], True):
                    if setting_key is not None:
                        detector.runtime = replace(
                            detector.runtime,
                            confidence_threshold=settings.get(setting_key[1], 20) / 100.0,
                        )
                    enabled_detectors.append(detector)
            worker.detectors = enabled_detectors

    alert_manager = getattr(application.state, "alert_manager", None)
    if alert_manager:
        alert_manager.config = replace(
            alert_manager.config,
            alarm_enabled=settings.get("alarm_enabled", True),
            save_snapshots=settings.get("save_snapshots", True),
        )

    audio_detector = getattr(application.state, "audio_detector", None)
    if audio_detector:
        audio_detector.classifier.config = replace(
            audio_detector.classifier.config,
            confidence_threshold=settings.get("audio_conf", 25) / 100.0,
        )

    heatmap = getattr(application.state, "heatmap_generator", None)
    if heatmap and manager:
        for worker in manager.workers():
            heatmap.set_enabled(
                worker.config.camera_id,
                settings.get("heatmap_enabled", True),
            )

# Setup directories
WEB_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_ROOT.parent


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.settings = load_settings()
    application.state.project_root = str(PROJECT_ROOT)

    from security_ai_system.utils.heatmap import HeatmapGenerator
    application.state.heatmap_generator = HeatmapGenerator()

    from security_ai_system.utils.telegram_notifier import TelegramNotifier
    settings = application.state.settings
    application.state.telegram_notifier = TelegramNotifier(
        bot_token=settings.get("telegram_bot_token", ""),
        chat_id=settings.get("telegram_chat_id", ""),
    )
    apply_settings_to_pipeline(application)
    try:
        yield
    finally:
        manager = getattr(application.state, "camera_manager", None)
        if manager:
            manager.stop_all()


app = FastAPI(title="AI Security Web Dashboard", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=WEB_ROOT / "static"), name="static")

# Mount snapshots for web UI access
snapshots_dir = PROJECT_ROOT / "outputs" / "snapshots"
snapshots_dir.mkdir(parents=True, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=snapshots_dir), name="snapshots")

templates = Jinja2Templates(directory=WEB_ROOT / "templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Redirect to dashboard, which will redirect to login if not authenticated."""
    return RedirectResponse(url="/dashboard", status_code=303)

@app.get("/settings", response_class=HTMLResponse)
async def get_settings(request: Request, username: Optional[str] = Depends(get_current_user_optional)):
    if not username:
        return RedirectResponse(url="/login", status_code=303)
        
    api_key = os.getenv("GEMINI_API_KEY", app.state.settings.get("gemini_api_key", ""))
    app.state.settings["gemini_api_key"] = api_key
    
    return templates.TemplateResponse(
        request=request,
        name="settings.html",
        context={"username": username, "settings": app.state.settings, "success": request.query_params.get("success")}
    )

@app.post("/settings")
async def post_settings(request: Request, username: str = Depends(get_current_user)):
    form = await request.form()
    
    settings = app.state.settings
    settings["gemini_api_key"] = form.get("gemini_api_key", "")
    settings["alarm_enabled"] = form.get("alarm_enabled") == "on"
    settings["bag_enabled"] = form.get("bag_enabled") == "on"
    settings["bag_conf"] = parse_confidence(form, "bag_conf", 50)
    settings["weapon_enabled"] = form.get("weapon_enabled") == "on"
    settings["weapon_conf"] = parse_confidence(form, "weapon_conf", 20)
    settings["behavior_enabled"] = form.get("behavior_enabled") == "on"
    settings["behavior_conf"] = parse_confidence(form, "behavior_conf", 50)
    settings["audio_enabled"] = form.get("audio_enabled") == "on"
    settings["audio_conf"] = parse_confidence(form, "audio_conf", 25)
    settings["telegram_bot_token"] = form.get("telegram_bot_token", "")
    settings["telegram_chat_id"] = form.get("telegram_chat_id", "")
    settings["heatmap_enabled"] = form.get("heatmap_enabled") == "on"
    
    save_settings(settings)
    
    if settings["gemini_api_key"]:
        env_path = Path(request.app.state.project_root) / ".env"
        if not env_path.exists(): env_path.touch()
        set_key(str(env_path), "GEMINI_API_KEY", settings["gemini_api_key"])
        os.environ["GEMINI_API_KEY"] = settings["gemini_api_key"]
        
        # Reload reporter
        alert_mgr = getattr(request.app.state, "alert_manager", None)
        if alert_mgr:
            from security_ai_system.utils.genai_reporter import GenAIReporter
            alert_mgr.reporter = GenAIReporter(output_dir=str(alert_mgr.output_dir / "reports"))
        
    # Dynamically update the pipeline by turning models on or off
    manager = getattr(request.app.state, "camera_manager", None)
    if manager:
        for worker in manager.workers():
            all_dets = getattr(worker, "_all_detectors", None)
            if all_dets is None:
                all_dets = list(worker.detectors)
                worker._all_detectors = all_dets

            new_detectors = []
            for det in all_dets:
                name = type(det).__name__
                if name in ("BagDetector", "SuspiciousBagDetector") and settings["bag_enabled"]:
                    det.runtime = replace(det.runtime, confidence_threshold=settings["bag_conf"] / 100.0)
                    new_detectors.append(det)
                elif name == "WeaponDetector" and settings["weapon_enabled"]:
                    det.runtime = replace(det.runtime, confidence_threshold=settings["weapon_conf"] / 100.0)
                    new_detectors.append(det)
                elif name == "BehaviorDetector" and settings["behavior_enabled"]:
                    det.runtime = replace(det.runtime, confidence_threshold=settings["behavior_conf"] / 100.0)
                    new_detectors.append(det)
            worker.detectors = new_detectors
        
    # Update system alarm state and audio detector dynamically
    alert_mgr = getattr(request.app.state, "alert_manager", None)
    if alert_mgr:
        alert_mgr.config = replace(alert_mgr.config, alarm_enabled=settings["alarm_enabled"])

    audio_det = getattr(request.app.state, "audio_detector", None)
    if audio_det:
        audio_det.classifier.config = replace(
            audio_det.classifier.config, 
            confidence_threshold=settings["audio_conf"] / 100.0
        )

    # Update Telegram notifier
    telegram = getattr(request.app.state, "telegram_notifier", None)
    if telegram:
        telegram.bot_token = settings.get("telegram_bot_token", "")
        telegram.chat_id = settings.get("telegram_chat_id", "")

    # Update heatmap state
    heatmap = getattr(request.app.state, "heatmap_generator", None)
    if heatmap and manager:
        for w in manager.workers():
            heatmap.set_enabled(w.config.camera_id, settings.get("heatmap_enabled", True))

    return RedirectResponse(url="/settings?success=1", status_code=303)

@app.post("/api/add_source")
async def add_source(request: Request, source: str = Form(...), username: str = Depends(get_current_user)):
    manager: CameraManager = request.app.state.camera_manager
    existing_ids = {worker.config.camera_id for worker in manager.workers()}
    camera_number = 1
    while f"camera-{camera_number}" in existing_ids:
        camera_number += 1
    camera_id = f"camera-{camera_number}"
    
    parsed_source = validate_source(source)
    source_type = infer_source_type(parsed_source)
    
    # Import detectors and reuse shared GlobalReIDManager
    from security_ai_system.detectors.bag_detector import BagDetector, BagDetectorConfig
    from security_ai_system.detectors.weapon_detector import WeaponDetector, WeaponDetectorConfig
    from security_ai_system.detectors.behavior_detector import BehaviorDetector, BehaviorDetectorConfig
    from security_ai_system.trackers.tracker import DeepSortTracker
    from security_ai_system.utils.reid_manager import GlobalReIDManager
    from security_ai_system.utils.types import ModelPathConfig
    
    reid_manager = getattr(request.app.state, "reid_manager", None)
    if reid_manager is None:
        reid_manager = GlobalReIDManager(similarity_threshold=0.75)
        request.app.state.reid_manager = reid_manager

    tracker = DeepSortTracker(reid_manager=reid_manager)
    bag_detector = BagDetector(
        camera_id=camera_id,
        tracker=tracker,
        config=BagDetectorConfig(
            model_paths=ModelPathConfig(
                yolo_object_weights=PROJECT_ROOT / "yolov8m.pt",
            )
        ),
    )
    
    configured_weapon_path = os.getenv("WEAPON_MODEL_PATH", "").strip()
    weapon_weights = (
        Path(configured_weapon_path)
        if configured_weapon_path
        else PROJECT_ROOT / "models" / "weapon_yolov8.pt"
    )
    if not weapon_weights.is_absolute():
        weapon_weights = PROJECT_ROOT / weapon_weights
    if not weapon_weights.exists():
        # The COCO checkpoint supports knife detection, not firearm detection.
        weapon_weights = PROJECT_ROOT / "yolov8m.pt"
    weapon_detector_config = WeaponDetectorConfig(
        model_paths=ModelPathConfig(yolo_weapon_weights=weapon_weights)
    )
    detectors = [bag_detector]
    if weapon_weights.exists():
        detectors.append(
            WeaponDetector(camera_id=camera_id, config=weapon_detector_config)
        )
    
    from security_ai_system.cameras.camera_manager import CameraSourceType
    if source_type != CameraSourceType.VIDEO_FILE:
        behavior_config = BehaviorDetectorConfig(
            model_paths=ModelPathConfig(yolo_pose_weights=PROJECT_ROOT / "models" / "yolov8m-pose.pt")
        )
        behavior_detector = BehaviorDetector(camera_id=camera_id, config=behavior_config)
        detectors.append(behavior_detector)
    
    config = CameraSourceConfig(
        camera_id=camera_id,
        source=parsed_source,
        source_type=source_type,
        display_name=f"Camera {camera_number}",
        target_fps=20.0,
    )
    
    worker = manager.add_camera(config, detectors=detectors)
    worker._all_detectors = list(detectors)
    worker.start()
    if not worker.wait_until_started(timeout=5.0):
        manager.remove_camera(camera_id)
        status = worker.status()
        detail = status.last_error or "Camera source did not become ready in time"
        raise HTTPException(status_code=503, detail=f"Unable to start camera source: {detail}")
    
    return RedirectResponse(url="/dashboard", status_code=303)

@app.post("/api/remove_source/{camera_id}")
async def remove_source(camera_id: str, request: Request, username: str = Depends(get_current_user)):
    manager: CameraManager = request.app.state.camera_manager
    try:
        manager.remove_camera(camera_id)
        return JSONResponse(content={"ok": True})
    except KeyError:
        return JSONResponse(content={"ok": False, "error": "Camera not found"}, status_code=404)

@app.get("/login", response_class=HTMLResponse)
async def get_login(request: Request):
    """Serve the login page."""
    return templates.TemplateResponse(request=request, name="login.html")

@app.post("/login")
async def post_login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    """Handle login form submission."""
    username = username.strip()
    key = client_key(request)
    now = time.monotonic()
    if not login_allowed(key, now):
        raise HTTPException(
            status_code=429,
            detail="Too many login attempts. Try again later.",
            headers={"Retry-After": str(int(_LOGIN_WINDOW_SECONDS))},
        )

    if len(username) > 128 or len(password) > 512 or not credentials_match(username, password):
        record_login_failure(key, now)
        return RedirectResponse(url="/login?error=1", status_code=303)

    with _login_lock:
        _login_failures.pop(key, None)
    token = create_access_token(data={"sub": username})
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        secure=os.getenv("COOKIE_SECURE", "0") == "1",
        max_age=86400,
        samesite="strict",
        path="/",
    )
    return response

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("session_token", samesite="strict", path="/")
    return response

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(request: Request, username: Optional[str] = Depends(get_current_user_optional)):
    """Serve the main dashboard page."""
    if not username:
        return RedirectResponse(url="/login", status_code=303)
        
    # We will get the camera manager from app.state
    manager: CameraManager = request.app.state.camera_manager
    cameras = [w.config.camera_id for w in manager.workers()]
    
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "username": username,
            "cameras": cameras
        }
    )

@app.get("/video_feed/{camera_id}")
async def video_feed(request: Request, camera_id: str, username: str = Depends(get_current_user)):
    """Stream MJPEG video for a specific camera."""
    manager: CameraManager = request.app.state.camera_manager
    try:
        manager.get_worker(camera_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Camera not found") from exc
    return StreamingResponse(
        generate_mjpeg_stream(camera_id, manager, request),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/events")
async def get_events(request: Request, username: str = Depends(get_current_user)):
    """Return latest threats and overall system status with O(1) in-memory retrieval."""
    manager = getattr(request.app.state, "camera_manager", None)
    statuses = manager.statuses() if manager else []
    alert_mgr = getattr(request.app.state, "alert_manager", None)

    events = []
    if alert_mgr and hasattr(alert_mgr, "recent_events"):
        events = alert_mgr.recent_events(limit=10)
    
    # Fallback to reading file if in-memory queue is empty
    if not events:
        project_root = getattr(request.app.state, "project_root", str(PROJECT_ROOT))
        log_file = Path(project_root) / "outputs" / "logs" / "threat_events.jsonl"
        if log_file.exists():
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    from collections import deque
                    recent_lines = deque(f, maxlen=10)
                    for line in reversed(recent_lines):
                        events.append(json.loads(line.strip()))
            except Exception:
                pass

    current_level = alert_mgr.threat_engine.current_state().level.value if alert_mgr else "LOW"

    return JSONResponse(content={
        "sys_health": {
            "cpu": psutil.cpu_percent(interval=None),
            "ram": psutil.virtual_memory().percent
        },
        "current_threat_level": current_level,
        "statuses": [
            {
                "camera_id": s.camera_id,
                "fps": round(s.fps, 1),
                "frame_count": s.frame_count,
                "connected": s.connected,
                "models_ready": s.models_ready,
            }
            for s in statuses
        ],
        "events": events[:10]
    })

@app.post("/api/heatmap/toggle")
async def toggle_heatmap(request: Request, username: str = Depends(get_current_user)):
    """Toggle heatmap overlay on/off for all cameras."""
    heatmap = getattr(request.app.state, "heatmap_generator", None)
    if not heatmap:
        return JSONResponse(content={"enabled": False})
    
    manager = getattr(request.app.state, "camera_manager", None)
    if not manager:
        return JSONResponse(content={"enabled": False})

    workers = manager.workers()
    if workers:
        current = heatmap.is_enabled(workers[0].config.camera_id)
        new_state = not current
        for w in workers:
            heatmap.set_enabled(w.config.camera_id, new_state)
        request.app.state.settings["heatmap_enabled"] = new_state
        save_settings(request.app.state.settings)
        return JSONResponse(content={"enabled": new_state})
    return JSONResponse(content={"enabled": False})

@app.post("/api/heatmap/reset")
async def reset_heatmap(request: Request, username: str = Depends(get_current_user)):
    """Reset heatmap accumulation data."""
    heatmap = getattr(request.app.state, "heatmap_generator", None)
    if heatmap:
        manager = getattr(request.app.state, "camera_manager", None)
        if manager:
            for w in manager.workers():
                heatmap.reset(w.config.camera_id)
    return JSONResponse(content={"ok": True})

# Reports cache
_REPORTS_CACHE: list[dict] = []
_REPORTS_LAST_LOADED: float = 0.0

@app.get("/api/reports")
async def get_reports(request: Request, username: str = Depends(get_current_user)):
    """Return all generated AI incident reports."""
    project_root = getattr(request.app.state, "project_root", str(PROJECT_ROOT))
    return JSONResponse(content=_load_reports(project_root))

@app.get("/api/search")
async def search_reports(request: Request, query: str, username: str = Depends(get_current_user)):
    """Search AI reports for a specific text query."""
    project_root = getattr(request.app.state, "project_root", str(PROJECT_ROOT))
    reports = _load_reports(project_root)
    query_str = query.lower()
    
    results = [
        r for r in reports 
        if query_str in r.get("ai_summary", "").lower() or query_str in r.get("threat_label", "").lower()
    ]
    return JSONResponse(content=results)

def _load_reports(project_root: str) -> list[dict]:
    global _REPORTS_CACHE, _REPORTS_LAST_LOADED
    reports_dir = Path(project_root) / "outputs" / "reports"
    if not reports_dir.exists():
        return []

    # Simple cache invalidation (cache for 3 seconds)
    import time
    now = time.time()
    if _REPORTS_CACHE and (now - _REPORTS_LAST_LOADED < 3.0):
        return _REPORTS_CACHE

    reports = []
    for report_file in sorted(reports_dir.glob("*.json"), reverse=True):
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                img_path = data.get("image_path", "")
                if img_path:
                    filename = Path(img_path).name
                    data["image_url"] = f"/snapshots/{filename}"
                reports.append(data)
        except Exception:
            pass
            
    _REPORTS_CACHE = reports
    _REPORTS_LAST_LOADED = now
    return reports

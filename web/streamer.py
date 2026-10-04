import asyncio
import cv2
from typing import AsyncIterator
from fastapi import Request
from security_ai_system.cameras.camera_manager import CameraManager


def _encode_jpeg(frame, quality: int = 70) -> bytes | None:
    """Synchronous OpenCV JPEG encoder executed in a worker thread."""
    try:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode(".jpg", frame, encode_param)
        return buffer.tobytes() if success else None
    except Exception:
        return None


async def generate_mjpeg_stream(
    camera_id: str,
    manager: CameraManager,
    request: Request,
) -> AsyncIterator[bytes]:
    """
    Asynchronous generator that yields MJPEG frames for a given camera.
    Uses asyncio.to_thread for non-blocking JPEG encoding and caches frames to eliminate
    redundant compression cycles when source FPS is below polling rate.
    """
    try:
        worker = manager.get_worker(camera_id)
    except KeyError:
        return

    heatmap = getattr(request.app.state, "heatmap_generator", None)
    polling_interval = 1.0 / 30.0
    last_frame_index = -1
    last_jpeg_bytes: bytes | None = None

    while True:
        if await request.is_disconnected():
            break

        try:
            result = worker.latest_result()
            raw_frame = worker.latest_frame()
        except Exception:
            break

        frame_index = result.frame_index if result is not None else (
            raw_frame.frame_index if raw_frame is not None else -1
        )
        if frame_index >= 0:
            if frame_index != last_frame_index:
                last_frame_index = frame_index
                display_frame = result.frame if result is not None else raw_frame.frame

                # Apply heatmap overlay if enabled
                if result is not None and heatmap and heatmap.is_enabled(camera_id):
                    centroids = [
                        det.bbox.center
                        for det_result in result.detector_results
                        for det in det_result.detections
                        if det.bbox is not None and det.label.lower() == "person"
                    ]
                    display_frame = heatmap.update(camera_id, display_frame, centroids)

                # Offload CPU-heavy JPEG compression to threadpool
                jpeg_bytes = await asyncio.to_thread(_encode_jpeg, display_frame, 70)
                if jpeg_bytes:
                    last_jpeg_bytes = jpeg_bytes

            if last_jpeg_bytes is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + last_jpeg_bytes
                    + b"\r\n"
                )

        await asyncio.sleep(polling_interval)

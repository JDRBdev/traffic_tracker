import asyncio
import base64
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from loguru import logger

from config import settings
from capture import StreamReader, get_stream_url
from detector import TrafficDetector
from notifier import EmailNotifier

# Global instances
detector: TrafficDetector | None = None
notifier: EmailNotifier | None = None
reader: StreamReader | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, notifier, reader
    logger.info("Starting up traffic-monitor server...")

    detector = TrafficDetector()
    notifier = EmailNotifier()

    # Exponential backoff for stream URL extraction
    max_attempts = 5
    stream_url = None
    for attempt in range(max_attempts):
        try:
            stream_url = get_stream_url(settings.YOUTUBE_URL)
            break
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"Failed to get stream URL (attempt {attempt + 1}/{max_attempts}): {e}. Retrying in {wait}s...")
            await asyncio.sleep(wait)

    if not stream_url:
        logger.critical("Could not extract stream URL after 5 attempts. Exiting.")
        raise RuntimeError("Failed to extract YouTube stream URL.")

    # Exponential backoff for opening the stream reader
    for attempt in range(max_attempts):
        try:
            reader = StreamReader(stream_url)
            break
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"Failed to open stream (attempt {attempt + 1}/{max_attempts}): {e}. Retrying in {wait}s...")
            await asyncio.sleep(wait)

    if not reader:
        logger.critical("Could not open OpenCV VideoCapture after 5 attempts. Exiting.")
        raise RuntimeError("Failed to open video stream.")

    yield

    logger.info("Shutting down...")
    if reader:
        reader.release()


app = FastAPI(lifespan=lifespan)


class ROIPayload(BaseModel):
    polygon: list[list[int]]


@app.get("/config/roi")
def get_roi():
    """Returns the current ROI polygon."""
    return {"polygon": detector.roi_polygon.tolist() if detector else []}


@app.post("/config/roi")
def update_roi(payload: ROIPayload):
    """Hot-updates the ROI polygon."""
    if detector:
        detector.update_roi(payload.polygon)
        # We also keep it in settings to keep state consistent during runtime
        settings.ROI_POLYGON = payload.polygon
        return {"status": "success", "vertices": len(payload.polygon)}
    return {"status": "error", "message": "Detector not initialized"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Fix: Removed BackgroundTasks from signature as it doesn't execute in WebSockets
    await websocket.accept()
    logger.info("WebSocket client connected.")

    frame_interval = 1.0 / settings.FRAME_RATE

    try:
        while True:
            start_time = asyncio.get_event_loop().time()

            # Read frame
            frame = reader.read_frame() if reader else None

            if frame is None:
                # If stream lags or drops, sleep briefly and try again
                await asyncio.sleep(0.1)
                continue

            # Run detection
            if detector:
                result = detector.detect(frame)

                # Check for pink vehicles and trigger notification
                pink_detected = len(result.pink_vehicles) > 0
                if pink_detected and notifier:
                    # Notify for each pink vehicle found
                    for crop, track_id in zip(result.pink_vehicles, result.pink_vehicle_ids):
                        asyncio.create_task(asyncio.to_thread(notifier.notify, crop, track_id))

                # Encode frame for frontend
                _, buf = cv2.imencode(".jpg", result.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                b64_frame = base64.b64encode(buf).decode("utf-8")

                # Build payload
                payload = {
                    "vehicle_count": result.vehicle_count,
                    "pedestrian_in_roi": result.pedestrian_in_roi,
                    "pink_detected": pink_detected,
                    "frame": b64_frame
                }

                await websocket.send_json(payload)

            # Throttle to meet FRAME_RATE
            elapsed = asyncio.get_event_loop().time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            await asyncio.sleep(sleep_time)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        await websocket.close()

# Mount the static frontend directory at the root (must be at the end to avoid intercepting routes)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

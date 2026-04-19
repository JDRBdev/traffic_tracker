# traffic-monitor — Agent Rules

## Project Overview

Real-time traffic analysis system. Reads frames from a YouTube live stream via
`yt-dlp` + OpenCV, runs YOLOv8 object detection, counts vehicles and pedestrians
inside a configurable ROI, detects pink vehicles via HSV masking, and fires email
alerts via SMTP or SendGrid.

**Stack:** Python 3.11 · FastAPI · OpenCV · Ultralytics YOLOv8 · Pydantic · Loguru · uvicorn

---

## Architecture — Do Not Mix Concerns

```
YouTube Stream
     │
     ▼ (yt-dlp + OpenCV)
 capture.py      ← stream reading only
     │
     ▼
 detector.py     ← YOLO inference, ROI logic, HSV pink filter
     │
     ├──► notifier.py   ← email dispatch + cooldown
     └──► server.py     ← FastAPI + WebSocket + static frontend
```

Each file owns exactly one concern. Never add detection logic to `server.py`,
never add HTTP logic to `detector.py`, etc.

---

## File Responsibilities

| File | Owns |
|---|---|
| `config.py` | All configuration via Pydantic `Settings` loaded from `.env` |
| `capture.py` | `get_stream_url()` + `StreamReader` class |
| `detector.py` | `TrafficDetector` class + `DetectionResult` dataclass |
| `notifier.py` | `EmailNotifier` class with cooldown |
| `server.py` | FastAPI app, WebSocket loop, REST endpoints |
| `frontend/index.html` | Single-page dashboard, no external JS frameworks |

---

## Non-Negotiable Rules

1. **No hardcoded values.** Every numeric constant, URL, or credential must come
   from `config.Settings`. Never use literals like `60`, `"smtp.gmail.com"`, or
   `0.15` directly in business logic.

2. **No `print()`.** Use `loguru` exclusively for all output.

3. **No disk I/O for frames.** All frame/image handling is in-memory
   (`cv2.imencode`, `BytesIO`). Disk writes are only allowed when `DEBUG=true`
   in `.env`.

4. **Load YOLO once.** The model is loaded in `TrafficDetector.__init__()`.
   Never reload it per frame or per request.

5. **Dual HSV range for pink.** Always apply both `[140–170]` and `[0–10]`
   ranges. Pink near red wraps around the 0°/180° hue boundary.

6. **Exponential backoff on stream errors.** Max 5 attempts, then log critical
   and exit. Pattern: `wait = 2 ** attempt`.

7. **Cooldown enforced inside `EmailNotifier`.** Not at the call site in
   `server.py`. The notifier is responsible for suppressing duplicate sends.

8. **ROI hot-reload.** `POST /config/roi` must update the polygon at runtime
   without restarting the server.

9. **Type hints on every public function.** No bare `def func(x):` signatures.

10. **`black` + `isort`, max 100 chars per line.**

---

## Configuration (`config.py`)

Use Pydantic `BaseSettings`. Load all values from `.env` via `python-dotenv`.

```python
class Settings(BaseSettings):
    YOUTUBE_URL: str
    FRAME_RATE: int = 5
    YOLO_MODEL: str = "yolov8n.pt"
    ROI_POLYGON: list[list[int]]        # e.g. [[120,380],[320,380],...]
    PINK_HUE_LOW: int = 140
    PINK_HUE_HIGH: int = 170
    PINK_SAT_LOW: int = 80
    PINK_VAL_LOW: int = 80
    PINK_PIXEL_THRESHOLD: float = 0.15
    EMAIL_COOLDOWN_SECONDS: int = 60
    SMTP_HOST: str
    SMTP_PORT: int = 587
    SMTP_USER: str
    SMTP_PASSWORD: str
    EMAIL_FROM: str
    EMAIL_TO: str
    SENDGRID_API_KEY: str = ""
    DEBUG: bool = False
```

---

## `capture.py` Contracts

- `get_stream_url(youtube_url: str) -> str` — calls `yt-dlp`, returns direct
  HLS URL for best stream ≤ 720p.
- `StreamReader.__init__(stream_url: str)` — opens `cv2.VideoCapture`.
- `StreamReader.read_frame() -> np.ndarray | None`
- `StreamReader.release() -> None`
- FPS throttling lives in `server.py`, not here.

---

## `detector.py` Contracts

```python
@dataclass
class DetectionResult:
    vehicle_count: int
    pedestrian_in_roi: int
    pink_vehicles: list[np.ndarray]   # BGR crops, in-memory only
    annotated_frame: np.ndarray
```

- YOLO vehicle classes: `car`, `truck`, `bus`, `motorcycle`
- YOLO pedestrian class: `person`
- ROI check: `cv2.pointPolygonTest(polygon, centroid, False) >= 0`
- `_draw_roi()` draws the polygon in **green**
- `_is_pink()` applies dual HSV mask:

```python
mask = (
    cv2.inRange(hsv, lower_pink, upper_pink)
    | cv2.inRange(hsv, np.array([0, SAT, VAL]), np.array([10, 255, 255]))
)
ratio = cv2.countNonZero(mask) / (h * w)
return ratio >= settings.PINK_PIXEL_THRESHOLD
```

---

## `notifier.py` Contracts

- SendGrid if `SENDGRID_API_KEY` is non-empty; otherwise SMTP.
- `notify(frame_crop: np.ndarray) -> None`:
  - Encodes image in memory: `_, buf = cv2.imencode(".jpg", crop)`
  - Email subject includes ISO 8601 timestamp.
  - Skips send and logs suppression if within cooldown window.

---

## `server.py` Contracts

- WebSocket `/ws` ticks every ~200 ms.
- JSON payload per tick:
  ```json
  {
    "vehicle_count": 12,
    "pedestrian_in_roi": 3,
    "pink_detected": true,
    "frame": "<base64-jpeg>"
  }
  ```
- `notify()` called via background task, never blocking the WS loop.
- `GET /config/roi` → returns current polygon.
- `POST /config/roi` → hot-updates polygon, no restart needed.

---

## `frontend/index.html` Contracts

- **No external frameworks** (no React, Vue, jQuery).
- On each WS message: decode base64 → draw on `<canvas>`.
- Show `🚗 Rosa detectado` badge for 5 s when `pink_detected === true`.
- ROI editor: click on canvas → define vertices → `POST /config/roi`.

---

## Testing

- `pytest tests/ -v`
- `test_detector.py`: feed static image, assert `DetectionResult` type, no exceptions.
- `test_notifier.py`: mock SMTP, assert second send within cooldown is suppressed.
- Use `pytest-asyncio` for WebSocket tests.
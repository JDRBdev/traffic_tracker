---
trigger: always_on
---

# traffic-monitor — Antigravity Rules (GEMINI.md)

> Antigravity-specific overrides. Applied with highest priority over AGENTS.md.

---

## Agent Behavior

- **Always plan before coding.** For any task touching more than one module,
  produce an `implementation_plan.md` that lists which files change and why
  before writing a single line of code.

- **Break tasks into sub-tasks ≤ 1 hour** inside `task.md`. Each sub-task must
  name the file it touches and its acceptance criterion.

- **Verify after every change.** Run `pytest tests/ -v` after any modification
  to `detector.py` or `notifier.py`. Do not mark a task complete without a
  passing test run.

- **Never modify `config.py` structure without user approval.** Adding or
  removing fields from `Settings` is a breaking change — ask first.

---

## Artifact Rules

- `implementation_plan.md` must include a **Security Implications** section
  whenever the task touches `notifier.py`, `.env`, or any credential handling.

- `walkthrough.md` must include the terminal output of `pytest tests/ -v`
  as proof that tests pass.

---

## Code Generation Style

- Python 3.11+ syntax only. Use `match` statements where appropriate.
- Use `X | Y` union syntax, not `Optional[X]` or `Union[X, Y]`.
- Dataclasses over dicts for structured returns (`DetectionResult`, etc.).
- All exceptions must be caught and logged with `logger.exception()`, never
  silently swallowed.

```python
# ✅ Correct
try:
    frame = reader.read_frame()
except Exception:
    logger.exception("Failed to read frame")
    return None

# ❌ Wrong
try:
    frame = reader.read_frame()
except:
    pass
```

---

## Browser Agent Rules

- When using the browser to test the dashboard (`frontend/index.html`):
  - Open `http://localhost:8000` after `uvicorn server:app --reload` is running.
  - Verify the canvas updates with annotated frames.
  - Verify counters increment when YOLO detects objects.
  - Verify the pink badge appears and auto-hides after 5 seconds.
- Do **not** browse to `youtube.com` or any external URL during testing unless
  explicitly asked.

---

## Security — Agent Allow List

The agent may only access:
- Local filesystem within the project directory.
- `localhost:8000` (dashboard testing).
- PyPI (dependency installs via `pip`).
- No outbound connections to external APIs during automated tasks, except
  `smtp.gmail.com:587` or `api.sendgrid.com` when explicitly testing email.

---

## Environment & Secrets

- Never read `.env` directly. Always use `from config import settings`.
- Never log credential values. Log only field names when debugging config.
- `.env` must never be committed. `.env.example` is the canonical reference.

```dotenv
YOUTUBE_URL=https://www.youtube.com/watch?v=87A5XEiV5fk
FRAME_RATE=5
YOLO_MODEL=yolov8n.pt
ROI_POLYGON=[[120,380],[320,380],[320,480],[120,480]]
PINK_HUE_LOW=140
PINK_HUE_HIGH=170
PINK_SAT_LOW=80
PINK_VAL_LOW=80
PINK_PIXEL_THRESHOLD=0.15
EMAIL_COOLDOWN_SECONDS=60
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=tu@gmail.com
SMTP_PASSWORD=xxxx-xxxx-xxxx-xxxx
EMAIL_FROM=tu@gmail.com
EMAIL_TO=destino@email.com
SENDGRID_API_KEY=
DEBUG=false
```

---

## Performance Targets

| Hardware      | Model   | Target fps | Max latency/frame |
|---------------|---------|------------|-------------------|
| CPU (4 cores) | yolov8n | ~5 fps     | 200 ms            |
| GPU RTX 3060  | yolov8s | ~30 fps    | 30 ms             |
| RPi 4         | yolov8n | ~2 fps     | 500 ms            |

Default `FRAME_RATE=5` is correct for CPU. Do not suggest increasing it
without confirming GPU availability.
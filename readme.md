# 🚗 Traffic Tracker

Real-time traffic analysis system that monitors YouTube live streams to detect vehicles and pedestrians, with a specialized focus on identifying "pink" vehicles and logging events to Google Sheets.

## 🌟 Key Features

- **Real-time Detection**: Processes live YouTube streams using YOLOv8 for high-accuracy vehicle and pedestrian detection.
- **ROI (Region of Interest)**: Define custom polygonal zones on the video feed to monitor specific areas for pedestrian activity.
- **Pink Vehicle Detector**: Specialized HSV color filtering to identify vehicles of a specific pink hue, triggering immediate alerts.
- **Live Dashboard**: A FastAPI-powered web interface with a WebSocket stream providing annotated frames and real-time traffic counts.
- **Heatmap Visualization**: Visualizes traffic density and movement patterns over time.
- **Automated Logging**: Integration with Google Sheets to record every detection event, including timestamps and image crops.
- **Instant Notifications**: Email alerts via SMTP or SendGrid when specialized vehicles (pink) are detected.
- **Dynamic Configuration**: Update the ROI polygon and toggle heatmaps in real-time via the dashboard without restarting the server.

## 🏗 Architecture

The system is designed with a clear separation of concerns:

`YouTube Stream` $\rightarrow$ `capture.py` (Stream Reading) $\rightarrow$ `detector.py` (YOLO Inference & ROI Logic) $\rightarrow$ `server.py` (API & WebSockets) $\rightarrow$ `notifier.py` / `sheets_logger.py` (Alerts & Logging)

## 🛠 Tech Stack

- **Language**: Python 3.12
- **AI/ML**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), [OpenCV](https://opencv.org/)
- **Stream Handling**: `yt-dlp`
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/), Uvicorn, WebSockets
- **Data & Logging**: `gspread`, `oauth2client` (Google Sheets API)
- **Notifications**: SendGrid / SMTP
- **Configuration**: Pydantic Settings (`.env`)
- **Logging**: Loguru

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- A Google Cloud project with the Google Sheets API enabled and a `credentials.json` service account key.
- A YouTube live stream URL.

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd traffic-tracker
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure the environment:
   Create a `.env` file in the root directory based on the provided example:
   ```env
   YOUTUBE_URL=your_youtube_stream_url
   FRAME_RATE=5
   YOLO_MODEL=yolov8n.pt
   
   # Google Sheets
   GOOGLE_SHEETS_CREDENTIALS_FILE=credentials.json
   GOOGLE_SHEET_NAME=Traffic_tracker
   
   # Email Alerts
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USER=your_email@gmail.com
   SMTP_PASSWORD=your_app_password
   EMAIL_FROM=your_email@gmail.com
   EMAIL_TO=alert_recipient@gmail.com
   SENDGRID_API_KEY=your_api_key_optional
   ```

4. Run the server:
   ```bash
   python server.py
   ```

5. Access the dashboard:
   Open `http://localhost:8000` in your browser.

## 📊 API & Dashboard

### WebSocket Endpoint
- `/ws`: Streams real-time JSON payloads containing:
  - `vehicle_count`: Total vehicles in frame.
  - `pedestrian_in_roi`: Pedestrians detected within the defined ROI.
  - `pink_detected`: Boolean flag for pink vehicle presence.
  - `frame`: Base64 encoded JPEG of the annotated frame.

### REST Endpoints
- `GET /config/roi`: Retrieve the current ROI polygon.
- `POST /config/roi`: Update the ROI polygon coordinates.
- `POST /config/heatmap`: Toggle the heatmap visualization on/off.
- `GET /api/history`: Retrieve historical traffic data for charts.

## 📝 Event Logging & Alerts

- **Google Sheets**: Every time a pedestrian enters the ROI or a pink vehicle is detected, a row is added to the `Traffic_tracker` sheet with a timestamp and a link to the captured image crop.
- **Email Alerts**: Pink vehicle detections trigger an email notification. A cooldown period (configurable in `.env`) is enforced to prevent spamming.

## 🧪 Testing

The project includes a suite of tests to ensure detector and notifier reliability.

```bash
pytest tests/ -v
```

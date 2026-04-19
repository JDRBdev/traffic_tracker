import subprocess
import cv2
import numpy as np
from loguru import logger

def get_stream_url(youtube_url: str) -> str:
    """
    Calls yt-dlp to extract the direct HLS/DASH stream URL from a YouTube URL.
    Requests the best format with height <= 720p to save bandwidth.
    """
    try:
        logger.info(f"Extracting stream URL for {youtube_url}")
        cmd = ["yt-dlp", "-f", "best[height<=720]", "-g", youtube_url]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stream_url = result.stdout.strip()
        if not stream_url:
            raise ValueError("yt-dlp returned an empty string.")
        return stream_url
    except subprocess.CalledProcessError as exc:
        logger.exception(f"yt-dlp failed: {exc.stderr}")
        raise
    except Exception:
        logger.exception("Failed to get stream URL")
        raise

class StreamReader:
    """Reads frames from a video stream URL."""
    
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.cap = cv2.VideoCapture(stream_url)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video stream: {stream_url}")
        logger.info("Stream opened successfully.")

    def read_frame(self) -> np.ndarray | None:
        """Reads the next frame. Returns None if the stream ends or fails."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self) -> None:
        """Releases the underlying OpenCV capture object."""
        if self.cap:
            self.cap.release()
            logger.info("Stream released.")

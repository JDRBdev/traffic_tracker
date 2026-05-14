import time
import os
import cv2
import gspread
import numpy as np
from datetime import datetime
from loguru import logger
from oauth2client.service_account import ServiceAccountCredentials

from config import settings

class GoogleSheetsLogger:
    def __init__(self):
        self.credentials_file = settings.GOOGLE_SHEETS_CREDENTIALS_FILE
        self.sheet_name = settings.GOOGLE_SHEET_NAME
        self.captures_dir = settings.EVENT_CAPTURES_DIR
        self.save_captures = settings.SAVE_CAPTURES_TO_DISK
        self.cooldown = settings.SHEETS_COOLDOWN_SECONDS
        
        self.client = None
        self.sheet = None
        self._last_logged = {}
        
        # Ensure captures directory exists
        if self.save_captures:
            os.makedirs(self.captures_dir, exist_ok=True)
            
        self._connect()

    def _connect(self):
        """Authenticates and connects to the Google Sheet."""
        try:
            if not os.path.exists(self.credentials_file):
                logger.warning(f"Google Sheets credentials file not found: {self.credentials_file}. Sheet logging disabled.")
                return
                
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            creds = ServiceAccountCredentials.from_json_keyfile_name(self.credentials_file, scope)
            self.client = gspread.authorize(creds)
            
            # Try to open the sheet
            self.sheet = self.client.open(self.sheet_name).sheet1
            logger.info(f"Connected to Google Sheet: {self.sheet_name}")
            
            # Check if headers exist, if not create them
            headers = self.sheet.row_values(1)
            expected_headers = ["timestamp", "tipo", "zona", "captura"]
            if not headers or headers != expected_headers:
                if not headers:
                    self.sheet.append_row(expected_headers)
                    logger.info("Added headers to empty Google Sheet.")
                else:
                    logger.warning(f"Google Sheet headers don't match expected. Found: {headers}")
                    
        except gspread.exceptions.SpreadsheetNotFound:
            logger.error(f"Google Sheet '{self.sheet_name}' not found. Make sure the service account has access.")
            self.client = None
            self.sheet = None
        except Exception as e:
            logger.exception(f"Failed to connect to Google Sheets: {e}")
            self.client = None
            self.sheet = None

    def _should_log(self, event_type: str, track_id: int) -> bool:
        """Checks if an event with this track_id is within cooldown to avoid duplicates."""
        # If no track_id, always log (though we shouldn't have many of these)
        if track_id is None or track_id == -1:
            return True
            
        now = time.time()
        key = f"{event_type}_{track_id}"
        
        if key in self._last_logged:
            elapsed = now - self._last_logged[key]
            if elapsed < self.cooldown:
                return False
                
        self._last_logged[key] = now
        
        # Cleanup old entries to avoid memory leak
        keys_to_delete = [k for k, v in self._last_logged.items() if (now - v) > self.cooldown * 2]
        for k in keys_to_delete:
            del self._last_logged[k]
            
        return True

    def log_event(self, event_type: str, zone: str, frame_or_crop: np.ndarray, track_id: int = None):
        """Logs an event to Google Sheets and saves the local capture."""
        if not self.sheet:
            return
            
        if not self._should_log(event_type, track_id):
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        safe_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        capture_url = "N/A"
        
        if self.save_captures and frame_or_crop is not None and frame_or_crop.size > 0:
            filename = f"{event_type.replace(' ', '_').lower()}_{safe_timestamp}_{track_id}.jpg"
            filepath = os.path.join(self.captures_dir, filename)
            try:
                cv2.imwrite(filepath, frame_or_crop)
                capture_url = f"/captures/{filename}"
            except Exception as e:
                logger.error(f"Failed to save capture {filepath}: {e}")
        
        # Log to sheet
        try:
            row = [timestamp, event_type, zone, capture_url]
            self.sheet.append_row(row)
            logger.info(f"Logged to Sheets: {event_type} in {zone}")
        except Exception as e:
            logger.error(f"Failed to append row to Google Sheets: {e}")
            # Try to reconnect once if it fails (token might have expired)
            self._connect()
            if self.sheet:
                try:
                    self.sheet.append_row(row)
                    logger.info(f"Logged to Sheets (after reconnect): {event_type} in {zone}")
                except Exception as e_retry:
                    logger.error(f"Failed to append row after reconnect: {e_retry}")

import smtplib
import base64
import time
from datetime import datetime, timezone
from email.message import EmailMessage

import cv2
import numpy as np
from loguru import logger
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (
    Mail,
    Attachment,
    FileContent,
    FileName,
    FileType,
    Disposition,
)

from config import settings


class EmailNotifier:
    """Handles sending email notifications with a cooldown window."""
    
    def __init__(self):
        self._last_sent_at: float = 0.0

    def notify(self, frame_crop: np.ndarray) -> None:
        """Sends an email with the attached image crop if cooldown has passed."""
        now = time.time()
        if now - self._last_sent_at < settings.EMAIL_COOLDOWN_SECONDS:
            logger.info("Notification suppressed due to cooldown window.")
            return

        logger.info("Preparing pink vehicle notification...")
        
        # Encode image in memory
        _, buf = cv2.imencode(".jpg", frame_crop)
        img_bytes = buf.tobytes()
        
        timestamp = datetime.now(timezone.utc).isoformat()
        subject = f"🚗 Vehículo rosa detectado — {timestamp}"

        try:
            if settings.SENDGRID_API_KEY:
                self._send_sendgrid(subject, img_bytes)
            else:
                self._send_smtp(subject, img_bytes)
            
            self._last_sent_at = now
            logger.info("Notification sent successfully.")
        except Exception:
            logger.exception("Failed to send notification")

    def _send_smtp(self, subject: str, img_bytes: bytes) -> None:
        if not settings.SMTP_USER or not settings.SMTP_PASSWORD:
            logger.warning("SMTP credentials not configured. Skipping email.")
            return

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = settings.EMAIL_FROM or settings.SMTP_USER
        msg["To"] = settings.EMAIL_TO
        msg.set_content("Se ha detectado un vehículo de color rosa en el stream. Se adjunta captura del frame.")
        
        msg.add_attachment(
            img_bytes, maintype="image", subtype="jpeg", filename="pink_vehicle.jpg"
        )

        logger.info(f"Connecting to SMTP {settings.SMTP_HOST}:{settings.SMTP_PORT}")
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
            server.starttls()
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.send_message(msg)

    def _send_sendgrid(self, subject: str, img_bytes: bytes) -> None:
        logger.info("Using SendGrid API for email.")
        message = Mail(
            from_email=settings.EMAIL_FROM,
            to_emails=settings.EMAIL_TO,
            subject=subject,
            html_content="<strong>Se ha detectado un vehículo de color rosa en el stream. Se adjunta captura.</strong>"
        )
        
        encoded_img = base64.b64encode(img_bytes).decode()
        attachment = Attachment(
            FileContent(encoded_img),
            FileName("pink_vehicle.jpg"),
            FileType("image/jpeg"),
            Disposition("attachment")
        )
        message.attachment = attachment

        sg = SendGridAPIClient(settings.SENDGRID_API_KEY)
        sg.send(message)

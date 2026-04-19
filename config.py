"""
config.py — All configuration via Pydantic Settings loaded from .env

Owns: YOUTUBE_URL, FRAME_RATE, YOLO_MODEL, ROI_POLYGON, pink HSV params,
email credentials, and DEBUG flag.
"""

import json
from typing import Any

from loguru import logger
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # --- Stream ---
    YOUTUBE_URL: str
    FRAME_RATE: int = 5
    YOLO_MODEL: str = "yolov8n.pt"

    # --- ROI ---
    ROI_POLYGON: list[list[int]] = [
        [120, 380],
        [320, 380],
        [320, 480],
        [120, 480],
    ]

    # --- Pink HSV filter ---
    PINK_HUE_LOW: int = 140
    PINK_HUE_HIGH: int = 170
    PINK_SAT_LOW: int = 80
    PINK_VAL_LOW: int = 80
    PINK_PIXEL_THRESHOLD: float = 0.15

    # --- Email ---
    EMAIL_COOLDOWN_SECONDS: int = 60
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    EMAIL_FROM: str = ""
    EMAIL_TO: str = ""
    SENDGRID_API_KEY: str = ""

    # --- Debug ---
    DEBUG: bool = False

    @field_validator("ROI_POLYGON", mode="before")
    @classmethod
    def parse_roi_polygon(cls, v: Any) -> list[list[int]]:
        """Parse ROI_POLYGON from JSON string or validate list directly."""
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"ROI_POLYGON must be valid JSON: {exc}"
                ) from exc
        if not isinstance(v, list) or len(v) < 3:
            raise ValueError(
                "ROI_POLYGON must be a list with at least 3 vertices"
            )
        for point in v:
            if (
                not isinstance(point, (list, tuple))
                or len(point) != 2
            ):
                raise ValueError(
                    f"Each ROI point must be [x, y], got: {point}"
                )
        return v


settings = Settings()

logger.info(
    "Config loaded — YOUTUBE_URL={}, FRAME_RATE={}, YOLO_MODEL={}, "
    "ROI vertices={}, DEBUG={}",
    settings.YOUTUBE_URL,
    settings.FRAME_RATE,
    settings.YOLO_MODEL,
    len(settings.ROI_POLYGON),
    settings.DEBUG,
)

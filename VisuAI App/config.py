"""
Author: Andy Yang
Project: VisuAI
Description: All settings in one place. Secrets (API keys, email password) come from
environment variables or a .env file -- never hard-code them in the source.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env(name, default=None):
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass
class Config:
    # Camera
    camera_index: int
    frame_width: int
    frame_height: int
    horizontal_fov: float          # degrees; check your webcam's spec sheet

    # Detection
    yolo_model: str                # smaller = faster ("yolov10n.pt"), larger = more accurate ("yolov10x.pt")
    confidence: float              # ignore detections below this confidence

    # Narration timing (seconds)
    scene_capture_interval: float  # how often to record what the camera sees
    summary_interval: float        # how often to speak a scene summary

    # Voice
    wake_word: str
    speech_rate: int

    # AI services (optional: the app still works without them, with fewer features)
    openai_api_key: str | None
    openai_model: str
    gemini_api_key: str | None
    gemini_model: str

    # Emergency email (optional)
    smtp_host: str
    smtp_port: int
    smtp_user: str | None
    smtp_password: str | None      # use a Gmail *app password*, not your real password
    emergency_contact: str | None
    user_name: str
    home_location: str | None

    # Web server
    port: int


def load_config() -> Config:
    return Config(
        camera_index=int(_env("CAMERA_INDEX", 0)),
        frame_width=int(_env("FRAME_WIDTH", 1280)),
        frame_height=int(_env("FRAME_HEIGHT", 720)),
        horizontal_fov=float(_env("HORIZONTAL_FOV", 70.0)),
        yolo_model=_env("YOLO_MODEL", "yolov10n.pt"),
        confidence=float(_env("CONFIDENCE", 0.45)),
        scene_capture_interval=float(_env("SCENE_CAPTURE_INTERVAL", 10)),
        summary_interval=float(_env("SUMMARY_INTERVAL", 60)),
        wake_word=_env("WAKE_WORD", "hey vision").lower(),
        speech_rate=int(_env("SPEECH_RATE", 175)),
        openai_api_key=_env("OPENAI_API_KEY"),
        openai_model=_env("OPENAI_MODEL", "gpt-4o-mini"),
        gemini_api_key=_env("GEMINI_API_KEY"),
        gemini_model=_env("GEMINI_MODEL", "gemini-2.5-flash"),
        smtp_host=_env("SMTP_HOST", "smtp.gmail.com"),
        smtp_port=int(_env("SMTP_PORT", 587)),
        smtp_user=_env("SMTP_USER"),
        smtp_password=_env("SMTP_PASSWORD"),
        emergency_contact=_env("EMERGENCY_CONTACT"),
        user_name=_env("USER_NAME", "A VisuAI user"),
        home_location=_env("HOME_LOCATION"),
        port=int(_env("PORT", 5000)),
    )
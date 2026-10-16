"""
Author: Andy Yang
Project: VisuAI
Description: Sends an emergency email with the user's saved location, an approximate
location from their internet connection, and a camera snapshot.
"""
from __future__ import annotations

import logging
import smtplib
import time
from email.message import EmailMessage

import requests

log = logging.getLogger(__name__)


def emergency_configured(cfg) -> bool:
    return bool(cfg.smtp_user and cfg.smtp_password and cfg.emergency_contact)


def approximate_location() -> str | None:
    """City-level location from the IP address. Rough, but better than nothing."""
    try:
        data = requests.get("https://ipinfo.io/json", timeout=3).json()
    except Exception:
        log.warning("Could not look up approximate location")
        return None
    place = ", ".join(p for p in (data.get("city"), data.get("region"), data.get("country")) if p)
    if not place:
        return None
    coords = data.get("loc")
    return f"{place} (coordinates {coords}, estimated from internet connection)" if coords else place


def send_emergency_email(cfg, snapshot_jpeg: bytes | None, approx_location: str | None):
    msg = EmailMessage()
    msg["Subject"] = f"EMERGENCY: {cfg.user_name} needs help"
    msg["From"] = cfg.smtp_user
    msg["To"] = cfg.emergency_contact
    msg.set_content(
        f"{cfg.user_name} triggered an emergency alert from VisuAI at "
        f"{time.strftime('%I:%M %p on %B %d, %Y')}.\n\n"
        f"Saved location: {cfg.home_location or 'not provided'}\n"
        f"Approximate current location: {approx_location or 'unavailable'}\n\n"
        f"{'A camera snapshot from the moment of the alert is attached.' if snapshot_jpeg else ''}\n"
        f"Please contact {cfg.user_name} right away."
    )
    if snapshot_jpeg:
        msg.add_attachment(snapshot_jpeg, maintype="image", subtype="jpeg", filename="snapshot.jpg")

    with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=15) as server:
        server.starttls()
        server.login(cfg.smtp_user, cfg.smtp_password)
        server.send_message(msg)
    log.info("Emergency email sent to %s", cfg.emergency_contact)
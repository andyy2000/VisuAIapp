"""
Author: Andy Yang
Project: VisuAI
Description: Object detection with YOLO, plus the math that turns a bounding box into
something a blind user can act on: direction (clock position), estimated distance,
size, and color. Also watches for vehicles that are close or approaching.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Typical real-world heights (meters) for common COCO classes. Used to estimate distance
# with the pinhole camera model: distance = real_height * focal_length_px / box_height_px.
# These are averages, so distances are rough estimates ("about 3 meters"), not measurements.
TYPICAL_HEIGHT_M = {
    "person": 1.7, "bicycle": 1.0, "car": 1.5, "motorcycle": 1.1, "bus": 3.2,
    "train": 3.8, "truck": 3.0, "traffic light": 0.9, "fire hydrant": 0.8,
    "stop sign": 0.75, "bench": 0.85, "dog": 0.55, "cat": 0.3, "backpack": 0.45,
    "suitcase": 0.6, "bottle": 0.25, "cup": 0.1, "chair": 0.9, "couch": 0.9,
    "potted plant": 0.5, "bed": 0.6, "dining table": 0.75, "toilet": 0.75,
    "tv": 0.6, "laptop": 0.25, "cell phone": 0.15, "microwave": 0.3,
    "refrigerator": 1.8, "sink": 0.25, "book": 0.23, "clock": 0.3,
}

# Objects worth an immediate spoken warning when they are close or approaching.
HAZARD_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "train"}


@dataclass
class Detection:
    name: str
    confidence: float
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 in pixels
    angle_deg: float                          # negative = left of center, positive = right
    distance_m: float | None
    size: str
    color: str

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def clock(self) -> int:
        return clock_position(self.angle_deg)


# ---------- Geometry ----------

def focal_length_px(frame_width: int, horizontal_fov_deg: float) -> float:
    """Focal length in pixels, from the camera's horizontal field of view."""
    return (frame_width / 2) / math.tan(math.radians(horizontal_fov_deg) / 2)


def horizontal_angle(x: float, frame_width: int, focal_px: float) -> float:
    """Angle (degrees) between straight ahead and a point at pixel column x."""
    return math.degrees(math.atan((x - frame_width / 2) / focal_px))


def clock_position(angle_deg: float) -> int:
    """Convert an angle to a clock direction (12 = straight ahead, 3 = right, 9 = left).
    Clock directions are a standard way of giving directions to blind people."""
    hour = round(angle_deg / 30) % 12
    return 12 if hour == 0 else hour


def estimate_distance_m(name: str, box_height_px: float, focal_px: float) -> float | None:
    real_height = TYPICAL_HEIGHT_M.get(name)
    if real_height is None or box_height_px < 1:
        return None
    return real_height * focal_px / box_height_px


def size_label(box, frame_width: int, frame_height: int) -> str:
    x1, y1, x2, y2 = box
    ratio = ((x2 - x1) * (y2 - y1)) / (frame_width * frame_height)
    if ratio < 0.05:
        return "small"
    if ratio < 0.2:
        return "medium"
    return "large"


# ---------- Color ----------

def color_name(frame: np.ndarray, box) -> str:
    """Name the dominant color of an object, using the center of its box
    (the edges of a box usually include background)."""
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, x2 = max(0, x1), min(fw, x2)
    y1, y2 = max(0, y1), min(fh, y2)
    w, h = x2 - x1, y2 - y1
    if w < 2 or h < 2:
        return ""
    region = frame[y1 + h // 4: y2 - h // 4, x1 + w // 4: x2 - w // 4]
    if region.size == 0:
        region = frame[y1:y2, x1:x2]

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hue, sat, val = np.median(hsv, axis=0)   # OpenCV hue runs 0-179

    if val < 50:
        return "black"
    if sat < 45:
        return "white" if val > 190 else "gray"
    if hue < 8 or hue >= 170:
        base = "red"
    elif hue < 20:
        base = "orange"
    elif hue < 33:
        base = "yellow"
    elif hue < 85:
        base = "green"
    elif hue < 100:
        base = "teal"
    elif hue < 130:
        base = "blue"
    elif hue < 150:
        base = "purple"
    else:
        base = "pink"
    if base in ("red", "orange") and val < 120:
        return "brown"
    return base


# ---------- Turning detections into speech ----------

def distance_phrase(distance_m: float | None) -> str:
    if distance_m is None:
        return ""
    if distance_m < 1:
        return "less than a meter away"
    if distance_m < 10:
        meters = round(distance_m)
        return f"about {meters} meter{'s' if meters != 1 else ''} away"
    return "more than 10 meters away"


def with_article(phrase: str) -> str:
    return ("an " if phrase[:1].lower() in "aeiou" else "a ") + phrase


def describe_detection(d: Detection) -> str:
    """e.g. 'large blue car at 12 o'clock, about 4 meters away'"""
    label = " ".join(p for p in (d.size, d.color, d.name) if p)
    text = f"{label} at {d.clock} o'clock"
    dist = distance_phrase(d.distance_m)
    return f"{text}, {dist}" if dist else text


def pluralize(name: str, count: int) -> str:
    if count == 1:
        return f"1 {name}"
    if name == "person":
        return f"{count} people"
    if name.endswith(("s", "sh", "ch")):
        return f"{count} {name}es"
    return f"{count} {name}s"


def local_summary(detections: list[Detection]) -> str:
    """A spoken summary built only from YOLO, so VisuAI still works with no internet AI."""
    if not detections:
        return "I don't see any objects I recognize right now."
    counts = Counter(d.name for d in detections)
    items = [pluralize(name, n) for name, n in counts.most_common(5)]
    listed = items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
    nearest = describe_detection(detections[0])
    return f"I see {listed}. Nearest is {with_article(nearest)}."


def build_detections(frame, result, names, focal_px) -> list[Detection]:
    fh, fw = frame.shape[:2]
    detections = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return detections
    for xyxy, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
        x1, y1, x2, y2 = max(0, xyxy[0]), max(0, xyxy[1]), min(fw, xyxy[2]), min(fh, xyxy[3])
        name = names[int(cls)]
        box = (x1, y1, x2, y2)
        detections.append(Detection(
            name=name,
            confidence=float(conf),
            box=box,
            angle_deg=horizontal_angle((x1 + x2) / 2, fw, focal_px),
            distance_m=estimate_distance_m(name, y2 - y1, focal_px),
            size=size_label(box, fw, fh),
            color=color_name(frame, box),
        ))
    # Nearest first; objects with no distance estimate go last, biggest first.
    detections.sort(key=lambda d: (d.distance_m is None, d.distance_m or 0, -d.height))
    return detections


# ---------- Hazard alerts ----------

class HazardTracker:
    """Warns when a vehicle is roughly in the user's path and either already close
    (fills much of the frame) or approaching (its box is growing quickly)."""

    def __init__(self, cooldown_s=6.0, window_s=1.5, growth=1.25, close_ratio=0.3, path_angle=25.0):
        self.cooldown_s = cooldown_s
        self.window_s = window_s
        self.growth = growth
        self.close_ratio = close_ratio
        self.path_angle = path_angle
        self._history: dict[str, deque] = defaultdict(deque)
        self._last_alert: dict[str, float] = {}

    def update(self, detections: list[Detection], frame_height: int, now: float) -> list[str]:
        biggest: dict[str, Detection] = {}
        for d in detections:
            if d.name in HAZARD_CLASSES and abs(d.angle_deg) <= self.path_angle:
                if d.name not in biggest or d.height > biggest[d.name].height:
                    biggest[d.name] = d

        alerts = []
        for name, d in biggest.items():
            ratio = d.height / frame_height
            history = self._history[name]
            history.append((now, ratio))
            while history and now - history[0][0] > self.window_s:
                history.popleft()

            approaching = len(history) >= 3 and history[0][1] > 0 and ratio / history[0][1] >= self.growth
            close = ratio >= self.close_ratio
            if (approaching or close) and now - self._last_alert.get(name, -1e9) >= self.cooldown_s:
                self._last_alert[name] = now
                state = "approaching" if approaching else "close"
                alerts.append(f"Caution! {name.capitalize()} {state} at {d.clock} o'clock.")

        for name in list(self._history):     # forget objects that left the path
            if name not in biggest:
                del self._history[name]
        return alerts


# ---------- Camera thread ----------

@dataclass
class Snapshot:
    raw_jpeg: bytes | None
    detections: list[Detection]


class CameraWorker(threading.Thread):
    """Reads the camera and runs YOLO continuously in its own thread, so detection
    never waits on speech, the microphone, or AI requests."""

    def __init__(self, cfg, on_hazard):
        super().__init__(daemon=True, name="camera")
        self.cfg = cfg
        self.on_hazard = on_hazard
        self.ready = threading.Event()
        self.error: str | None = None
        self.fps = 0.0
        self.class_names: list[str] = []
        self._lock = threading.Lock()
        self._frame = None
        self._annotated_jpeg = None
        self._detections: list[Detection] = []
        self._stop_event = threading.Event()
        self._hazards = HazardTracker()

    def run(self):
        try:
            from ultralytics import YOLO
            model = YOLO(self.cfg.yolo_model)
            self.class_names = list(model.names.values())
        except Exception as e:
            self.error = f"Could not load the detection model: {e}"
            log.exception(self.error)
            self.ready.set()
            return

        cap = cv2.VideoCapture(self.cfg.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)
        if not cap.isOpened():
            self.error = "Could not open the camera."
            log.error(self.error)
            self.ready.set()
            return

        # Use the resolution the camera actually gave us, not the one we asked for.
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        focal = focal_length_px(width, self.cfg.horizontal_fov)
        log.info("Camera running at %dx%d", width, height)
        self.ready.set()

        last = time.perf_counter()
        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue

                result = model(frame, conf=self.cfg.confidence, verbose=False)[0]
                detections = build_detections(frame, result, model.names, focal)
                ok, buf = cv2.imencode(".jpg", result.plot(), [cv2.IMWRITE_JPEG_QUALITY, 80])

                with self._lock:
                    self._frame = frame
                    self._detections = detections
                    if ok:
                        self._annotated_jpeg = buf.tobytes()

                now = time.perf_counter()
                if now > last:
                    self.fps = 0.9 * self.fps + 0.1 * (1 / (now - last)) if self.fps else 1 / (now - last)
                last = now

                for alert in self._hazards.update(detections, height, time.monotonic()):
                    self.on_hazard(alert)
        finally:
            cap.release()

    def stop(self):
        self._stop_event.set()

    def latest_jpeg(self) -> bytes | None:
        """Frame with boxes drawn, for the web page."""
        with self._lock:
            return self._annotated_jpeg

    def snapshot(self) -> Snapshot | None:
        """Raw frame (no boxes) for the AI models, plus current detections."""
        with self._lock:
            frame, detections = self._frame, list(self._detections)
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Snapshot(raw_jpeg=buf.tobytes() if ok else None, detections=detections)
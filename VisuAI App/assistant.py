"""
Author: Andy Yang
Project: VisuAI
Description: The "brain" of VisuAI.
- Narrator: every few seconds records what YOLO and Gemini see; about once a minute
  GPT turns those notes into one short spoken update.
- Commands: routes what the user says (find / describe / question / emergency).
Everything degrades gracefully: with no API keys, VisuAI still finds objects and
summarizes the scene using YOLO alone.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque

from emergency import approximate_location, emergency_configured, send_emergency_email
from speech import Priority
from vision import describe_detection, local_summary, with_article

log = logging.getLogger(__name__)

SCENE_PROMPT = (
    "You are the eyes of a blind user. Describe this camera image in 2-3 short sentences: "
    "the most important objects and where they are (left, right, straight ahead, near, far), "
    "any people and what they are doing, and anything that could be a hazard. "
    "Be concrete and do not invent details you cannot see."
)

SUMMARY_PROMPT = (
    "You narrate surroundings for a blind user. You get two kinds of notes: [detector] notes "
    "from an object detector (reliable positions as clock directions and rough distances) and "
    "[scene] notes from a vision model (context and activity). Combine them into ONE natural "
    "spoken update under 50 words. Mention anything safety-relevant first. Do not repeat the "
    "previous update unless something changed. If nothing important changed, say so briefly."
)

ROUTER_PROMPT = """You route voice commands for VisuAI, an assistant for blind users.
Reply with JSON only: {"intent": "...", "target": "...", "question": "...", "reply": "..."}
intent must be one of:
- "find": the user wants to locate an object. Set "target" to the closest match from this
  list of detectable objects if one fits, otherwise to the object they named: {classes}
- "describe": the user wants to know what is around them.
- "question": any other question about what the camera sees. Copy it into "question".
- "emergency": the user is hurt, in danger, or urgently asking for help.
- "other": anything else. Put a short, helpful spoken answer in "reply".
"""

INTENTS = {"find", "describe", "question", "emergency", "other"}

# Everyday words mapped to the names YOLO uses.
SYNONYMS = {
    "phone": "cell phone", "cellphone": "cell phone", "mobile": "cell phone",
    "table": "dining table", "sofa": "couch", "television": "tv", "computer": "laptop",
    "people": "person", "someone": "person", "somebody": "person", "man": "person",
    "woman": "person", "bike": "bicycle", "plant": "potted plant", "mug": "cup",
    "fridge": "refrigerator", "bag": "backpack",
}

FIND_PHRASES = ("where is my", "where are my", "where is the", "where are the", "where's my",
                "where's the", "where is", "where are", "help me find", "find my", "find the",
                "find a", "find", "locate", "look for")
EMERGENCY_PHRASES = ("emergency", "help me", "call for help", "i'm hurt", "i am hurt",
                     "i fell", "i'm in danger", "i need help")
DESCRIBE_PHRASES = ("what do you see", "what's around", "what is around", "describe",
                    "look around", "what's in front", "what is in front")
CANCEL_WORDS = ("cancel", "stop", "false alarm", "never mind", "nevermind")


def normalize_target(target: str) -> str:
    target = target.lower().strip(" ?.!,")
    for article in ("my ", "the ", "a ", "an ", "some "):
        if target.startswith(article):
            target = target[len(article):]
    return SYNONYMS.get(target, target)


def matches_target(name: str, target: str) -> bool:
    return name == target or target in name or name in target


def route_by_keywords(text: str) -> dict:
    """Simple command routing used when no OpenAI key is set (or the request fails)."""
    t = text.lower()
    for phrase in FIND_PHRASES:          # checked first so "help me find" isn't an emergency
        if phrase in t:
            target = t.split(phrase, 1)[1]
            if target.strip(" ?.!,"):
                return {"intent": "find", "target": normalize_target(target)}
    if t.strip(" !.") == "help" or any(p in t for p in EMERGENCY_PHRASES):
        return {"intent": "emergency"}
    if any(p in t for p in DESCRIBE_PHRASES):
        return {"intent": "describe"}
    return {"intent": "question", "question": text}


class Assistant:
    def __init__(self, cfg, camera, speaker):
        self.cfg = cfg
        self.camera = camera
        self.speaker = speaker
        self._openai = self._make_openai()
        self._gemini = self._make_gemini()
        self._notes: deque = deque(maxlen=30)        # (kind, text) since the last summary
        self._last_summary = ""
        self._status = "Starting up..."
        self._lock = threading.Lock()
        self._busy = threading.Lock()                # one command at a time
        self._emergency_cancel: threading.Event | None = None

    # ---------- AI services ----------

    def _make_openai(self):
        if not self.cfg.openai_api_key:
            log.warning("OPENAI_API_KEY not set: using keyword commands and detector-only summaries")
            return None
        from openai import OpenAI
        return OpenAI(api_key=self.cfg.openai_api_key, timeout=20)

    def _make_gemini(self):
        if not self.cfg.gemini_api_key:
            log.warning("GEMINI_API_KEY not set: scene descriptions and visual questions disabled")
            return None
        from google import genai
        from google.genai import types
        return genai.Client(api_key=self.cfg.gemini_api_key,
                            http_options=types.HttpOptions(timeout=20_000))

    def _ask_gemini(self, jpeg: bytes | None, prompt: str) -> str | None:
        if not self._gemini or not jpeg:
            return None
        from google.genai import types
        try:
            response = self._gemini.models.generate_content(
                model=self.cfg.gemini_model,
                contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), prompt],
            )
            return (response.text or "").strip() or None
        except Exception:
            log.exception("Gemini request failed")
            return None

    def _ask_openai(self, system: str, user: str, json_mode: bool = False) -> str | None:
        if not self._openai:
            return None
        try:
            extra = {"response_format": {"type": "json_object"}} if json_mode else {}
            response = self._openai.chat.completions.create(
                model=self.cfg.openai_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                **extra,
            )
            return (response.choices[0].message.content or "").strip() or None
        except Exception:
            log.exception("OpenAI request failed")
            return None

    # ---------- Status shown on the web page ----------

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    def _set_status(self, text: str):
        with self._lock:
            self._status = text

    # ---------- Narrator ----------

    def run_narrator(self):
        """Runs forever on its own thread."""
        last_capture = last_summary = time.monotonic()
        while True:
            time.sleep(0.5)
            now = time.monotonic()
            if now - last_capture >= self.cfg.scene_capture_interval:
                last_capture = now
                self._capture_scene()
            if now - last_summary >= self.cfg.summary_interval:
                last_summary = now
                self.speaker.say(self._summarize(), Priority.SUMMARY)

    def _capture_scene(self):
        snap = self.camera.snapshot()
        if snap is None:
            return
        detector_note = local_summary(snap.detections)
        scene_note = self._ask_gemini(snap.raw_jpeg, SCENE_PROMPT)
        with self._lock:
            self._notes.append(("detector", detector_note))
            if scene_note:
                self._notes.append(("scene", scene_note))
        self._set_status(scene_note or detector_note)

    def _summarize(self) -> str | None:
        with self._lock:
            notes = list(self._notes)
            self._notes.clear()
        if not notes:
            return None
        lines = "\n".join(f"[{kind}] {text}" for kind, text in notes)
        summary = self._ask_openai(
            SUMMARY_PROMPT,
            f"Notes since the last update:\n{lines}\n\n"
            f"Previous update (avoid repeating it):\n{self._last_summary or '(none yet)'}",
        )
        if not summary:   # offline fallback: the most recent detector note
            summary = next((text for kind, text in reversed(notes) if kind == "detector"), None)
        if not summary or summary == self._last_summary:
            return None
        self._last_summary = summary
        self._set_status(summary)
        return summary

    # ---------- Commands ----------

    def on_phrase(self, text: str) -> bool:
        """Sees every phrase (no wake word needed). Used to cancel a pending emergency alert."""
        if self._emergency_cancel and any(w in text for w in CANCEL_WORDS):
            self._emergency_cancel.set()
            return True
        return False

    def handle_command(self, text: str):
        # Run on a separate thread so the microphone keeps listening.
        threading.Thread(target=self._handle, args=(text,), daemon=True, name="command").start()

    def _handle(self, text: str):
        if route_by_keywords(text)["intent"] == "emergency":   # never wait behind another task
            self.trigger_emergency()
            return
        if not self._busy.acquire(blocking=False):
            self.speaker.say("One moment, I'm still working on your last request.")
            return
        try:
            route = self.route(text)
            intent = route.get("intent")
            if intent == "find":
                self.find(route.get("target") or text)
            elif intent == "describe":
                self.describe_now()
            elif intent == "question":
                self.answer(route.get("question") or text)
            elif intent == "emergency":
                self.trigger_emergency()
            else:
                self.speaker.say(route.get("reply") or
                                 "Sorry, I didn't catch that. You can ask me to find something, "
                                 "describe what's around you, or say help in an emergency.")
        finally:
            self._busy.release()

    def route(self, text: str) -> dict:
        classes = ", ".join(sorted(set(self.camera.class_names))) or "(model still loading)"
        raw = self._ask_openai(ROUTER_PROMPT.replace("{classes}", classes), text, json_mode=True)
        if raw:
            try:
                data = json.loads(raw)
                if data.get("intent") in INTENTS:
                    if data.get("target"):
                        data["target"] = normalize_target(data["target"])
                    return data
            except json.JSONDecodeError:
                log.warning("Router returned invalid JSON: %s", raw)
        return route_by_keywords(text)

    def find(self, target: str):
        target = normalize_target(target)
        snap = self.camera.snapshot()
        if snap is None:
            self.speaker.say("The camera isn't ready yet.")
            return

        matches = [d for d in snap.detections if matches_target(d.name, target)]
        if matches:   # detections are sorted nearest first
            message = f"I see {with_article(describe_detection(matches[0]))}."
            if len(matches) > 1:
                message += f" There are {len(matches)} in view."
            self.speaker.say(message)
            return

        # YOLO only knows 80 object types (no keys, wallets, etc.), so ask the vision model.
        answer = self._ask_gemini(
            snap.raw_jpeg,
            f"A blind user is looking for: {target}. If it is visible, say where it is in one short "
            "sentence using left, right, straight ahead, near, or far. If not, say you don't see it.",
        )
        self.speaker.say(answer or f"I can't find the {target} right now. "
                                   "Try turning slowly so I can scan the room.")

    def describe_now(self):
        snap = self.camera.snapshot()
        if snap is None:
            self.speaker.say("The camera isn't ready yet.")
            return
        self.speaker.say(self._ask_gemini(snap.raw_jpeg, SCENE_PROMPT) or local_summary(snap.detections))

    def answer(self, question: str):
        snap = self.camera.snapshot()
        if snap is None:
            self.speaker.say("The camera isn't ready yet.")
            return
        reply = self._ask_gemini(
            snap.raw_jpeg,
            f'A blind user asks about what their camera sees: "{question}". Answer in one or two '
            "short spoken sentences. If the image doesn't show enough to answer, say so.",
        )
        if not reply:
            reply = self._ask_openai(
                "Answer a blind user's question in one or two short sentences, using only the "
                "object detector notes provided. If they aren't enough, say so.",
                f"Detector notes: {local_summary(snap.detections)}\nQuestion: {question}",
            )
        self.speaker.say(reply or local_summary(snap.detections))

    # ---------- Emergency ----------

    def trigger_emergency(self):
        if not emergency_configured(self.cfg):
            self.speaker.say("No emergency contact is set up. If you are in danger, call 9 1 1.",
                             Priority.HAZARD)
            return
        if self._emergency_cancel:      # an alert is already counting down
            return

        cancel = threading.Event()
        self._emergency_cancel = cancel
        try:
            self.speaker.say("Sending an emergency alert to your contact. "
                             "Say cancel within 10 seconds to stop.", Priority.HAZARD)
            if cancel.wait(10):
                self.speaker.say("Emergency alert cancelled.", Priority.HAZARD)
                return
            snap = self.camera.snapshot()
            try:
                send_emergency_email(self.cfg, snap.raw_jpeg if snap else None, approximate_location())
                self.speaker.say("Emergency alert sent to your contact. "
                                 "If you are in danger, also call 9 1 1.", Priority.HAZARD)
            except Exception:
                log.exception("Emergency email failed")
                self.speaker.say("I could not send the alert. Please call 9 1 1.", Priority.HAZARD)
        finally:
            self._emergency_cancel = None
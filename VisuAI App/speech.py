"""
Author: Andy Yang
Project: VisuAI
Description: Text-to-speech and wake-word listening, each on its own thread.

- Speaker: a single thread that speaks messages from a priority queue, so a hazard
  warning jumps ahead of a routine scene summary.
- Listener: listens in the background and only acts after the wake word, so the
  camera loop never stops to wait for the microphone.
"""
from __future__ import annotations

import itertools
import logging
import queue
import re
import threading
import time
from enum import IntEnum

log = logging.getLogger(__name__)


class Priority(IntEnum):
    HAZARD = 0      # spoken first
    RESPONSE = 1    # answers to the user's commands
    SUMMARY = 2     # periodic scene summaries


class Speaker(threading.Thread):
    def __init__(self, rate: int = 175):
        super().__init__(daemon=True, name="speaker")
        self.rate = rate
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._order = itertools.count()          # keeps equal-priority messages in order
        self._speaking = threading.Event()
        self._last_finished = 0.0

    def say(self, text: str | None, priority: Priority = Priority.RESPONSE):
        if not text:
            return
        if priority == Priority.SUMMARY and self._queue.qsize() > 1:
            return                                # skip stale summaries instead of piling up
        log.info("SAY [%s]: %s", priority.name, text)
        self._queue.put((priority, next(self._order), text))

    @property
    def is_busy(self) -> bool:
        """True while speaking (or just after), so the microphone ignores our own voice."""
        return self._speaking.is_set() or time.monotonic() - self._last_finished < 1.0

    def run(self):
        try:  # Windows text-to-speech needs COM initialized on this thread
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass
        import pyttsx3

        while True:
            _, _, text = self._queue.get()
            self._speaking.set()
            try:
                # A fresh engine per message avoids pyttsx3's known hang on repeated runAndWait().
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception:
                log.exception("Text-to-speech failed")
            finally:
                self._speaking.clear()
                self._last_finished = time.monotonic()


# Common ways speech recognition mishears "hey vision".
DEFAULT_WAKE_ALIASES = ["hey vision", "hey visions", "hey vision's", "hay vision", "a vision", "hey fishin"]


class Listener:
    def __init__(self, wake_word, speaker, on_command, on_phrase=None, on_wake=None, command_window_s=8.0):
        self.speaker = speaker
        self.on_command = on_command          # called with the command text after the wake word
        self.on_phrase = on_phrase            # called with every phrase first; return True to consume it
        self.on_wake = on_wake                # called when the wake word is heard on its own
        self.command_window_s = command_window_s
        self.aliases = DEFAULT_WAKE_ALIASES if wake_word == "hey vision" else [wake_word]
        self._awaiting_until = 0.0
        self._stop_listening = None

    def start(self) -> bool:
        """Start background listening. Returns False if no microphone is available."""
        try:
            import speech_recognition as sr
            self._sr = sr
            self.recognizer = sr.Recognizer()
            self.recognizer.dynamic_energy_threshold = True
            microphone = sr.Microphone()
            with microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self._stop_listening = self.recognizer.listen_in_background(
                microphone, self._on_audio, phrase_time_limit=8
            )
            log.info("Listening for the wake word")
            return True
        except Exception:
            log.exception("Microphone unavailable; voice commands disabled")
            return False

    def stop(self):
        if self._stop_listening:
            self._stop_listening(wait_for_stop=False)

    def _after_wake_word(self, text: str) -> str | None:
        """Return what was said after the wake word, or None if no wake word.
        Longest variants are checked first and must be whole words, so "hey visions"
        doesn't leave a stray "s" at the start of the command."""
        for alias in sorted(self.aliases, key=len, reverse=True):
            match = re.search(r"\b" + re.escape(alias) + r"\b", text)
            if match:
                return text[match.end():].strip(" ,.!?")
        return None

    def _on_audio(self, recognizer, audio):
        if self.speaker.is_busy:
            return                            # probably heard VisuAI's own voice
        try:
            text = recognizer.recognize_google(audio).lower().strip()
        except self._sr.UnknownValueError:
            return                            # silence or unclear speech
        except self._sr.RequestError as e:
            log.warning("Speech recognition service error: %s", e)
            return
        if not text:
            return
        log.info("HEARD: %s", text)
        self.handle_text(text)

    def handle_text(self, text: str):
        """Wake-word logic, separated from the microphone so it can be unit tested."""
        if self.on_phrase and self.on_phrase(text):
            return

        now = time.monotonic()
        if now < self._awaiting_until:        # user said "hey vision", then paused
            self._awaiting_until = 0.0
            self.on_command(text)
            return

        rest = self._after_wake_word(text)
        if rest is None:
            return
        if rest:                              # "hey vision, find my phone"
            self.on_command(rest)
        else:                                 # "hey vision" ... wait for the command
            self._awaiting_until = now + self.command_window_s
            if self.on_wake:
                self.on_wake()
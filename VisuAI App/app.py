"""
Author: Andy Yang
Project: VisuAI
Description: Flask web app and startup. Starts four background threads once:
  camera (YOLO + hazard alerts), speaker (text-to-speech), listener (wake word),
  and narrator (periodic scene summaries). Web pages only read their latest results.

Run:  python app.py            then open http://127.0.0.1:5000
"""
import argparse
import logging
import threading
import time

from flask import Flask, Response, jsonify, render_template

from assistant import Assistant
from config import load_config
from speech import Listener, Priority, Speaker
from vision import CameraWorker, describe_detection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(threadName)s %(levelname)s: %(message)s")
log = logging.getLogger("visuai")

GREETING = ("Hello, I'm VisuAI. Say hey vision, then ask me to find something, describe what's "
            "around you, or ask a question. In an emergency, say hey vision, help.")


def create_app(cfg):
    speaker = Speaker(rate=cfg.speech_rate)
    camera = CameraWorker(cfg, on_hazard=lambda text: speaker.say(text, Priority.HAZARD))
    assistant = Assistant(cfg, camera, speaker)
    listener = Listener(
        cfg.wake_word, speaker,
        on_command=assistant.handle_command,
        on_phrase=assistant.on_phrase,
        on_wake=lambda: speaker.say("I'm listening.", Priority.RESPONSE),
    )

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/camera")
    def camera_page():
        return render_template("camera.html")

    @app.route("/video_feed")
    def video_feed():
        def stream():
            last = None
            while True:
                jpeg = camera.latest_jpeg()
                if jpeg is None or jpeg is last:   # no new frame yet
                    time.sleep(0.03)
                    continue
                last = jpeg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/status")
    def status():
        snap = camera.snapshot()
        return jsonify(
            description=assistant.status,
            objects=[describe_detection(d) for d in snap.detections[:5]] if snap else [],
            fps=round(camera.fps, 1),
            error=camera.error,
        )

    def start_workers():
        speaker.start()
        camera.start()
        threading.Thread(target=assistant.run_narrator, daemon=True, name="narrator").start()
        speaker.say(GREETING)
        if not listener.start():
            speaker.say("Voice commands are unavailable because no microphone was found.")

    return app, start_workers


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description="VisuAI: AI assistant for visually impaired users")
    parser.add_argument("--camera", type=int, default=cfg.camera_index, help="webcam index")
    parser.add_argument("--model", default=cfg.yolo_model, help="YOLO weights, e.g. yolov10n.pt")
    parser.add_argument("--fov", type=float, default=cfg.horizontal_fov, help="horizontal field of view (degrees)")
    parser.add_argument("--webcam-resolution", type=int, nargs=2, default=[cfg.frame_width, cfg.frame_height])
    parser.add_argument("--port", type=int, default=cfg.port)
    args = parser.parse_args()

    cfg.camera_index, cfg.yolo_model, cfg.horizontal_fov, cfg.port = args.camera, args.model, args.fov, args.port
    cfg.frame_width, cfg.frame_height = args.webcam_resolution

    app, start_workers = create_app(cfg)
    start_workers()
    log.info("Open http://127.0.0.1:%d in your browser", cfg.port)
    # use_reloader=False: the debug reloader would start the camera twice.
    app.run(host="127.0.0.1", port=cfg.port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
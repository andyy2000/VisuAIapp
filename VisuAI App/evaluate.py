"""
Author: Andy Yang
Project: VisuAI
Description: Measures detection accuracy and speed, so the resume can cite real numbers.

1. Accuracy: label 100-200 of your own frames (e.g. in Roboflow), export in YOLO format,
   then run:   python evaluate.py --data path/to/data.yaml
2. Speed:      python evaluate.py --speed      (times the model on your webcam)
"""
import argparse
import time

import cv2
from ultralytics import YOLO


def accuracy(model, data, conf):
    metrics = model.val(data=data, conf=conf, verbose=False)
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall:    {metrics.box.mr:.3f}")
    print(f"mAP@50:    {metrics.box.map50:.3f}")


def speed(model, camera_index, frames=100):
    cap = cv2.VideoCapture(camera_index)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read from the camera.")
    model(frame, verbose=False)                       # warm-up run
    start = time.perf_counter()
    for _ in range(frames):
        ok, frame = cap.read()
        if ok:
            model(frame, verbose=False)
    elapsed = time.perf_counter() - start
    cap.release()
    print(f"{frames / elapsed:.1f} frames per second ({1000 * elapsed / frames:.0f} ms per frame)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov10n.pt")
    parser.add_argument("--data", help="YOLO data.yaml for your labeled test set")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--speed", action="store_true")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    yolo = YOLO(args.model)
    if args.data:
        accuracy(yolo, args.data, args.conf)
    if args.speed:
        speed(yolo, args.camera)
    if not args.data and not args.speed:
        parser.print_help()
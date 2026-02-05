import os
import time
import argparse
import threading

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import torch
from PIL import Image
from vision import CameraCapture


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live open-vocabulary detection (Hugging Face Grounding DINO)"
    )
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--infer-width", type=int, default=640)
    parser.add_argument("--infer-height", type=int, default=360)
    parser.add_argument(
        "--model",
        type=str,
        default="IDEA-Research/grounding-dino-tiny",
        help="Hugging Face model id",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="stylo, cahier, colle, regle, gomme, pen, notebook, glue stick, ruler, eraser",
        help="Comma-separated labels (open-vocabulary prompt)",
    )
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--min-area", type=int, default=2500)
    parser.add_argument("--max-dets", type=int, default=20)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cpu"],
        help="Inference device",
    )
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument(
        "--infer-fps",
        type=float,
        default=2.0,
        help="Max inference rate (0 disables cap)",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=15.0,
        help="Cap display loop to this FPS (0 disables)",
    )
    return parser.parse_args()


def pick_device(device_arg: str) -> str:
    if device_arg == "mps":
        return "mps"
    if device_arg == "cpu":
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_detector(model_id: str, device: str):
    try:
        from transformers import pipeline
    except Exception as exc:
        raise RuntimeError(
            "transformers not installed. Run: pip install transformers"
        ) from exc

    # pipeline accepts device="mps" on recent versions of transformers
    return pipeline("zero-shot-object-detection", model=model_id, device=device)


def draw_detections(frame, detections):
    for det in detections:
        box = det["box"]
        xmin, ymin, xmax, ymax = int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])
        label = det.get("label", "object")
        score = det.get("score", 0.0)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 200, 255), 2)
        text = f"{label} {score:.2f}"
        cv2.putText(
            frame,
            text,
            (xmin, max(0, ymin - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
        )
    return frame


def filter_detections(detections, min_area, max_dets):
    out = []
    for det in detections:
        box = det["box"]
        w = max(0, int(box["xmax"]) - int(box["xmin"]))
        h = max(0, int(box["ymax"]) - int(box["ymin"]))
        if w * h < min_area:
            continue
        out.append(det)
    out.sort(key=lambda d: d.get("score", 0.0), reverse=True)
    return out[:max_dets]


def main():
    args = parse_args()
    device = pick_device(args.device)

    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    if not labels:
        raise SystemExit("No labels provided")

    cap = CameraCapture(args.camera_id, args.width, args.height)
    if not cap.open():
        raise SystemExit(1)

    detector = build_detector(args.model, device)
    print(f"✅ Model: {args.model}")
    print(f"✅ Device: {device}")

    cv2.namedWindow("detections", cv2.WINDOW_NORMAL)
    cv2.namedWindow("controls", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("box_thresh(%)", "controls", int(args.box_threshold * 100), 100, lambda v: None)
    cv2.createTrackbar("text_thresh(%)", "controls", int(args.text_threshold * 100), 100, lambda v: None)
    cv2.createTrackbar("min_area", "controls", args.min_area, 20000, lambda v: None)
    cv2.createTrackbar("max_dets", "controls", args.max_dets, 50, lambda v: None)

    lock = threading.Lock()
    shared = {
        "frame": None,
        "dets": [],
        "stop": False,
        "busy": False,
        "infer_fps": 0.0,
        "last_infer": 0.0,
    }

    def infer_loop():
        infer_count = 0
        t0 = time.perf_counter()
        while True:
            if shared["stop"]:
                return

            with lock:
                if shared["busy"] or shared["frame"] is None:
                    frame = None
                else:
                    frame = shared["frame"]
                    shared["busy"] = True

            if frame is None:
                time.sleep(0.005)
                continue

            # Optional inference FPS cap
            if args.infer_fps and args.infer_fps > 0:
                now = time.perf_counter()
                min_dt = 1.0 / args.infer_fps
                if now - shared["last_infer"] < min_dt:
                    with lock:
                        shared["busy"] = False
                    time.sleep(0.002)
                    continue

            box_thr = cv2.getTrackbarPos("box_thresh(%)", "controls") / 100.0
            text_thr = cv2.getTrackbarPos("text_thresh(%)", "controls") / 100.0
            min_area = cv2.getTrackbarPos("min_area", "controls")
            max_dets = max(1, cv2.getTrackbarPos("max_dets", "controls"))

            infer_w = args.infer_width or args.width
            infer_h = args.infer_height or args.height
            if infer_w != args.width or infer_h != args.height:
                resized = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
            else:
                resized = frame

            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)

            dets = detector(
                image,
                candidate_labels=labels,
                box_threshold=box_thr,
                text_threshold=text_thr,
            )

            if infer_w != args.width or infer_h != args.height:
                scale_x = args.width / infer_w
                scale_y = args.height / infer_h
                for det in dets:
                    box = det["box"]
                    box["xmin"] = box["xmin"] * scale_x
                    box["xmax"] = box["xmax"] * scale_x
                    box["ymin"] = box["ymin"] * scale_y
                    box["ymax"] = box["ymax"] * scale_y

            dets = filter_detections(dets, min_area, max_dets)

            with lock:
                shared["dets"] = dets
                shared["busy"] = False
                shared["last_infer"] = time.perf_counter()

            infer_count += 1
            dt = shared["last_infer"] - t0
            if dt > 0.5:
                with lock:
                    shared["infer_fps"] = infer_count / dt
                infer_count = 0
                t0 = shared["last_infer"]

    thread = threading.Thread(target=infer_loop, daemon=True)
    thread.start()
    frame_count = 0
    t0 = time.perf_counter()
    fps = 0.0

    try:
        while True:
            loop_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            if frame_count % max(args.stride, 1) == 0:
                with lock:
                    shared["frame"] = frame.copy()

            out = frame.copy()
            with lock:
                dets = list(shared["dets"])
                infer_fps = shared["infer_fps"]
            out = draw_detections(out, dets)

            t1 = time.perf_counter()
            dt = t1 - t0
            if dt > 0.5:
                fps = frame_count / dt
                t0 = t1
                frame_count = 0

            cv2.putText(
                out,
                f"FPS: {fps:.1f} | infer: {infer_fps:.1f} | device: {device}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.imshow("detections", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Simple FPS cap to reduce load
            if args.target_fps and args.target_fps > 0:
                elapsed = time.perf_counter() - loop_start
                target_dt = 1.0 / args.target_fps
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)
    finally:
        shared["stop"] = True
        cap.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

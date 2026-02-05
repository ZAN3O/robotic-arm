import cv2
from vision import CameraCapture
from perception import ObjectDetector, ObjectColor, ObjectShape

# ---- Tuning knobs (more selective) ----
CAMERA_ID = 0
FRAME_W, FRAME_H = 640, 480
ROI = (int(FRAME_W * 0.1), int(FRAME_H * 0.25), int(FRAME_W * 0.8), int(FRAME_H * 0.7))  # x, y, w, h
MIN_BBOX_AREA = 1400
MIN_DIM = 20
MIN_CONFIDENCE = 0.75
MIN_ASPECT = 0.4
MAX_ASPECT = 2.5
ALLOWED_COLORS = {ObjectColor.RED, ObjectColor.GREEN, ObjectColor.BLUE, ObjectColor.YELLOW}


def filter_objects(objs, min_area, min_dim, min_conf, min_aspect, max_aspect):
    filtered = []
    for obj in objs:
        x, y, w, h = obj.bbox
        if w * h < min_area:
            continue
        if min(w, h) < min_dim:
            continue
        if obj.confidence < min_conf:
            continue
        if obj.shape == ObjectShape.UNKNOWN:
            continue
        if obj.color not in ALLOWED_COLORS:
            continue
        aspect = w / h if h else 0
        if aspect < min_aspect or aspect > max_aspect:
            continue
        filtered.append(obj)
    return filtered


def offset_objects(objs, dx, dy):
    for obj in objs:
        x, y, w, h = obj.bbox
        obj.bbox = (x + dx, y + dy, w, h)
        cx, cy = obj.center_px
        obj.center_px = (cx + dx, cy + dy)
    return objs


cap = CameraCapture(CAMERA_ID, FRAME_W, FRAME_H)
if not cap.open():
    raise SystemExit(1)

det = ObjectDetector()

# Trackbars for live tuning
cv2.namedWindow("controls", cv2.WINDOW_NORMAL)
cv2.createTrackbar("min_area", "controls", MIN_BBOX_AREA, 10000, lambda v: None)
cv2.createTrackbar("min_dim", "controls", MIN_DIM, 100, lambda v: None)
cv2.createTrackbar("min_conf(%)", "controls", int(MIN_CONFIDENCE * 100), 100, lambda v: None)
cv2.createTrackbar("min_aspect(%)", "controls", int(MIN_ASPECT * 100), 300, lambda v: None)
cv2.createTrackbar("max_aspect(%)", "controls", int(MAX_ASPECT * 100), 400, lambda v: None)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    x0, y0, w0, h0 = ROI
    roi = frame[y0:y0 + h0, x0:x0 + w0]

    objs = det.detect_objects(roi)
    min_area = cv2.getTrackbarPos("min_area", "controls")
    min_dim = cv2.getTrackbarPos("min_dim", "controls")
    min_conf = cv2.getTrackbarPos("min_conf(%)", "controls") / 100.0
    min_aspect = cv2.getTrackbarPos("min_aspect(%)", "controls") / 100.0
    max_aspect = cv2.getTrackbarPos("max_aspect(%)", "controls") / 100.0
    if max_aspect < min_aspect:
        max_aspect = min_aspect

    objs = filter_objects(objs, min_area, min_dim, min_conf, min_aspect, max_aspect)
    objs = offset_objects(objs, x0, y0)

    out = det.draw_detections(frame, objs)
    cv2.rectangle(out, (x0, y0), (x0 + w0, y0 + h0), (255, 255, 255), 1)
    cv2.imshow("detections", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.close()
cv2.destroyAllWindows()

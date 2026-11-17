"""Face and eye detection with the Haar cascades bundled in opencv-python 4.x (no training, no downloads).

Classes: "face", "eye_left", "eye_right" (left/right as seen in the image).
Scores: cascades have no probability; the stage weight from detectMultiScale3 is squashed with a
logistic function into 0..1. Treat it as a ranking score, not a calibrated confidence.
Haar cascades are trained on photos of frontal human faces; they do not work on anime or side views.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

from detector import DetectionBox


def _cv2():
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError('OpenCV is needed for detection: pip install "opencv-python-headless>=4.8,<5"') from e
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError("This OpenCV build has no Haar cascades (OpenCV 5 moved them out). Install opencv-python-headless<5.")
    return cv2


def _score(weight: float) -> float:
    return round(1.0 / (1.0 + math.exp(-float(weight))), 4)


def detect(image_path: str | Path, min_face: int = 60) -> List[DetectionBox]:
    cv2 = _cv2()
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"cannot read image: {image_path}")
    gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    faces_cc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eyes_cc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    boxes: List[DetectionBox] = []
    faces, _, face_w = faces_cc.detectMultiScale3(gray, 1.1, 5, minSize=(min_face, min_face), outputRejectLevels=True)
    for (x, y, w, h), fw in zip(faces, face_w.ravel() if len(faces) else []):
        boxes.append(DetectionBox(len(boxes) + 1, "face", _score(fw), float(x), float(y), float(x + w), float(y + h)))
        # Eyes sit in the upper 60 % of a face box and are at most a fifth of the face wide.
        roi = gray[y:y + int(h * 0.6), x:x + w]
        eyes, _, eye_w = eyes_cc.detectMultiScale3(roi, 1.05, 3, minSize=(w // 10, w // 10),
                                                   maxSize=(w // 5, w // 5), outputRejectLevels=True)
        best = {}
        for (ex, ey, ew, eh), weight in zip(eyes, eye_w.ravel() if len(eyes) else []):
            side = "eye_left" if ex + ew / 2 < w / 2 else "eye_right"
            if side not in best or weight > best[side][1]:
                best[side] = ((ex, ey, ew, eh), weight)
        for side, ((ex, ey, ew, eh), weight) in sorted(best.items()):
            boxes.append(DetectionBox(len(boxes) + 1, side, _score(weight),
                                      float(x + ex), float(y + ey), float(x + ex + ew), float(y + ey + eh)))
    return boxes


def draw(image_path: str | Path, out_path: str | Path, predictions: List[DetectionBox],
         ground_truth: List[DetectionBox] | None = None) -> Path:
    """Predictions as solid boxes with labels, ground truth as thin white boxes."""
    cv2 = _cv2()
    image = cv2.imread(str(image_path))
    colors = {"face": (229, 70, 79), "eye_left": (112, 124, 14), "eye_right": (12, 65, 194)}
    for g in ground_truth or []:
        cv2.rectangle(image, (int(g.x1), int(g.y1)), (int(g.x2), int(g.y2)), (255, 255, 255), 1, cv2.LINE_AA)
    placed = []  # label rectangles already drawn, to keep labels from covering each other

    def overlaps(r):
        return any(r[0] < q[2] and q[0] < r[2] and r[1] < q[3] and q[1] < r[3] for q in placed)

    for b in sorted(predictions, key=lambda b: b.x1):
        c = colors.get(b.class_name, (0, 200, 255))
        cv2.rectangle(image, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), c, 2, cv2.LINE_AA)
        label = f"{b.class_name} {b.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        above = (int(b.x1), int(b.y1) - th - 6, int(b.x1) + tw + 6, int(b.y1))
        below = (int(b.x1), int(b.y2), int(b.x1) + tw + 6, int(b.y2) + th + 6)
        r = below if overlaps(above) or above[1] < 0 else above
        placed.append(r)
        cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), c, -1)
        cv2.putText(image, label, (r[0] + 3, r[3] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    out = Path(out_path)
    cv2.imwrite(str(out), image)
    return out

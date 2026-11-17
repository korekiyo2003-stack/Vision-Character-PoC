"""
Vision Character Feature Detection & Inspection PoC

Detector-agnostic post-processing and evaluation layer: takes raw boxes from any
detector (e.g. a YOLO model via `boxes_from_ultralytics`), applies confidence
filtering and per-class NMS, and scores the result against ground-truth boxes
(precision / recall / F1 at an IoU threshold). No model is bundled.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class DetectionBox:
    box_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"box {self.box_id}: expected x1<=x2 and y1<=y2, got {self}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"box {self.box_id}: confidence {self.confidence} outside 0..1")


@dataclass
class DetectionReport:
    image_name: str
    total_detections: int
    mean_confidence: float
    classes_detected: Dict[str, int]
    boxes: List[DetectionBox]
    latency_ms: float  # post-processing time only; model inference is not included
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_summary_markdown(self) -> str:
        lines = [
            f"### Detection Report: {self.image_name}",
            f"- **Total Objects Detected**: {self.total_detections}",
            f"- **Mean Confidence**: {self.mean_confidence:.3f}",
            f"- **Post-processing Time**: {self.latency_ms:.2f} ms",
            "",
            "| Class | Count | Avg Confidence |",
            "| :--- | :---: | :---: |"
        ]
        for cname, count in self.classes_detected.items():
            avg_conf = sum(b.confidence for b in self.boxes if b.class_name == cname) / max(count, 1)
            lines.append(f"| `{cname}` | {count} | {avg_conf:.3f} |")
        return "\n".join(lines)


class CharacterDetector:
    def __init__(self, confidence_threshold: float = 0.50, iou_threshold: float = 0.45):
        self.conf_thresh = confidence_threshold
        self.iou_thresh = iou_threshold

    def calculate_iou(self, b1: DetectionBox, b2: DetectionBox) -> float:
        ix1 = max(b1.x1, b2.x1)
        iy1 = max(b1.y1, b2.y1)
        ix2 = min(b1.x2, b2.x2)
        iy2 = min(b1.y2, b2.y2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        union = b1.area + b2.area - inter
        return inter / union if union > 0 else 0.0

    def apply_nms(self, boxes: List[DetectionBox]) -> List[DetectionBox]:
        """Apply Non-Maximum Suppression to remove overlapping redundant predictions."""
        if not boxes:
            return []
        sorted_boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
        selected: List[DetectionBox] = []

        while sorted_boxes:
            best = sorted_boxes.pop(0)
            selected.append(best)
            sorted_boxes = [b for b in sorted_boxes if b.class_name != best.class_name or self.calculate_iou(best, b) < self.iou_thresh]

        return selected

    def run_detection(self, image_path: str | Path, candidate_boxes: Optional[List[DetectionBox]] = None) -> DetectionReport:
        t0 = time.perf_counter()
        p = Path(image_path)

        # Boxes come from the caller's detector (see boxes_from_ultralytics); none is bundled.
        raw_boxes = candidate_boxes or []
        filtered = [b for b in raw_boxes if b.confidence >= self.conf_thresh]
        final_boxes = self.apply_nms(filtered)

        class_counts: Dict[str, int] = {}
        for b in final_boxes:
            class_counts[b.class_name] = class_counts.get(b.class_name, 0) + 1

        mean_conf = sum(b.confidence for b in final_boxes) / len(final_boxes) if final_boxes else 0.0
        latency = (time.perf_counter() - t0) * 1000

        return DetectionReport(
            image_name=p.name,
            total_detections=len(final_boxes),
            mean_confidence=round(mean_conf, 3),
            classes_detected=class_counts,
            boxes=final_boxes,
            latency_ms=round(latency, 2),
            status="PASSED" if final_boxes else "NO_OBJECTS"
        )


def boxes_from_ultralytics(result: Any) -> List[DetectionBox]:
    """Convert one ultralytics `Results` object (YOLOv8/YOLO11 `model(img)[i]`) to DetectionBox list."""
    names = result.names
    out: List[DetectionBox] = []
    xyxy = result.boxes.xyxy.tolist()
    confs = result.boxes.conf.tolist()
    classes = result.boxes.cls.tolist()
    for i, ((x1, y1, x2, y2), conf, cls) in enumerate(zip(xyxy, confs, classes), start=1):
        out.append(DetectionBox(box_id=i, class_name=str(names[int(cls)]), confidence=float(conf),
                                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)))
    return out


@dataclass
class ClassMetrics:
    class_name: str
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        d = self.true_positives + self.false_positives
        return self.true_positives / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.true_positives + self.false_negatives
        return self.true_positives / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def evaluate(predictions: Sequence[DetectionBox], ground_truth: Sequence[DetectionBox],
             iou_threshold: float = 0.5) -> Dict[str, ClassMetrics]:
    """Greedy one-to-one matching per class, highest confidence first (VOC/COCO style)."""
    iou = CharacterDetector().calculate_iou
    classes = sorted({b.class_name for b in predictions} | {b.class_name for b in ground_truth})
    metrics: Dict[str, ClassMetrics] = {}
    for cname in classes:
        preds = sorted((b for b in predictions if b.class_name == cname), key=lambda b: b.confidence, reverse=True)
        gts = [b for b in ground_truth if b.class_name == cname]
        used = [False] * len(gts)
        tp = fp = 0
        for p in preds:
            best_j, best_iou = -1, iou_threshold
            for j, g in enumerate(gts):
                if not used[j]:
                    v = iou(p, g)
                    if v >= best_iou:
                        best_j, best_iou = j, v
            if best_j >= 0:
                used[best_j] = True
                tp += 1
            else:
                fp += 1
        metrics[cname] = ClassMetrics(cname, tp, fp, len(gts) - sum(used))
    return metrics


def metrics_markdown(metrics: Dict[str, ClassMetrics], iou_threshold: float = 0.5) -> str:
    lines = [f"| Class | TP | FP | FN | Precision | Recall | F1 (IoU>={iou_threshold}) |",
             "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for m in metrics.values():
        lines.append(f"| `{m.class_name}` | {m.true_positives} | {m.false_positives} | {m.false_negatives} | "
                     f"{m.precision:.3f} | {m.recall:.3f} | {m.f1:.3f} |")
    return "\n".join(lines)

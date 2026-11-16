"""
Vision Character Feature Detection & Inspection PoC
Extracts object/facial bounding boxes and confidence evaluation reports for vision verification pipelines.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
        return self.width * self.height


@dataclass
class DetectionReport:
    image_name: str
    total_detections: int
    mean_confidence: float
    classes_detected: Dict[str, int]
    boxes: List[DetectionBox]
    latency_ms: float
    status: str

    def to_summary_markdown(self) -> str:
        lines = [
            f"### Detection Accuracy Report: {self.image_name}",
            f"- **Total Objects Detected**: {self.total_detections}",
            f"- **Mean Confidence**: {self.mean_confidence:.3f}",
            f"- **Inference Latency**: {self.latency_ms:.1f} ms",
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
        import time
        t0 = time.time()
        p = Path(image_path)

        # In production this calls YOLOv8 (ultralytics); here we provide the deterministic evaluation layer
        raw_boxes = candidate_boxes or []
        filtered = [b for b in raw_boxes if b.confidence >= self.conf_thresh]
        final_boxes = self.apply_nms(filtered)

        class_counts: Dict[str, int] = {}
        for b in final_boxes:
            class_counts[b.class_name] = class_counts.get(b.class_name, 0) + 1

        mean_conf = sum(b.confidence for b in final_boxes) / len(final_boxes) if final_boxes else 0.0
        latency = (time.time() - t0) * 1000

        return DetectionReport(
            image_name=p.name,
            total_detections=len(final_boxes),
            mean_confidence=round(mean_conf, 3),
            classes_detected=class_counts,
            boxes=final_boxes,
            latency_ms=round(latency, 2),
            status="PASSED" if final_boxes else "NO_OBJECTS"
        )

"""Command line: detect (or load) boxes for one image, filter them, and score them against ground truth.

    python detect.py photo.jpg                                   OpenCV face/eye detection, summary
    python detect.py photo.jpg --gt photo.gt.json --draw out.png precision/recall + annotated image
    python detect.py photo.jpg --pred yolo_boxes.json --gt gt.json   boxes from any other detector
    python detect.py photo.jpg --json report.json                full report as JSON

Box files are JSON lists: [{"class": "face", "box": [x1, y1, x2, y2], "score": 0.9}, ...]
("score" is optional in ground-truth files).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from detector import CharacterDetector, DetectionBox, evaluate, metrics_markdown


def load_boxes(path: Path, default_score: float = 1.0) -> List[DetectionBox]:
    items = json.loads(path.read_text(encoding="utf-8"))
    return [DetectionBox(i, str(d["class"]), float(d.get("score", default_score)), *map(float, d["box"]))
            for i, d in enumerate(items, start=1)]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="detect.py", description="検出・NMS・正解データとの照合（適合率・再現率・F1）")
    p.add_argument("image", type=Path)
    p.add_argument("--pred", type=Path, help="other detector's boxes (JSON); default: OpenCV Haar face/eye detection")
    p.add_argument("--gt", type=Path, help="ground-truth boxes (JSON) to compute precision / recall / F1")
    p.add_argument("--conf", type=float, default=0.5, help="confidence threshold (default 0.5)")
    p.add_argument("--nms", type=float, default=0.45, help="NMS IoU threshold (default 0.45)")
    p.add_argument("--iou", type=float, default=0.5, help="IoU needed to count as a match (default 0.5)")
    p.add_argument("--draw", type=Path, help="write an annotated copy of the image (needs OpenCV)")
    p.add_argument("--json", type=Path, help="write the full report as JSON")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.image.exists():
        print(f"image not found: {args.image}", file=sys.stderr)
        return 2
    try:
        if args.pred:
            raw = load_boxes(args.pred)
        else:
            import haar
            raw = haar.detect(args.image)
    except (RuntimeError, ValueError, KeyError, json.JSONDecodeError) as e:
        print(str(e), file=sys.stderr)
        return 2

    report = CharacterDetector(args.conf, args.nms).run_detection(args.image, raw)
    print(report.to_summary_markdown())
    data = report.to_dict()

    gt = load_boxes(args.gt) if args.gt else None
    if gt is not None:
        metrics = evaluate(report.boxes, gt, args.iou)
        print("\n" + metrics_markdown(metrics, args.iou))
        data["evaluation"] = {k: {"tp": m.true_positives, "fp": m.false_positives, "fn": m.false_negatives,
                                  "precision": round(m.precision, 4), "recall": round(m.recall, 4), "f1": round(m.f1, 4)}
                              for k, m in metrics.items()}
    if args.draw:
        try:
            import haar
            print(f"\nannotated image: {haar.draw(args.image, args.draw, report.boxes, gt)}")
        except RuntimeError as e:
            print(f"\n(no drawing: {e})", file=sys.stderr)
    if args.json:
        args.json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Test suite for Character Feature Detection PoC.
Validates NMS suppression, confidence thresholding, report output, box validation,
and precision/recall evaluation against ground truth.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from detector import CharacterDetector, DetectionBox, boxes_from_ultralytics, evaluate, metrics_markdown


def box(i, cls, conf, x1, y1, x2, y2):
    return DetectionBox(box_id=i, class_name=cls, confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2)


def run_tests():
    detector = CharacterDetector(confidence_threshold=0.50, iou_threshold=0.45)

    # Hand-made detector output: one duplicate eye (NMS) and one low-confidence face (threshold)
    candidates = [
        box(1, "eye_left", 0.92, 40, 50, 70, 80),
        box(2, "eye_left", 0.78, 42, 51, 71, 81),
        box(3, "eye_right", 0.89, 100, 52, 130, 82),
        box(4, "face", 0.96, 30, 30, 145, 160),
        box(5, "face", 0.35, 35, 35, 140, 155),
    ]
    report = detector.run_detection("sample_character_01.png", candidate_boxes=candidates)
    print(report.to_summary_markdown())

    assert report.total_detections == 3
    assert report.classes_detected == {"eye_left": 1, "face": 1, "eye_right": 1}
    assert {b.box_id for b in report.boxes} == {1, 3, 4}
    assert report.mean_confidence > 0.90
    data = json.loads(report.to_json())
    assert data["total_detections"] == 3 and len(data["boxes"]) == 3

    # Same-class boxes that barely overlap must both survive NMS; other classes never suppress each other
    kept = detector.apply_nms([box(1, "eye", 0.9, 0, 0, 10, 10), box(2, "eye", 0.8, 8, 8, 18, 18),
                               box(3, "face", 0.7, 0, 0, 10, 10)])
    assert {b.box_id for b in kept} == {1, 2, 3}

    # Invalid boxes are rejected instead of producing negative areas / IoU
    for bad in [(5, 0, 0, 10), (0, 0, 10, 10, 1.5)]:
        try:
            if len(bad) == 4:
                box(9, "x", 0.5, *bad)
            else:
                box(9, "x", bad[4], *bad[:4])
            raise AssertionError(f"invalid box accepted: {bad}")
        except ValueError:
            pass

    # Evaluation against ground truth
    ground_truth = [
        box(1, "face", 1.0, 32, 28, 146, 158),
        box(2, "eye_left", 1.0, 41, 49, 71, 79),
        box(3, "eye_right", 1.0, 160, 52, 190, 82),  # detector put its right eye in the wrong place
    ]
    metrics = evaluate(report.boxes, ground_truth, iou_threshold=0.5)
    print("\n" + metrics_markdown(metrics))
    assert (metrics["face"].true_positives, metrics["face"].false_positives) == (1, 0)
    assert metrics["eye_left"].recall == 1.0
    m = metrics["eye_right"]
    assert (m.true_positives, m.false_positives, m.false_negatives) == (0, 1, 1)
    assert m.precision == 0.0 and m.recall == 0.0 and m.f1 == 0.0

    # A duplicate that survives NMS counts as a false positive, not a second hit
    dup = evaluate([box(1, "face", 0.9, 30, 30, 145, 160), box(2, "face", 0.8, 31, 31, 146, 161)],
                   [box(1, "face", 1.0, 30, 30, 145, 160)])
    assert (dup["face"].true_positives, dup["face"].false_positives) == (1, 1)

    # ultralytics adapter, with a stand-in for a Results object
    tensor = lambda rows: SimpleNamespace(tolist=lambda: rows)
    fake = SimpleNamespace(names={0: "face", 1: "eye"},
                           boxes=SimpleNamespace(xyxy=tensor([[1, 2, 30, 40], [5, 5, 9, 9]]),
                                                 conf=tensor([0.91, 0.66]), cls=tensor([0.0, 1.0])))
    converted = boxes_from_ultralytics(fake)
    assert [(b.class_name, b.confidence, b.x2) for b in converted] == [("face", 0.91, 30.0), ("eye", 0.66, 9.0)]

    print("\n>>> ALL VISION POC TESTS PASSED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    run_tests()

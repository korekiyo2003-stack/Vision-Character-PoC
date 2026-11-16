"""
Test suite for Character Feature Detection PoC.
Validates NMS suppression, confidence thresholding, and structured report output.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from detector import CharacterDetector, DetectionBox


def run_tests():
    detector = CharacterDetector(confidence_threshold=0.50, iou_threshold=0.45)

    # Simulated model output with overlapping bounding boxes (NMS test) and low confidence noise
    mock_candidates = [
        DetectionBox(box_id=1, class_name="eye_left", confidence=0.92, x1=40, y1=50, x2=70, y2=80),
        DetectionBox(box_id=2, class_name="eye_left", confidence=0.78, x1=42, y1=51, x2=71, y2=81), # duplicate to be suppressed
        DetectionBox(box_id=3, class_name="eye_right", confidence=0.89, x1=100, y1=52, x2=130, y2=82),
        DetectionBox(box_id=4, class_name="face", confidence=0.96, x1=30, y1=30, x2=145, y2=160),
        DetectionBox(box_id=5, class_name="face", confidence=0.35, x1=35, y1=35, x2=140, y2=155),   # below threshold
    ]

    report = detector.run_detection("sample_character_01.png", candidate_boxes=mock_candidates)

    print("=== DETECTION ACCURACY REPORT ===")
    print(report.to_summary_markdown())

    # Assertions
    assert report.total_detections == 3, f"Expected 3 detections after NMS and threshold, got {report.total_detections}"
    assert "face" in report.classes_detected and report.classes_detected["face"] == 1
    assert "eye_left" in report.classes_detected and report.classes_detected["eye_left"] == 1
    assert "eye_right" in report.classes_detected and report.classes_detected["eye_right"] == 1
    assert report.mean_confidence > 0.90, "Mean confidence should exceed 0.90"

    print("\n>>> ALL VISION POC TESTS PASSED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    run_tests()

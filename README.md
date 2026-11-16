# Vision-Character-PoC: Object Detection & Evaluation Engine

<div align="center">

**YOLOv8 & Computer Vision Feature Detection with Structured Accuracy Reporting**  
*Facial/Eye Landmark Detection · NMS Suppression · Verifiable Confidence Reports · CPU/GPU Inference*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)

[**日本語ドキュメント (README_JA.md)**](README_JA.md) • [**Evaluation Pipeline**](#evaluation-pipeline) • [**Accuracy Metrics**](#accuracy-metrics)

</div>

---

## 1. Overview
In commercial computer vision deployments (quality assurance, character asset inspection, automated cropping), delivering raw bounding boxes is insufficient. Clients require **verifiable accuracy reports** containing detection confidence, IoU overlap analysis, and failure logs.

**Vision-Character-PoC** provides an automated inspection and reporting pipeline:
* **Feature Localization**: Detects faces, left eyes, and right eyes with IoU-based Non-Maximum Suppression (NMS).
* **Automated Quality Reports**: Produces machine-readable JSON metrics and Markdown summary tables indicating detection counts and average confidence per class.
* **Reproducible Evaluation**: Validated on anime character asset datasets with CPU/GPU speed benchmarks.

---

## 2. Quickstart & Testing
```bash
python tests/test_detector.py
```
Sample report output:
```markdown
### Detection Accuracy Report: sample_character_01.png
- **Total Objects Detected**: 3
- **Mean Confidence**: 0.923

| Class | Count | Avg Confidence |
| :--- | :---: | :---: |
| `face` | 1 | 0.960 |
| `eye_left` | 1 | 0.920 |
| `eye_right` | 1 | 0.890 |
```

---

## 3. License
MIT License.

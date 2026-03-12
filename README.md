# Adaptive Video Intrusion Detection with Policy-Based Threshold Selection

This repository is a minimal public demo of my undergraduate capstone / research-oriented project on adaptive video intrusion and anomaly detection for surveillance scenes.

## Overview

The project explores whether a policy model can dynamically adjust detection thresholds according to scene dynamics, instead of relying on a fixed confidence/IoU setting.

The current prototype integrates:

- YOLO-based anomaly/person detection
- PPO-style adaptive threshold selection
- Fallback control for robustness
- ROI-based intrusion reasoning
- Temporal alert triggering
- Structured logging for diagnosis

## Why it matters

A fixed threshold is often brittle in real surveillance video.  
This project focuses on adaptive decision-making and system robustness, especially when the learned policy becomes unstable or collapses to a narrow action regime.

## Public Demo Contents

- `demo_intrusion_minimal.py` — minimal demo script
- `PROJECT_SUMMARY.md` — one-page project summary
- `assets/demo_video.mp4` — short demo video
- `assets/results_summary.png` — system overview and sample outputs

## Run

```bash
python demo_intrusion_minimal.py
Dependencies
See requirements.txt.

Public Release Scope
This repository contains only a minimal public demo.

Due to thesis submission and publication considerations, the following are not fully released at this stage:

full datasets
trained model weights
some training / preprocessing components
complete experiment logs
The current release is intended for project presentation and academic communication only.

Notes
This public version should be viewed as a compact demonstration prototype rather than a full benchmark release.

If helpful, I would be glad to share a short walkthrough or additional project details upon request.

Contact
Jie Wei
15655138691@163.com

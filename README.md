# Traffic Detection Pipeline

A deep-learning-powered real-time object detection and tracking system for traffic footage. The system supports two model architectures (YOLOv8s and SSDLite320), a PyQt6 desktop GUI with live inference, and a headless CLI pipeline for batch processing. It was developed as a final project for AI2002 at FAST-NUCES.

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Directory Structure](#directory-structure)
- [Detected Classes](#detected-classes)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Dataset](#dataset)
- [Team Contributions](#team-contributions)

---

## Project Overview

Raw CCTV or dashcam video is ingested, run through a trained object detection model, tracked across frames using IoU-based multi-object tracking, and displayed through either an interactive GUI or saved as an annotated output video. Per-frame statistics including class counts, average confidence, and FPS are logged to JSON at the end of each session.

---

## System Architecture

The system is organized into six loosely coupled layers:

| Layer | Responsibility | Key Modules |
|---|---|---|
| Data Ingestion | Load local video files and RTSP streams | `video_loader.py`, `stream_handler.py` |
| Data Processing | Frame extraction, augmentation | `augmentor.py` |
| Model Training | Training loop, validation, checkpointing | `trainer.py`, `ssd_trainer.py`, `validator.py` |
| Inference Engine | Per-frame detection, tracking, post-processing | `detector.py`, `tracker.py`, `postprocessor.py` |
| Analytics | Per-frame and session statistics | `stats_collector.py` |
| Presentation | Desktop GUI | `main_window.py`, `video_widget.py`, `controls.py`, `stats_panel.py` |

---

## Directory Structure

```
traffic-detection/
├── configs/
│   └── dataset.yaml
├── data/
│   └── data.yaml
├── evaluation_outputs/
│   └── plots/
├── notebooks/
│   ├── Final-Notebook.ipynb
│   └── Phase-I.ipynb
├── scripts/
│   ├── compare_models.py
│   ├── evaluate.py
│   ├── evaluate_ssd.py
│   ├── generate_report.py
│   ├── merge_datasets.py
│   ├── train.py
│   └── train_ssd.py
├── src/
│   ├── analytics/
│   │   └── stats_collector.py
│   ├── dataset/
│   │   └── voc_dataset.py
│   ├── gui/
│   │   ├── controls.py
│   │   ├── inference_worker.py
│   │   ├── main_window.py
│   │   ├── stats_panel.py
│   │   └── video_widget.py
│   ├── inference/
│   │   ├── detector.py
│   │   ├── exporter.py
│   │   ├── postprocessor.py
│   │   └── tracker.py
│   ├── ingestion/
│   │   ├── stream_handler.py
│   │   └── video_loader.py
│   ├── models/
│   │   └── base_detector.py
│   ├── processing/
│   │   └── augmentor.py
│   └── training/
│       ├── evaluator.py
│       ├── hyperparams.yaml
│       ├── ssd_hyperparams.yaml
│       ├── ssd_trainer.py
│       ├── trainer.py
│       └── validator.py
├── weights/
│   ├── yolo_best.pt
│   └── ssd_best.pt
├── tests/
├── environment-gpu.yml
├── environment-cpu.yml
├── requirements.txt
├── run_demo.py
└── run_gui.py
```

---

## Detected Classes

The system detects ten object classes:

`bike`, `bus`, `car`, `motor`, `person`, `rider`, `traffic light`, `traffic sign`, `train`, `truck`

---

## Installation

### Prerequisites

- Windows 10/11 x86-64
- NVIDIA GPU with CUDA 11.8 support (for GPU environment)
- Miniconda or Anaconda

### GPU Environment (recommended)

```
conda env create -f environment-gpu.yml --prefix C:\path\to\your\envs\traffic-detection
conda activate C:\path\to\your\envs\traffic-detection
```

### CPU Environment

```
pip install -r requirements.txt
```

### Important Note for Windows + Conda

Do not install `pyside6`, `qt6-main`, or any conda-forge Qt package into this environment. All Qt binaries are bundled inside the pip-installed `PyQt6` wheel. Mixing conda-forge Qt packages with the pip PyQt6 wheel causes DLL conflicts on Windows that prevent the GUI from launching.

### Weights

Place trained model weights in the `weights/` directory before running either the GUI or the CLI:

```
weights/
├── yolo_best.pt
└── ssd_best.pt
```

---

## Usage

### Desktop GUI

```
python run_gui.py
```

Once the window opens:

1. Click **Upload Video** and select an `.mp4`, `.avi`, `.mkv`, or `.mov` file.
2. Select a model from the **Model** dropdown (`yolo` or `ssd`). The model can be switched mid-playback.
3. Adjust the **Confidence** slider to filter detections in real time.
4. Use the **Seek** slider to jump to any frame. Tracker IDs reset on each seek.
5. Use the **Speed** dropdown to adjust playback rate (0.5x to 2x).
6. Click **Play/Pause** to pause and resume.
7. The right-hand sidebar shows live per-class counts, average confidence, and FPS.
8. A session statistics JSON is saved to `evaluation_outputs/gui_session_stats.json` when playback ends or the window is closed.

### Headless CLI

```
python run_demo.py --model yolo --video path/to/video.mp4
python run_demo.py --model ssd  --video path/to/video.mp4
```

Output files are written to the project root:

- `demo_output_yolo.mp4` — annotated video
- `demo_stats_yolo.json` — per-frame statistics log

### Model Export

To export trained YOLOv8s weights to ONNX or TorchScript:

```python
from src.inference.exporter import export_model

export_model(weights_path="weights/yolo_best.pt", export_format="onnx")
export_model(weights_path="weights/yolo_best.pt", export_format="torchscript")
```

---

## Model Performance

Models were evaluated on a validation split of 1,661 images.

### Detection Accuracy

| Metric | YOLOv8s | SSDLite320 |
|---|---|---|
| mAP@0.5 | 0.4744 | 0.1025 |
| mAP@0.5:0.95 | 0.2749 | 0.0483 |
| Precision | 0.6730 | 0.0970 |
| Recall | 0.4357 | 0.0373 |

### Inference Speed

| Metric | YOLOv8s | SSDLite320 |
|---|---|---|
| FPS | 76.75 | 142.51 |
| Latency (ms) | 13.03 | 7.02 |

### Per-Class AP@0.5

| Class | YOLOv8s | SSDLite320 |
|---|---|---|
| Bus | 0.8033 | 0.2495 |
| Car | 0.7705 | 0.1368 |
| Traffic Sign | 0.6153 | 0.0056 |
| Truck | 0.5764 | 0.0851 |

YOLOv8s is the recommended model for all operational use. While SSDLite320 achieves higher raw inference speed, its detection accuracy is insufficient for safety-critical traffic monitoring. YOLOv8s at 76.75 FPS exceeds the 30 FPS threshold required for real-time video processing by a significant margin.

---

## Dataset

| Phase | Dataset | Purpose |
|---|---|---|
| Phase I | BDD100K subset | Foundation training on four primary traffic classes |
| Phase II | BDD100K + UA-DETRAC | Robustness improvement using high-density traffic camera footage |

All data was ingested and standardized via Roboflow. A custom `merge_datasets.py` script maps class IDs from both datasets into a unified 10-class system and synchronizes image/label pairs.



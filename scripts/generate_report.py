"""
    python scripts/generate_report.py --out-dir evaluation_outputs/plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


YOLO_EVAL    = Path("evaluation_outputs/yolov8s_80ep_evaluation.json")
SSD_EVAL     = Path("evaluation_outputs/ssd_evaluation.json")
COMPARISON   = Path("evaluation_outputs/comparison_report.json")
SSD_LOG      = Path("runs/ssd/bdd100k_ssdlite_100ep/training_log.json")
YOLO_CSV     = Path("runs/train/bdd100k_yolov8s_80ep/results.csv")

YOLO_COLOUR  = "#1565C0"
SSD_COLOUR   = "#B71C1C"
TRAIN_ALPHA  = 0.85
VAL_ALPHA    = 0.5


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}\n"
            "Make sure you ran evaluate.py, evaluate_ssd.py, and compare_models.py first."
        )
    with open(path) as f:
        return json.load(f)


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / name
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {dest}")


def plot_ssd_loss_curve(out_dir: Path) -> None:
    print("[1/5] SSD loss curve")
    log = _load_json(SSD_LOG)
    history = log["history"]

    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_loss, color=SSD_COLOUR,   alpha=TRAIN_ALPHA,
            linewidth=2, label="Train Loss")
    ax.plot(epochs, val_loss,   color=SSD_COLOUR,   alpha=VAL_ALPHA,
            linewidth=2, linestyle="--", label="Val Loss")

    best_epoch = min(history, key=lambda h: h["val_loss"])
    ax.axvline(best_epoch["epoch"], color="grey", linestyle=":",
               linewidth=1.2, label=f"Best epoch {best_epoch['epoch']} (val={best_epoch['val_loss']:.4f})")

    ax.set_title("SSDLite320 — Training & Validation Loss", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "ssd_loss_curve.png")


def plot_yolo_loss_curve(out_dir: Path) -> None:
    print("[2/5] YOLOv8 loss curve")
    if not YOLO_CSV.exists():
        print(f"  WARNING: {YOLO_CSV} not found — skipping YOLOv8 loss curve.")
        return

    import csv
    rows = []
    with open(YOLO_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v.strip() for k, v in row.items()})

    if not rows:
        print("  WARNING: results.csv is empty — skipping.")
        return

    def _col(rows, *candidates):
        for c in candidates:
            if c in rows[0]:
                return [float(r[c]) for r in rows]
        return None

    epochs     = list(range(1, len(rows) + 1))
    train_loss = _col(rows, "train/box_loss", "train/box_om")
    val_loss   = _col(rows, "val/box_loss",   "val/box_om")

    if train_loss is None or val_loss is None:
        print(f"  WARNING: Could not locate loss columns in results.csv.")
        print(f"  Available columns: {list(rows[0].keys())}")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_loss, color=YOLO_COLOUR, alpha=TRAIN_ALPHA,
            linewidth=2, label="Train Box Loss")
    ax.plot(epochs, val_loss,   color=YOLO_COLOUR, alpha=VAL_ALPHA,
            linewidth=2, linestyle="--", label="Val Box Loss")

    ax.set_title("YOLOv8s — Training & Validation Loss", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "yolo_loss_curve.png")


def plot_metric_comparison(out_dir: Path) -> None:
    print("[3/5] Metric comparison bar chart")
    comp = _load_json(COMPARISON)

    metrics    = ["map50", "map50_95", "precision", "recall"]
    labels     = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall"]
    yolo_vals  = [comp["metrics"][m]["yolov8s"] for m in metrics]
    ssd_vals   = [comp["metrics"][m]["ssdlite"]  for m in metrics]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_y = ax.bar(x - width/2, yolo_vals, width, label="YOLOv8s",
                    color=YOLO_COLOUR, alpha=0.85)
    bars_s = ax.bar(x + width/2, ssd_vals,  width, label="SSDLite320",
                    color=SSD_COLOUR,  alpha=0.85)

    for bar in list(bars_y) + list(bars_s):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("YOLOv8s vs SSDLite320 — Detection Metrics",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "metric_comparison.png")


def plot_per_class_ap(out_dir: Path) -> None:
    print("[4/5] Per-class AP@0.5 comparison")
    comp    = _load_json(COMPARISON)
    pc      = comp["per_class"]
    classes = sorted(pc.keys())

    yolo_ap = [pc[c]["yolov8s_ap50"] for c in classes]
    ssd_ap  = [pc[c]["ssdlite_ap50"] for c in classes]

    x     = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, yolo_ap, width, label="YOLOv8s",
           color=YOLO_COLOUR, alpha=0.85)
    ax.bar(x + width/2, ssd_ap,  width, label="SSDLite320",
           color=SSD_COLOUR,  alpha=0.85)

    ax.set_title("Per-Class AP@0.5 — YOLOv8s vs SSDLite320",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("AP@0.5")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "per_class_ap_comparison.png")


def plot_speed_comparison(out_dir: Path) -> None:
    print("[5/5] Speed comparison")
    comp     = _load_json(COMPARISON)
    yolo_fps = comp["metrics"]["fps"]["yolov8s"]
    ssd_fps  = comp["metrics"]["fps"]["ssdlite"]
    yolo_ms  = comp["metrics"]["latency_ms"]["yolov8s"]
    ssd_ms   = comp["metrics"]["latency_ms"]["ssdlite"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    for ax, vals, title, ylabel in [
        (ax1, [yolo_fps, ssd_fps], "Frames Per Second (higher is better)", "FPS"),
        (ax2, [yolo_ms,  ssd_ms ], "Latency per Image (lower is better)",  "ms"),
    ]:
        colours = [YOLO_COLOUR, SSD_COLOUR]
        bars    = ax.bar(["YOLOv8s", "SSDLite320"], vals, color=colours, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Inference Speed Comparison", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "speed_comparison.png")


def print_summary(out_dir: Path) -> None:
    comp = _load_json(COMPARISON)
    print("\n=== Report Summary ===")
    print(f"Recommended model : {comp['recommendation']}")
    counts = comp["overall_winner_count"]
    print(f"YOLOv8s wins      : {counts['yolov8s']} / {len(comp['metrics'])} metrics")
    print(f"SSDLite wins      : {counts['ssdlite']} / {len(comp['metrics'])} metrics")
    print(f"\nPlots saved to    : {out_dir.resolve()}")
    plots = list(out_dir.glob("*.png"))
    for p in sorted(plots):
        print(f"  {p.name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate model comparison report plots")
    p.add_argument("--out-dir", default="evaluation_outputs/plots",
                   help="Directory to save plots")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir)

    print(f"Output directory: {out_dir}\n")

    plot_ssd_loss_curve(out_dir)
    plot_yolo_loss_curve(out_dir)
    plot_metric_comparison(out_dir)
    plot_per_class_ap(out_dir)
    plot_speed_comparison(out_dir)
    print_summary(out_dir)


if __name__ == "__main__":
    main()
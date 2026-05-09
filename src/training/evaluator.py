"""
evaluator.py

Rich evaluation utilities for the traffic detection model.
Reads Ultralytics results.csv, generates plots, and produces
a clean comparison table across multiple model runs.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Class metadata 

CLASS_NAMES = [
    "bike", "bus", "car", "motor", "person",
    "rider", "traffic light", "traffic sign", "train", "truck"
]

CLASS_COLOURS = {
    "bike":          "#26C6DA",
    "bus":           "#1976D2",
    "car":           "#1565C0",
    "motor":         "#00ACC1",
    "person":        "#B71C1C",
    "rider":         "#EF6C00",
    "traffic light": "#F9A825",
    "traffic sign":  "#2E7D32",
    "train":         "#6A1B9A",
    "truck":         "#0D47A1",
}

# Results CSV reader 

def read_results_csv(results_dir: str) -> list[dict]:
    """
    Read Ultralytics results.csv into a list of epoch dicts.

    Args:
        results_dir: Path to the run folder containing results.csv
                     e.g. runs/detect/runs/train/bdd100k_yolov8s_30ep

    Returns:
        List of dicts, one per epoch, with cleaned column names as keys.
    """
    csv_path = Path(results_dir) / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"results.csv not found in {results_dir}")

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from keys and values
            cleaned = {k.strip(): v.strip() for k, v in row.items()}
            rows.append(cleaned)
    return rows


def extract_metric(rows: list[dict], column: str) -> list[float]:
    """Extract a single metric column across all epochs."""
    return [float(r[column]) for r in rows if column in r]


# Training curve plots 

def plot_training_curves(
    results_dir: str,
    output_dir: str = None,
    show: bool = False,
) -> str:
    """
    Generate a 2x3 training curves figure from results.csv.

    Plots:
      - Box loss (train & val)
      - Class loss (train & val)
      - DFL loss (train & val)
      - mAP@0.5
      - mAP@0.5:0.95
      - Precision & Recall

    Args:
        results_dir: Run folder containing results.csv
        output_dir:  Where to save the figure (defaults to results_dir)
        show:        If True, display the plot interactively

    Returns:
        Path to saved figure.
    """
    rows       = read_results_csv(results_dir)
    epochs     = list(range(1, len(rows) + 1))
    out_path   = Path(output_dir or results_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Training Curves — BDD100K YOLOv8s", fontsize=14, fontweight="bold")

    def _plot(ax, col_train, col_val=None, title="", ylabel=""):
        train_vals = extract_metric(rows, col_train)
        ax.plot(epochs, train_vals, color="#1565C0", linewidth=2, label="Train")
        if col_val:
            val_vals = extract_metric(rows, col_val)
            ax.plot(epochs, val_vals, color="#B71C1C",
                    linewidth=2, linestyle="--", label="Val")
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, len(epochs))

    _plot(axes[0][0],
          "train/box_loss", "val/box_loss",
          "Box Loss", "Loss")
    _plot(axes[0][1],
          "train/cls_loss", "val/cls_loss",
          "Classification Loss", "Loss")
    _plot(axes[0][2],
          "train/dfl_loss", "val/dfl_loss",
          "DFL Loss", "Loss")

    # mAP@0.5
    map50 = extract_metric(rows, "metrics/mAP50(B)")
    axes[1][0].plot(epochs, map50, color="#2E7D32", linewidth=2)
    axes[1][0].set_title("mAP @ 0.5", fontsize=10, fontweight="bold")
    axes[1][0].set_xlabel("Epoch", fontsize=8)
    axes[1][0].set_ylabel("mAP", fontsize=8)
    axes[1][0].grid(True, alpha=0.3)
    axes[1][0].set_xlim(1, len(epochs))
    axes[1][0].axhline(y=max(map50), color="#2E7D32",
                        linestyle=":", alpha=0.5,
                        label=f"Best: {max(map50):.4f}")
    axes[1][0].legend(fontsize=8)

    # mAP@0.5:0.95
    map5095 = extract_metric(rows, "metrics/mAP50-95(B)")
    axes[1][1].plot(epochs, map5095, color="#F9A825", linewidth=2)
    axes[1][1].set_title("mAP @ 0.5:0.95", fontsize=10, fontweight="bold")
    axes[1][1].set_xlabel("Epoch", fontsize=8)
    axes[1][1].set_ylabel("mAP", fontsize=8)
    axes[1][1].grid(True, alpha=0.3)
    axes[1][1].set_xlim(1, len(epochs))
    axes[1][1].axhline(y=max(map5095), color="#F9A825",
                        linestyle=":", alpha=0.5,
                        label=f"Best: {max(map5095):.4f}")
    axes[1][1].legend(fontsize=8)

    # Precision & Recall
    precision = extract_metric(rows, "metrics/precision(B)")
    recall    = extract_metric(rows, "metrics/recall(B)")
    axes[1][2].plot(epochs, precision, color="#1565C0",
                    linewidth=2, label="Precision")
    axes[1][2].plot(epochs, recall, color="#B71C1C",
                    linewidth=2, linestyle="--", label="Recall")
    axes[1][2].set_title("Precision & Recall", fontsize=10, fontweight="bold")
    axes[1][2].set_xlabel("Epoch", fontsize=8)
    axes[1][2].set_ylabel("Score", fontsize=8)
    axes[1][2].legend(fontsize=8)
    axes[1][2].grid(True, alpha=0.3)
    axes[1][2].set_xlim(1, len(epochs))

    plt.tight_layout()
    save_path = out_path / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"  Training curves saved → {save_path}")
    return str(save_path)


# Per-class bar chart 

def plot_per_class_metrics(
    per_class_results: dict,
    output_dir: str,
    show: bool = False,
) -> str:
    """
    Generate a grouped bar chart of per-class AP@0.5, Precision, Recall.

    Args:
        per_class_results: Dict from validator.validate() keyed by class name.
        output_dir:        Where to save the figure.
        show:              Display interactively.

    Returns:
        Path to saved figure.
    """
    classes   = [c for c in per_class_results if not c.startswith("_")]
    ap50_vals = [per_class_results[c]["ap50"]      for c in classes]
    prec_vals = [per_class_results[c]["precision"]  for c in classes]
    rec_vals  = [per_class_results[c]["recall"]     for c in classes]

    x     = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, ap50_vals, width, label="AP@0.5",    color="#1565C0", alpha=0.85)
    ax.bar(x,         prec_vals, width, label="Precision",  color="#2E7D32", alpha=0.85)
    ax.bar(x + width, rec_vals,  width, label="Recall",     color="#B71C1C", alpha=0.85)

    ax.set_title("Per-Class Detection Metrics — YOLOv8s @ 30 Epochs",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Value labels on bars
    for bars in [
        ax.containers[0], ax.containers[1], ax.containers[2]
    ]:
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)

    plt.tight_layout()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "per_class_metrics.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"  Per-class chart saved  → {save_path}")
    return str(save_path)


# Summary JSON 

def save_evaluation_summary(
    results: dict,
    model_name: str,
    output_dir: str,
) -> str:
    """
    Save evaluation results as a structured JSON file.

    Args:
        results:    Output from validator.validate()
        model_name: e.g. 'yolov8s_30ep'
        output_dir: Where to write the JSON

    Returns:
        Path to saved JSON.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / f"{model_name}_evaluation.json"

    summary = {
        "model":   model_name,
        "summary": results.get("_summary", {}),
        "per_class": {
            k: v for k, v in results.items()
            if not k.startswith("_")
        },
    }
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Evaluation JSON saved  → {save_path}")
    return str(save_path)


# Model comparison table 

def print_comparison_table(summaries: list[dict]) -> None:
    """
    Print a formatted comparison table across multiple model runs.

    Args:
        summaries: List of dicts, each with keys:
                   model, mAP50, mAP50_95, precision, recall, fps
    """
    print("\n" + "="*70)
    print("  MODEL COMPARISON TABLE")
    print("="*70)
    print(f"  {'Model':<30} {'mAP@0.5':>8} {'mAP@0.5:0.95':>13} "
          f"{'P':>7} {'R':>7} {'FPS':>7}")
    print("  " + "-"*65)
    for s in summaries:
        print(f"  {s['model']:<30} {s['mAP50']:>8.4f} "
              f"{s['mAP50_95']:>13.4f} "
              f"{s['precision']:>7.4f} {s['recall']:>7.4f} "
              f"{s.get('fps', 0):>7.1f}")
    print("="*70 + "\n")
"""
scripts/evaluate.py

Standalone evaluation script. Run this after training to get
full metrics, plots, and a saved JSON summary.

Usage:
    python scripts/evaluate.py --weights weights/best.pt
    python scripts/evaluate.py --weights weights/best.pt --show-plots
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.training.validator import validate
from src.training.evaluator import (
    plot_training_curves,
    plot_per_class_metrics,
    print_comparison_table,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained traffic detection model"
    )
    parser.add_argument("--weights",     required=True)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--dataset",     default="configs/dataset.yaml")
    parser.add_argument("--output-dir",  default="evaluation_outputs")
    parser.add_argument("--show-plots",  action="store_true")
    parser.add_argument("--model-name",  default=None)
    return parser.parse_args()


def _measure_fps(weights: str, dataset: str, n_warmup: int = 5) -> tuple[float, float]:
    """
    Run a small timing loop using Ultralytics to get real FPS and latency.
    Returns (fps, latency_ms).
    """
    try:
        from ultralytics import YOLO
        model  = YOLO(weights)
        source = Path(dataset).parent.parent / "master_yolo_dataset" / "valid" / "images"
        if not source.exists():
            return 0.0, 0.0

        images = list(source.glob("*.jpg"))[:50]
        if not images:
            return 0.0, 0.0

        for img in images[:n_warmup]:
            model(str(img), verbose=False)

        t0 = time.perf_counter()
        for img in images:
            model(str(img), verbose=False)
        elapsed = time.perf_counter() - t0

        n          = len(images)
        fps        = round(n / elapsed, 2)
        latency_ms = round((elapsed / n) * 1000, 2)
        return fps, latency_ms
    except Exception:
        return 0.0, 0.0


def _save_flat_json(
    results: dict,
    model_name: str,
    fps: float,
    latency_ms: float,
    output_dir: str,
) -> None:
    summary  = results.get("_summary", {})
    per_class_raw = {
        k: v for k, v in results.items() if k != "_summary"
    }

    per_class = {
        cls: {"ap50": round(float(vals.get("ap50", 0.0)), 4)}
        for cls, vals in per_class_raw.items()
    }

    output = {
        "model":      model_name,
        "weights":    "",
        "split":      "valid",
        "num_images": 0,
        "map50":      round(float(summary.get("mAP50",     0.0)), 4),
        "map50_95":   round(float(summary.get("mAP50_95",  0.0)), 4),
        "precision":  round(float(summary.get("precision", 0.0)), 4),
        "recall":     round(float(summary.get("recall",    0.0)), 4),
        "fps":        fps,
        "latency_ms": latency_ms,
        "per_class":  per_class,
    }

    out_path = Path(output_dir) / "yolov8s_80ep_evaluation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved → {out_path}")


def main():
    args       = parse_args()
    model_name = args.model_name or Path(args.weights).stem

    print("\n" + "=" * 55)
    print("  Traffic Detection — Full Evaluation")
    print("=" * 55)
    print(f"  Model      : {model_name}")
    print(f"  Weights    : {args.weights}")
    print(f"  Output dir : {args.output_dir}")
    print("=" * 55 + "\n")

    print("── Step 1: Running model inference on val set ──")
    results = validate(
        model_path=args.weights,
        dataset_config=args.dataset,
    )

    if args.results_dir:
        print("\n── Step 2: Generating training curves ──────────")
        try:
            plot_training_curves(
                results_dir=args.results_dir,
                output_dir=args.output_dir,
                show=args.show_plots,
            )
        except FileNotFoundError as e:
            print(f"  Skipping training curves: {e}")
    else:
        print("\n── Step 2: Skipped (no --results-dir provided) ─")

    print("\n── Step 3: Generating per-class bar chart ──────")
    plot_per_class_metrics(
        per_class_results=results,
        output_dir=args.output_dir,
        show=args.show_plots,
    )

    print("\n── Step 4: Measuring FPS ───────────────────────")
    fps, latency_ms = _measure_fps(args.weights, args.dataset)
    print(f"  FPS        : {fps}")
    print(f"  Latency ms : {latency_ms}")

    print("\n── Step 5: Saving flat evaluation JSON ─────────")
    _save_flat_json(results, model_name, fps, latency_ms, args.output_dir)

    print("\n── Step 6: Summary table ────────────────────────")
    summary = results.get("_summary", {})
    print_comparison_table([{
        "model":     model_name,
        "mAP50":     summary.get("mAP50",    0),
        "mAP50_95":  summary.get("mAP50_95", 0),
        "precision": summary.get("precision", 0),
        "recall":    summary.get("recall",   0),
        "fps":       fps,
    }])

    print(f"\n  All outputs saved to: {args.output_dir}/")
    print("  Done.\n")


if __name__ == "__main__":
    main()
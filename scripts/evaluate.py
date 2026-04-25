"""
scripts/evaluate.py

Standalone evaluation script. Run this after training to get
full metrics, plots, and a saved JSON summary.

Usage:
    # Evaluate our trained model
    python scripts/evaluate.py \
        --weights weights/bdd100k_yolov8s_30ep_best.pt \
        --results-dir "runs/detect/runs/train/bdd100k_yolov8s_30ep"

    # Evaluate and show plots interactively
    python scripts/evaluate.py \
        --weights weights/bdd100k_yolov8s_30ep_best.pt \
        --results-dir "runs/detect/runs/train/bdd100k_yolov8s_30ep" \
        --show-plots
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.training.validator import validate
from src.training.evaluator import (
    plot_training_curves,
    plot_per_class_metrics,
    save_evaluation_summary,
    print_comparison_table,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained traffic detection model"
    )
    parser.add_argument(
        "--weights",     required=True,
        help="Path to .pt weights file"
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Run folder containing results.csv for training curves"
    )
    parser.add_argument(
        "--dataset",     default="configs/dataset.yaml",
        help="Path to dataset.yaml"
    )
    parser.add_argument(
        "--output-dir",  default="evaluation_outputs",
        help="Where to save plots and JSON"
    )
    parser.add_argument(
        "--show-plots",  action="store_true",
        help="Display plots interactively"
    )
    parser.add_argument(
        "--model-name",  default=None,
        help="Name for this model in outputs (defaults to weights filename)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_name = args.model_name or Path(args.weights).stem

    print("\n" + "="*55)
    print("  Traffic Detection — Full Evaluation")
    print("="*55)
    print(f"  Model      : {model_name}")
    print(f"  Weights    : {args.weights}")
    print(f"  Output dir : {args.output_dir}")
    print("="*55 + "\n")

    # 1. Run model evaluation 
    print("── Step 1: Running model inference on val set ──")
    results = validate(
        model_path=args.weights,
        dataset_config=args.dataset,
    )

    # 2. Training curves 
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

    # 3. Per-class bar chart 
    print("\n── Step 3: Generating per-class bar chart ──────")
    plot_per_class_metrics(
        per_class_results=results,
        output_dir=args.output_dir,
        show=args.show_plots,
    )

    # 4. Save JSON summary
    print("\n── Step 4: Saving evaluation JSON ──────────────")
    save_evaluation_summary(
        results=results,
        model_name=model_name,
        output_dir=args.output_dir,
    )

    # 5. Comparison table
    print("\n── Step 5: Summary table ────────────────────────")
    summary = results.get("_summary", {})
    print_comparison_table([{
        "model":      model_name,
        "mAP50":      summary.get("mAP50", 0),
        "mAP50_95":   summary.get("mAP50_95", 0),
        "precision":  summary.get("precision", 0),
        "recall":     summary.get("recall", 0),
        "fps":        1000 / 8.5,  # from inference speed (8.5ms/img)
    }])

    print(f"\n  All outputs saved to: {args.output_dir}/")
    print("  Done.\n")


if __name__ == "__main__":
    main()
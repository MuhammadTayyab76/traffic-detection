"""
scripts/generate_report.py

Reads all evaluation JSON files in evaluation_outputs/ and
prints a final comparison table. Run this after evaluating
multiple models to compare them side by side.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --output-dir evaluation_outputs
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.training.evaluator import print_comparison_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="evaluation_outputs",
        help="Folder containing *_evaluation.json files"
    )
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    json_files = sorted(out_path.glob("*_evaluation.json"))

    if not json_files:
        print(f"No evaluation JSON files found in {args.output_dir}/")
        print("Run scripts/evaluate.py first.")
        return

    summaries = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)

        s = data.get("summary", {})
        summaries.append({
            "model":     data.get("model", jf.stem),
            "mAP50":     s.get("mAP50",     0),
            "mAP50_95":  s.get("mAP50_95",  0),
            "precision": s.get("precision", 0),
            "recall":    s.get("recall",    0),
            "fps":       s.get("fps",       0),
        })

    print_comparison_table(summaries)

    # Per-class breakdown for each model
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        print(f"\n  Per-class — {data['model']}")
        print(f"  {'Class':<16} {'AP@0.5':>8} {'Precision':>10} {'Recall':>8}")
        print("  " + "-"*44)
        for cls, metrics in data.get("per_class", {}).items():
            print(f"  {cls:<16} {metrics['ap50']:>8.4f} "
                  f"{metrics['precision']:>10.4f} "
                  f"{metrics['recall']:>8.4f}")


if __name__ == "__main__":
    main()
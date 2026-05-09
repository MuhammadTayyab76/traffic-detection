"""
Load evaluation JSONs for YOLOv8s and SSDLite and print a side-by-side
comparison table. Saves comparison to evaluation_outputs/comparison_report.json.

Usage (from project root):
    python scripts/compare_models.py
    python scripts/compare_models.py
        --yolo evaluation_outputs/yolov8s_80ep_evaluation.json
        --ssd  evaluation_outputs/ssd_evaluation.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


YOLO_DEFAULT = "evaluation_outputs/yolov8s_80ep_evaluation.json"
SSD_DEFAULT  = "evaluation_outputs/ssd_evaluation.json"
OUT_DEFAULT  = "evaluation_outputs/comparison_report.json"

METRICS = [
    ("map50",      "mAP@0.5",      ".4f"),
    ("map50_95",   "mAP@0.5:0.95", ".4f"),
    ("precision",  "Precision",    ".4f"),
    ("recall",     "Recall",       ".4f"),
    ("fps",        "FPS",          ".1f"),
    ("latency_ms", "Latency (ms)", ".1f"),
]


def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {p}\n"
            "Run evaluate.py / evaluate_ssd.py first."
        )
    with open(p) as f:
        return json.load(f)


def _winner(yolo_val: float, ssd_val: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        return "YOLOv8s" if yolo_val >= ssd_val else "SSDLite"
    return "YOLOv8s" if yolo_val <= ssd_val else "SSDLite"


def build_comparison(yolo: dict, ssd: dict) -> dict:
    metrics: dict = {}
    for key, label, _ in METRICS:
        yv = yolo.get(key, 0.0)
        sv = ssd.get(key,  0.0)
        higher = key != "latency_ms"
        metrics[key] = {
            "label":       label,
            "yolov8s":     yv,
            "ssdlite":     sv,
            "winner":      _winner(yv, sv, higher),
            "difference":  round(abs(yv - sv), 4),
        }

    yolo_wins = sum(1 for m in metrics.values() if m["winner"] == "YOLOv8s")
    ssd_wins  = sum(1 for m in metrics.values() if m["winner"] == "SSDLite")

    per_class_comparison: dict = {}
    yolo_pc = yolo.get("per_class", {})
    ssd_pc  = ssd.get("per_class",  {})
    all_classes = sorted(set(list(yolo_pc.keys()) + list(ssd_pc.keys())))
    for cls in all_classes:
        y_ap = yolo_pc.get(cls, {}).get("ap50", 0.0)
        s_ap = ssd_pc.get(cls,  {}).get("ap50", 0.0)
        per_class_comparison[cls] = {
            "yolov8s_ap50": y_ap,
            "ssdlite_ap50": s_ap,
            "winner":       _winner(y_ap, s_ap),
        }

    return {
        "models": {
            "yolov8s": yolo.get("model", "yolov8s"),
            "ssdlite": ssd.get("model",  "ssdlite320_mobilenet_v3_large"),
        },
        "num_images": {
            "yolov8s": yolo.get("num_images", 0),
            "ssdlite": ssd.get("num_images",  0),
        },
        "metrics":              metrics,
        "per_class":            per_class_comparison,
        "overall_winner_count": {"yolov8s": yolo_wins, "ssdlite": ssd_wins},
        "recommendation":       "YOLOv8s" if yolo_wins >= ssd_wins else "SSDLite",
    }


def print_table(comparison: dict) -> None:
    col_w = 18
    sep   = "+" + "-" * 22 + "+" + "-" * col_w + "+" + "-" * col_w + "+" + "-" * 12 + "+"

    print("\n" + sep)
    print(f"| {'Metric':<20} | {'YOLOv8s':>{col_w-2}} | {'SSDLite':>{col_w-2}} | {'Winner':<10} |")
    print(sep)

    for key, _, fmt in METRICS:
        m   = comparison["metrics"][key]
        yv  = format(m["yolov8s"], fmt)
        sv  = format(m["ssdlite"], fmt)
        win = m["winner"]
        print(f"| {m['label']:<20} | {yv:>{col_w-2}} | {sv:>{col_w-2}} | {win:<10} |")

    print(sep)
    counts = comparison["overall_winner_count"]
    print(f"| {'Overall wins':<20} | {counts['yolov8s']:>{col_w-2}} | {counts['ssdlite']:>{col_w-2}} | {'':10} |")
    print(sep)

    print(f"\nRecommended model: {comparison['recommendation']}")

    print("\nPer-class AP@0.5:")
    pc_sep = "+" + "-" * 18 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+"
    print(pc_sep)
    print(f"| {'Class':<16} | {'YOLOv8s':>10} | {'SSDLite':>10} | {'Winner':<10} |")
    print(pc_sep)
    for cls, vals in comparison["per_class"].items():
        print(
            f"| {cls:<16} | "
            f"{vals['yolov8s_ap50']:>10.4f} | "
            f"{vals['ssdlite_ap50']:>10.4f} | "
            f"{vals['winner']:<10} |"
        )
    print(pc_sep)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare YOLOv8s vs SSDLite evaluation results")
    p.add_argument("--yolo",   default=YOLO_DEFAULT)
    p.add_argument("--ssd",    default=SSD_DEFAULT)
    p.add_argument("--output", default=OUT_DEFAULT)
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    yolo       = _load(args.yolo)
    ssd        = _load(args.ssd)
    comparison = build_comparison(yolo, ssd)

    print_table(comparison)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
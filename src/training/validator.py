"""
validator.py

Runs evaluation on the test/validation set and prints a clean
metrics report. Call this after training to get your final numbers.
"""

import yaml
import torch
from pathlib import Path
from ultralytics import YOLO

CLASS_NAMES = {
    0: "bike",
    1: "bus",
    2: "car",
    3: "motor",
    4: "person",
    5: "rider",
    6: "traffic light",
    7: "traffic sign",
    8: "train",
    9: "truck",
}

def validate(
    model_path: str,
    dataset_config: str = "configs/dataset.yaml",
    image_size: int = 640,
    split: str = "val",
) -> dict:
    """
    Evaluate a trained YOLOv8 model and print a full metrics report.
    """
    print("\n" + "="*55)
    print("  Traffic Detection — Model Evaluation")
    print("="*55)
    print(f"  Model  : {model_path}")
    print(f"  Split  : {split}")
    print("="*55 + "\n")

    device = "0" if torch.cuda.is_available() else "cpu"
    model  = YOLO(model_path)

    metrics = model.val(
        data    = dataset_config,
        imgsz   = image_size,
        split   = split,
        device  = device,
        plots   = True,
        verbose = True,
    )

    # Extract and print clean report
    print("\n")
    print("  EVALUATION RESULTS")
    print("\n")

    map50    = metrics.box.map50
    map5095  = metrics.box.map
    precision = metrics.box.mp     # mean precision
    recall    = metrics.box.mr     # mean recall

    print(f"  mAP @ 0.5       : {map50:.4f}")
    print(f"  mAP @ 0.5:0.95  : {map5095:.4f}")
    print(f"  Mean Precision  : {precision:.4f}")
    print(f"  Mean Recall     : {recall:.4f}")

    # Per-class breakdown
    print("\n  Per-class breakdown:")
    print(f"  {'Class':<16} {'AP@0.5':>8} {'Precision':>10} {'Recall':>8}")
    print("  " + "-"*44)

    per_class_ap = metrics.box.ap50          # list, one per class
    per_class_p  = metrics.box.p             # precision per class
    per_class_r  = metrics.box.r             # recall per class

    results = {}
    for i, (ap, p, r) in enumerate(
        zip(per_class_ap, per_class_p, per_class_r)
    ):
        name = CLASS_NAMES.get(i, f"class_{i}")
        print(f"  {name:<16} {ap:>8.4f} {p:>10.4f} {r:>8.4f}")
        results[name] = {"ap50": ap, "precision": p, "recall": r}

    print("\n")

    results["_summary"] = {
        "mAP50":     map50,
        "mAP50_95":  map5095,
        "precision": precision,
        "recall":    recall,
    }
    return results


def compare_models(model_paths: list[str],
                   dataset_config: str = "configs/dataset.yaml") -> None:
    """
    Evaluate multiple model checkpoints and print a comparison table.
    Useful for comparing yolov8s vs yolov8m after both are trained.
    """
    print("\n")
    print("  MODEL COMPARISON")
    print("\n")
    print(f"  {'Model':<35} {'mAP@0.5':>8} {'mAP@0.5:0.95':>13} {'P':>6} {'R':>6}")
    print("\n")

    for path in model_paths:
        if not Path(path).exists():
            print(f"  {path:<35} NOT FOUND")
            continue

        device = "0" if torch.cuda.is_available() else "cpu"
        model  = YOLO(path)
        m      = model.val(
            data=dataset_config, verbose=False,
            plots=False, device=device
        )
        name   = Path(path).stem[:35]
        print(f"  {name:<35} {m.box.map50:>8.4f} "
              f"{m.box.map:>13.4f} "
              f"{m.box.mp:>6.4f} {m.box.mr:>6.4f}")

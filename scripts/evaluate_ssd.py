"""
Evaluate trained SSDLite320 on the validation set.
Computes mAP@0.5, mAP@0.5:0.95, per-class precision/recall, and FPS.
Outputs evaluation_outputs/ssd_evaluation.json in the same schema
as the YOLOv8 evaluation JSON so compare_models.py can read both.

Usage (from project root, conda env active):
    python scripts/evaluate_ssd.py
    python scripts/evaluate_ssd.py --weights weights/ssd_best.pt
                                   --data    data/master_voc_dataset
                                   --split   valid
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from src.dataset.voc_dataset import VOCDetectionDataset, NUM_CLASSES
from src.training.ssd_trainer import _build_model, _collate

CLASS_NAMES = [
    "bike", "bus", "car", "motor", "person",
    "rider", "traffic light", "traffic sign", "train", "truck",
]

IDX_TO_NAME = {i + 1: name for i, name in enumerate(CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SSDLite320 on validation set")
    p.add_argument("--weights",    default="weights/ssd_best.pt")
    p.add_argument("--data", default="data/master_voc_dataset")
    p.add_argument("--split",      default="valid")
    p.add_argument("--image-size", type=int, default=320)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--device",     default="0")
    p.add_argument("--output",     default="evaluation_outputs/ssd_evaluation.json")
    return p.parse_args()


def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device(
        f"cuda:{args.device}"
        if args.device.isdigit() and torch.cuda.is_available()
        else "cpu"
    )
    print(f"[SSD Eval] device : {device}")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found at {weights_path}.\n"
            "Run python scripts/train_ssd.py first."
        )

    model = _build_model(num_classes=NUM_CLASSES, pretrained_backbone=True)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"[SSD Eval] loaded : {weights_path}")

    dataset = VOCDetectionDataset(
        root=args.data,
        split=args.split,
        image_size=args.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=_collate,
        pin_memory=device.type == "cuda",
    )

    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

    total_images  = 0
    total_time_ms = 0.0

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]

            t0 = time.perf_counter()
            preds = model(images)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            total_images  += len(images)
            total_time_ms += elapsed_ms

            preds_cpu   = [{k: v.cpu() for k, v in p.items()} for p in preds]
            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
            metric.update(preds_cpu, targets_cpu)

    results   = metric.compute()
    fps       = round(total_images / (total_time_ms / 1000), 2)
    avg_ms    = round(total_time_ms / max(total_images, 1), 2)

    per_class_ap50 = results.get("map_per_class", torch.tensor([])).tolist()
    per_class: dict = {}
    for i, ap in enumerate(per_class_ap50):
        name = IDX_TO_NAME.get(i + 1, f"class_{i+1}")
        per_class[name] = {"ap50": round(float(ap), 4)}

    output = {
        "model":        "ssdlite320_mobilenet_v3_large",
        "weights":      str(weights_path),
        "split":        args.split,
        "num_images":   total_images,
        "map50":        round(float(results["map_50"].item()),    4),
        "map50_95":     round(float(results["map"].item()),       4),
        "precision":    round(float(results.get("mar_100",
                              results["map_50"]).item()),         4),
        "recall":       round(float(results.get("mar_1",
                              results["map_50"]).item()),         4),
        "fps":          fps,
        "latency_ms":   avg_ms,
        "per_class":    per_class,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[SSD Eval] mAP@0.5    : {output['map50']}")
    print(f"[SSD Eval] mAP@0.5:95 : {output['map50_95']}")
    print(f"[SSD Eval] FPS        : {fps}")
    print(f"[SSD Eval] Saved to   : {out_path}")

    return output


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
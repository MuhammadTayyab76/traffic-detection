"""
scripts/export_model.py

Export the trained model to ONNX and/or TorchScript.

Usage:
    # Export to ONNX (default)
    python scripts/export_model.py \
        --weights weights/bdd100k_yolov8s_30ep_best.pt

    # Export to both formats
    python scripts/export_model.py \
        --weights weights/bdd100k_yolov8s_30ep_best.pt \
        --format all

    # Export to TorchScript only
    python scripts/export_model.py \
        --weights weights/bdd100k_yolov8s_30ep_best.pt \
        --format torchscript

    # Export ONNX and benchmark it
    python scripts/export_model.py \
        --weights weights/bdd100k_yolov8s_30ep_best.pt \
        --format onnx \
        --benchmark
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference.exporter import (
    export_model,
    export_all,
    verify_onnx,
    benchmark_onnx,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export trained YOLOv8 model to deployment formats"
    )
    parser.add_argument(
        "--weights",   required=True,
        help="Path to .pt weights file"
    )
    parser.add_argument(
        "--format",    default="onnx",
        choices=["onnx", "torchscript", "all"],
        help="Export format (default: onnx)"
    )
    parser.add_argument(
        "--size",      type=int, default=640,
        help="Image size — must match training (default: 640)"
    )
    parser.add_argument(
        "--output-dir", default="weights",
        help="Where to save exported files (default: weights/)"
    )
    parser.add_argument(
        "--dynamic",   action="store_true",
        help="ONNX: enable dynamic batch size"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run inference speed benchmark after ONNX export"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.format == "all":
        results = export_all(
            weights_path = args.weights,
            image_size   = args.size,
            output_dir   = args.output_dir,
        )
        exported_paths = results

    else:
        path = export_model(
            weights_path  = args.weights,
            export_format = args.format,
            image_size    = args.size,
            output_dir    = args.output_dir,
            dynamic       = args.dynamic,
        )
        exported_paths = {args.format: path}

    # ONNX verification & benchmark 
    if "onnx" in exported_paths and exported_paths["onnx"]:
        onnx_path = exported_paths["onnx"]
        print("\n── Verifying ONNX model ─────────────────────────")
        verify_onnx(onnx_path)

        if args.benchmark:
            print("\n── Benchmarking ONNX inference ──────────────────")
            benchmark_onnx(onnx_path, image_size=args.size)

    # Final summary 
    print("\n── Export Summary ───────────────────────────────")
    for fmt, path in exported_paths.items():
        status = f"→ {path}" if path else "FAILED"
        print(f"  {fmt.upper():<14} {status}")
    print()


if __name__ == "__main__":
    main()
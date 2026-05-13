"""
Exports a trained YOLOv8 .pt model to:
  - ONNX         : cross-platform, runs on any runtime
  - TorchScript  : PyTorch native, fastest on GPU"
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import torch
from ultralytics import YOLO


# Supported export formats 

SUPPORTED_FORMATS = {
    "onnx":        "ONNX          (.onnx) — cross-platform inference",
    "torchscript": "TorchScript   (.torchscript) — PyTorch native",
}


# Core export function 

def export_model(
    weights_path: str,
    export_format: str = "onnx",
    image_size: int = 640,
    output_dir: str = "weights",
    dynamic: bool = False,
    simplify: bool = True,
) -> str:
    weights_path  = Path(weights_path)
    export_format = export_format.lower()

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    if export_format not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{export_format}'. "
            f"Choose from: {list(SUPPORTED_FORMATS.keys())}"
        )

    print("\n  Traffic Detection — Model Export\n")
    print(f"  Source   : {weights_path}")
    print(f"  Format   : {SUPPORTED_FORMATS[export_format]}")
    print(f"  img size : {image_size}")
    print(f"  Device   : {'GPU' if torch.cuda.is_available() else 'CPU'}")

    model = YOLO(str(weights_path))

    # Export 
    export_kwargs = dict(
        format  = export_format,
        imgsz   = image_size,
        simplify= simplify,
        dynamic = dynamic,
    )

    exported_path = model.export(**export_kwargs)
    exported_path = Path(exported_path)

    # Copy to output_dir 
    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest     = out_dir / exported_path.name

    import shutil
    if exported_path.resolve() != dest.resolve():
        shutil.copy(exported_path, dest)
    print(f"\n  Exported model copied → {dest}")
    print(f"  File size            : {dest.stat().st_size / 1e6:.1f} MB")

    return str(dest)


# Export all formats at once 

def export_all(
    weights_path: str,
    image_size: int = 640,
    output_dir: str = "weights",
) -> dict:
    """
    Export the model to all supported formats in one call.

    Args:
        weights_path: Path to .pt weights file.
        image_size:   Must match training size.
        output_dir:   Where to save all exported files.

    Returns:
        Dict mapping format name → exported file path.
    """
    results = {}
    for fmt in SUPPORTED_FORMATS:
        print(f"\n Exporting to {fmt.upper()} ")
        try:
            path = export_model(
                weights_path  = weights_path,
                export_format = fmt,
                image_size    = image_size,
                output_dir    = output_dir,
            )
            results[fmt] = path
            print(f"  ✓ {fmt.upper()} export successful")
        except Exception as e:
            print(f"  ✗ {fmt.upper()} export failed: {e}")
            results[fmt] = None

    return results


# Verify ONNX export

def verify_onnx(onnx_path: str) -> bool:
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print(f"  ONNX verification passed → {onnx_path}")
        return True
    except Exception as e:
        print(f"  ONNX verification failed: {e}")
        return False

def benchmark_onnx(
    onnx_path: str,
    image_size: int = 640,
    n_runs: int = 100,
) -> dict:
    import numpy as np
    import time

    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed — skipping benchmark")
        return {}

    # Choose provider
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if torch.cuda.is_available()
        else ["CPUExecutionProvider"]
    )

    session  = ort.InferenceSession(onnx_path, providers=providers)
    inp_name = session.get_inputs()[0].name

    # Dummy input — batch of 1
    dummy = np.random.rand(1, 3, image_size, image_size).astype(np.float32)

    # Warmup
    for _ in range(10):
        session.run(None, {inp_name: dummy})

    # Benchmark
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {inp_name: dummy})
        times.append((time.perf_counter() - t0) * 1000)

    mean_ms = float(np.mean(times))
    std_ms  = float(np.std(times))
    fps     = 1000 / mean_ms

    provider_used = session.get_providers()[0]
    print(f"\n  ONNX Benchmark ({provider_used}):")
    print(f"  Mean latency : {mean_ms:.2f} ms ± {std_ms:.2f}")
    print(f"  Throughput   : {fps:.1f} FPS")

    return {"mean_ms": mean_ms, "std_ms": std_ms, "fps": fps}
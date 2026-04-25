"""
trainer.py

Wraps Ultralytics YOLOv8 training with our project config.
Handles device selection, hyperparameter loading, and
checkpoint management automatically.
"""

import yaml
import torch
from pathlib import Path
from ultralytics import YOLO


def load_hyperparams(yaml_path: str = "src/training/hyperparams.yaml") -> dict:
    """Load training configuration from YAML."""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def get_device(requested: int | str) -> str:
    """
    Resolve training device.
    Returns '0' for first GPU if available, 'cpu' as fallback.
    """
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        print(f"  GPU detected : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM total   : "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return str(requested)
    else:
        print("  WARNING: No GPU found — falling back to CPU.")
        print("  Training will be significantly slower.")
        return "cpu"


def train(hyperparams_path: str = "src/training/hyperparams.yaml") -> dict:
    """
    Run a full YOLOv8 training run using the project hyperparams.

    Args:
        hyperparams_path: Path to hyperparams.yaml

    Returns:
        dict containing best_model_path, results_dir, and metrics summary.
    """
    hp = load_hyperparams(hyperparams_path)

    print("\n" + "="*55)
    print("  Traffic Detection — YOLOv8 Training")
    print("="*55)
    print(f"  Architecture : {hp['model']['architecture']}")
    print(f"  Dataset      : {hp['dataset']['config']}")
    print(f"  Epochs       : {hp['training']['epochs']}")
    print(f"  Batch size   : {hp['training']['batch_size']}")
    print(f"  Image size   : {hp['dataset']['image_size']}")
    print("="*55 + "\n")

    device = get_device(hp["training"]["device"])

    # Load Model
    # If pretrained=True, Ultralytics downloads COCO weights automatically
    model = YOLO(hp["model"]["architecture"])

    # Training args
    aug   = hp["augmentation"]
    train_cfg = hp["training"]
    opt   = hp["optimizer"]
    out   = hp["output"]

    results = model.train(
        # Dataset
        data        = hp["dataset"]["config"],
        imgsz       = hp["dataset"]["image_size"],

        # Training schedule
        epochs      = train_cfg["epochs"],
        batch       = train_cfg["batch_size"],
        patience    = train_cfg["patience"],
        workers     = train_cfg["workers"],
        device      = device,

        # Optimiser
        optimizer   = opt["name"],
        lr0         = opt["lr0"],
        lrf         = opt["lrf"],
        momentum    = opt["momentum"],
        weight_decay= opt["weight_decay"],
        warmup_epochs = opt["warmup_epochs"],

        # Augmentation
        hsv_h       = aug["hsv_h"],
        hsv_s       = aug["hsv_s"],
        hsv_v       = aug["hsv_v"],
        degrees     = aug["degrees"],
        translate   = aug["translate"],
        scale       = aug["scale"],
        flipud      = aug["flipud"],
        fliplr      = aug["fliplr"],
        mosaic      = aug["mosaic"],
        mixup       = aug["mixup"],

        # Output
        project     = out["project"],
        name        = out["name"],
        save_period = out["save_period"],

        # Extras
        exist_ok    = True,    # overwrite run folder if same name
        pretrained  = hp["model"]["pretrained"],
        verbose     = True,
        plots       = True,    # save training curve plots automatically
    )

    # Locate best weights
    best_weights = (
        Path(out["project"]) / out["name"] / "weights" / "best.pt"
    )

    # Copy best weights to weights/ folder 
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    dest = weights_dir / f"{out['name']}_best.pt"

    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, dest)
        print(f"\n  Best weights saved to : {dest}")
    else:
        print("\n  WARNING: best.pt not found — check runs/train/ folder")

    return {
        "best_model_path": str(dest),
        "results_dir":     str(Path(out["project"]) / out["name"]),
        "results":         results,
    }
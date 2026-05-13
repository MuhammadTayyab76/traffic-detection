"""
    python scripts/train.py --epochs 20
    python scripts/train.py --model yolov8m.pt --epochs 80
    python scripts/train.py --eval-only --weights weights/bdd100k_yolov8s_30ep_best.pt
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.training.trainer   import train, load_hyperparams
from src.training.validator import validate



def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or evaluate traffic detection model"
    )
    parser.add_argument(
        "--hyperparams", default="src/training/hyperparams.yaml",
        help="Path to hyperparams yaml"
    )
    parser.add_argument(
        "--model", default=None,
        help="Override model architecture (e.g. yolov8m.pt)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training, only run evaluation"
    )
    parser.add_argument(
        "--weights", default=None,
        help="Path to weights for --eval-only mode"
    )
    return parser.parse_args()


def apply_overrides(hp: dict, args) -> dict:
    """Apply CLI overrides on top of yaml settings."""
    if args.model:
        hp["model"]["architecture"] = args.model
        print(f"  Override — model    : {args.model}")
    if args.epochs:
        hp["training"]["epochs"] = args.epochs
        print(f"  Override — epochs   : {args.epochs}")
    if args.batch:
        hp["training"]["batch_size"] = args.batch
        print(f"  Override — batch    : {args.batch}")
    return hp


def main():
    args = parse_args()

    if args.eval_only:
        # Evaluation only mode 
        if not args.weights:
            print("ERROR: --eval-only requires --weights path")
            sys.exit(1)
        validate(model_path=args.weights)
        return

    # Training mode 
    hp = load_hyperparams(args.hyperparams)
    hp = apply_overrides(hp, args)

    # Write overridden hp back temporarily so trainer picks them up
    # Pass hp directly to avoid re-reading from disk
    import src.training.trainer as trainer_module

    # Monkey-patch load_hyperparams to return our modified hp
    trainer_module.load_hyperparams = lambda *a, **kw: hp

    result = train(args.hyperparams)

    print("\n── Training complete ──────────────────────────")
    print(f"  Best weights : {result['best_model_path']}")
    print(f"  Results dir  : {result['results_dir']}")

    # Auto-run evaluation on the best weights
    print("\n── Running evaluation on best weights ─────────")
    validate(
        model_path=result["best_model_path"],
        dataset_config=hp["dataset"]["config"],
        image_size=hp["dataset"]["image_size"],
        split="test",
    )


if __name__ == "__main__":
    main()
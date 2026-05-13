"""
    python scripts/train_ssd.py --config src/training/ssd_hyperparams.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure project root is on the path regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.ssd_trainer import SSDTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SSDLite320 on BDD100K VOC dataset")
    p.add_argument(
        "--config",
        type=str,
        default="src/training/ssd_hyperparams.yaml",
        help="Path to ssd_hyperparams.yaml",
    )
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    config = Path(args.config)

    if not config.exists():
        raise FileNotFoundError(
            f"Config not found: {config}\n"
            "Make sure you are running from the project root directory."
        )

    trainer = SSDTrainer(config)
    result  = trainer.train()

    print("\n=== SSD Training Summary ===")
    print(f"Best weights : {result['best_model_path']}")
    print(f"Best val loss: {result['best_val_loss']:.5f}")
    print(f"Log          : {result['log_path']}")


if __name__ == "__main__":
    main()
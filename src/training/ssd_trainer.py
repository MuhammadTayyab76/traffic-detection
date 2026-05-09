"""
SSDLite320 training pipeline.
Model:   torchvision ssdlite320_mobilenet_v3_large
Input:   PASCAL VOC annotations from data/bdd100k_voc/
Output:  weights/ssd_best.pt + runs/ssd/<name>/training_log.json
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

from src.dataset.voc_dataset import VOCDetectionDataset, NUM_CLASSES


# helpers 

def _collate(batch):
    return tuple(zip(*batch))


def _build_model(num_classes: int, pretrained_backbone: bool) -> torch.nn.Module:
    """
    Build SSDLite320 with MobileNetV3-Large backbone.
    pretrained_backbone=True loads ImageNet weights for the backbone only;
    the detection head is always randomly initialised for our class count.
    """
    backbone_weights = (
        MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    )
    model = ssdlite320_mobilenet_v3_large(
        weights=None,                      # no full pretrained SSD weights
        weights_backbone=backbone_weights, # pretrained feature extractor
        num_classes=num_classes,
    )
    return model


def _val_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.train()
    
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()

    total, count = 0.0, 0
    with torch.no_grad():
        for images, targets in loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            total += sum(loss_dict.values()).item()
            count += 1
    return total / max(count, 1)


def _warmup_lr(optimizer: SGD, epoch: int, warmup_epochs: int, base_lr: float) -> None:
    """Linear LR warmup for the first `warmup_epochs` epochs."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for pg in optimizer.param_groups:
            pg["lr"] = lr


# trainer

class SSDTrainer:
    def __init__(self, config_path: str | Path) -> None:
        with open(config_path) as f:
            self.hp = yaml.safe_load(f)

        tr = self.hp["training"]
        self.device = torch.device(
            f"cuda:{tr['device']}"
            if isinstance(tr["device"], int) and torch.cuda.is_available()
            else "cpu"
        )

        out = self.hp["output"]
        self.run_dir     = Path(out["project"]) / out["name"]
        self.weights_dir = Path(out["weights_dir"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        print(f"[SSDTrainer] device      : {self.device}")
        print(f"[SSDTrainer] run dir     : {self.run_dir}")
        print(f"[SSDTrainer] num_classes : {self.hp['model']['num_classes']}")

    def train(self) -> Dict:
        hp      = self.hp
        ds_cfg  = hp["dataset"]
        tr_cfg  = hp["training"]
        mod_cfg = hp["model"]
        out_cfg = hp["output"]

        # data 
        train_ds = VOCDetectionDataset(
            ds_cfg["root"], split="train", image_size=ds_cfg["image_size"]
        )
        val_ds = VOCDetectionDataset(
            ds_cfg["root"], split="valid", image_size=ds_cfg["image_size"]
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=tr_cfg["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=tr_cfg["workers"],
            collate_fn=_collate,
            pin_memory=self.device.type == "cuda",
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=tr_cfg["batch_size"],
            shuffle=False,
            num_workers=tr_cfg["workers"],
            collate_fn=_collate,
            pin_memory=self.device.type == "cuda",
        )

        # model 
        model = _build_model(
            num_classes=mod_cfg["num_classes"],
            pretrained_backbone=mod_cfg["pretrained_backbone"],
        ).to(self.device)

        # optimiser + scheduler 
        optimizer = SGD(
            model.parameters(),
            lr=tr_cfg["lr"],
            momentum=tr_cfg["momentum"],
            weight_decay=tr_cfg["weight_decay"],
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=tr_cfg["epochs"])

        epochs         = tr_cfg["epochs"]
        patience       = tr_cfg["patience"]
        warmup_epochs  = tr_cfg.get("warmup_epochs", 3)
        save_period    = out_cfg["save_period"]
        best_model_path = str(self.weights_dir / "ssd_best.pt")

        best_val_loss    = math.inf
        patience_counter = 0
        history: List[Dict] = []

        # loop 
        for epoch in range(1, epochs + 1):
            _warmup_lr(optimizer, epoch - 1, warmup_epochs, tr_cfg["lr"])

            model.train()
            epoch_loss = 0.0
            t0 = time.time()

            for images, targets in train_loader:
                images  = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict  = model(images, targets)
                total_loss = sum(loss_dict.values())

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += total_loss.item()

            # Step scheduler only after warmup
            if epoch > warmup_epochs:
                scheduler.step()

            train_loss = epoch_loss / len(train_loader)
            val_loss   = _val_loss(model, val_loader, self.device)
            elapsed    = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"lr={current_lr:.6f} | "
                f"{elapsed:.0f}s"
            )

            history.append({
                "epoch":      epoch,
                "train_loss": round(train_loss, 5),
                "val_loss":   round(val_loss,   5),
                "lr":         round(current_lr, 8),
                "time_s":     round(elapsed, 2),
            })

            # Periodic checkpoint
            if epoch % save_period == 0:
                ckpt = str(self.weights_dir / f"ssd_epoch{epoch}.pt")
                torch.save(model.state_dict(), ckpt)
                print(f"  → checkpoint saved: {ckpt}")

            # Best model + early stopping
            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"  → best model updated (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  patience {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"[SSDTrainer] Early stopping triggered at epoch {epoch}.")
                    break

        # Save training log
        log = {"config": hp, "history": history, "best_val_loss": best_val_loss}
        log_path = self.run_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

        print(f"\n[SSDTrainer] Complete. Best weights → {best_model_path}")
        print(f"[SSDTrainer] Log            → {log_path}")

        return {
            "best_model_path": best_model_path,
            "best_val_loss":   best_val_loss,
            "history":         history,
            "log_path":        str(log_path),
        }
"""
dataset_builder.py

PyTorch Dataset for YOLO-format object detection data.
Returns (image_tensor, bboxes, class_ids) per sample.

Works with:
  - Raw Roboflow BDD100K export  (data/bdd100k/train/)
  - Augmented copies             (data/augmented/train/)
  - Any folder following images/ + labels/ structure
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from src.processing.augmentor import (
    build_train_pipeline,
    build_val_pipeline,
    augment_sample,
)


class TrafficDataset(Dataset):
    """
    PyTorch Dataset for YOLO-format traffic detection data.

    Directory structure expected:
        root/
        ├── images/
        │   ├── frame_000001.jpg
        │   └── ...
        └── labels/
            ├── frame_000001.txt
            └── ...

    Args:
        images_dir:  Path to the images/ folder.
        labels_dir:  Path to the labels/ folder.
        image_size:  Target square image size (default 640).
        augment:     If True, apply training augmentations.
                     If False, only resize (use for val/test).
        max_boxes:   Maximum number of boxes per image for padding.
                     Images with more boxes will be truncated.
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        image_size: int = 640,
        augment: bool = True,
        max_boxes: int = 100,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_size = image_size
        self.max_boxes  = max_boxes

        # Choose pipeline based on mode
        self.pipeline = (
            build_train_pipeline(image_size) if augment
            else build_val_pipeline(image_size)
        )

        # Collect all image paths that have a matching label file
        all_images = sorted(
            list(self.images_dir.glob("*.jpg")) +
            list(self.images_dir.glob("*.png"))
        )
        self.samples = [
            img for img in all_images
            if (self.labels_dir / (img.stem + ".txt")).exists()
        ]

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No matched image+label pairs found.\n"
                f"  images_dir : {images_dir}\n"
                f"  labels_dir : {labels_dir}"
            )

        print(f"  Dataset loaded: {len(self.samples)} samples "
              f"({'train+aug' if augment else 'val/test'})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.samples[idx]
        lbl_path = self.labels_dir / (img_path.stem + ".txt")

        # Load image 
        image = cv2.imread(str(img_path))
        if image is None:
            # Return a blank sample rather than crashing the DataLoader
            image = np.zeros(
                (self.image_size, self.image_size, 3), dtype=np.uint8
            )

        # Load labels
        yolo_boxes = []
        raw = lbl_path.read_text().strip().splitlines()
        for line in raw:
            parts = line.strip().split()
            if len(parts) == 5:
                yolo_boxes.append([float(p) for p in parts])

        # Augment
        image, yolo_boxes = augment_sample(image, yolo_boxes, self.pipeline)

        # Convert image to tensor
        # OpenCV: BGR HxWxC uint8  →  PyTorch: RGB CxHxW float32 [0,1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # CxHxW

        # Needed so DataLoader can batch variable-length box lists
        boxes   = np.zeros((self.max_boxes, 4), dtype=np.float32)
        classes = np.full(self.max_boxes, -1,   dtype=np.int64)

        n = min(len(yolo_boxes), self.max_boxes)
        for i in range(n):
            cls, x_c, y_c, w, h = yolo_boxes[i]
            boxes[i]   = [x_c, y_c, w, h]
            classes[i] = int(cls)

        return {
            "image":      image_tensor,           # [3, H, W] float32
            "boxes":      torch.from_numpy(boxes), # [max_boxes, 4] float32
            "classes":    torch.from_numpy(classes),# [max_boxes] int64
            "n_boxes":    n,                        # actual box count (int)
            "image_path": str(img_path),            # for debugging
        }


# DataLoader

def build_dataloaders(
    dataset_root: str,
    image_size: int = 640,
    batch_size: int = 16,
    num_workers: int = 4,
    max_boxes: int = 100,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from a Roboflow-style dataset root.

    Expected structure:
        dataset_root/
        ├── train/
        │   ├── images/
        │   └── labels/
        └── valid/
            ├── images/
            └── labels/

    Args:
        dataset_root: Path to the dataset root folder (e.g. 'data/bdd100k').
        image_size:   Target image size (default 640).
        batch_size:   Samples per batch (reduce if you get OOM errors).
        num_workers:  Parallel data loading workers.
                      Set to 0 on Windows if you get multiprocessing errors.
        max_boxes:    Max boxes per image for padding.

    Returns:
        (train_loader, val_loader)
    """
    root = Path(dataset_root)

    train_dataset = TrafficDataset(
        images_dir=str(root / "train" / "images"),
        labels_dir=str(root / "train" / "labels"),
        image_size=image_size,
        augment=True,
        max_boxes=max_boxes,
    )

    val_dataset = TrafficDataset(
        images_dir=str(root / "valid" / "images"),
        labels_dir=str(root / "valid" / "labels"),
        image_size=image_size,
        augment=False,
        max_boxes=max_boxes,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
"""
augmentor.py

Albumentations-based augmentation pipeline for YOLO-format
object detection data. All transforms are bounding-box aware —
every box is transformed consistently with its image.

YOLO bbox format coming IN  : [class_id, x_center, y_center, w, h]  (normalised)
Albumentations internal fmt : [x_min, y_min, x_max, y_max]          (normalised)
YOLO bbox format going OUT  : [class_id, x_center, y_center, w, h]  (normalised)

The conversion happens internally — callers always speak YOLO format.
"""

import cv2
import numpy as np
import albumentations as A
from pathlib import Path


# Bbox format helpers 

def yolo_to_albumentations(bboxes: list) -> tuple[list, list]:
    """
    Convert YOLO bbox rows to Albumentations format.

    Args:
        bboxes: List of [class_id, x_center, y_center, w, h] rows.

    Returns:
        (alb_bboxes, class_labels)
        alb_bboxes  : [[x_min, y_min, x_max, y_max], ...]  normalised
        class_labels: [class_id, ...]
    """
    alb_bboxes   = []
    class_labels = []

    for row in bboxes:
        cls, x_c, y_c, w, h = row
        x_min = x_c - w / 2
        y_min = y_c - h / 2
        x_max = x_c + w / 2
        y_max = y_c + h / 2

        # Clamp to [0, 1]
        x_min = max(0.0, min(1.0, x_min))
        y_min = max(0.0, min(1.0, y_min))
        x_max = max(0.0, min(1.0, x_max))
        y_max = max(0.0, min(1.0, y_max))

        if x_max > x_min and y_max > y_min:
            alb_bboxes.append([x_min, y_min, x_max, y_max])
            class_labels.append(int(cls))

    return alb_bboxes, class_labels


def albumentations_to_yolo(alb_bboxes: list, class_labels: list) -> list:
    """
    Convert Albumentations bbox output back to YOLO format rows.

    Returns:
        List of [class_id, x_center, y_center, w, h] rows.
    """
    yolo_rows = []
    for (x_min, y_min, x_max, y_max), cls in zip(alb_bboxes, class_labels):
        x_c = (x_min + x_max) / 2
        y_c = (y_min + y_max) / 2
        w   = x_max - x_min
        h   = y_max - y_min
        if w > 0 and h > 0:
            yolo_rows.append([cls, x_c, y_c, w, h])
    return yolo_rows


# Pipeline builder

def build_train_pipeline(image_size: int = 640) -> A.Compose:
    """
    Build the training augmentation pipeline.

    Transforms applied (all bbox-aware):
      - Horizontal flip          : mirrors the most common real-world variation
      - Random brightness/contrast: simulates lighting changes (day/night/overcast)
      - HSV shift                : colour temperature variation across cameras
      - Gaussian noise           : simulates low-quality CCTV sensor noise
      - Motion blur              : simulates fast-moving vehicles
      - Random shadow            : simulates tree/building shadows on road
      - Safe rotate (±10°)       : slight camera tilt variation
      - Random scale             : object size variation
      - Resize to target size    : final resize always happens last

    Args:
        image_size: Target square image size (default 640 for YOLOv8).

    Returns:
        An albumentations Compose pipeline.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),

            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5,
            ),

            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.4,
            ),

            A.GaussNoise(
                std_range=(0.02, 0.1),
                p=0.3,
            ),

            A.MotionBlur(
                blur_limit=(3, 7),
                p=0.2,
            ),

            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=5,
                p=0.2,
            ),

            A.SafeRotate(
                limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=0.3,
            ),
            A.RandomScale(
                scale_limit=0.2,
                p=0.3,
            ),

            A.Resize(
                height=image_size,
                width=image_size,
            ),
        ],
        bbox_params=A.BboxParams(
            format="albumentations",   # [x_min, y_min, x_max, y_max] normalised
            label_fields=["class_labels"],
            min_area=256,              # drop boxes smaller than 256 px² after transform
            min_visibility=0.3,        # drop boxes that become >70% occluded
        ),
    )


def build_val_pipeline(image_size: int = 640) -> A.Compose:
    """
    Validation pipeline — only resize, no random transforms.
    Keeps evaluation deterministic.
    """
    return A.Compose(
        [A.Resize(height=image_size, width=image_size)],
        bbox_params=A.BboxParams(
            format="albumentations",
            label_fields=["class_labels"],
            min_area=256,
            min_visibility=0.3,
        ),
    )


# Single-sample augmentation

def augment_sample(
    image: np.ndarray,
    yolo_bboxes: list,
    pipeline: A.Compose,
) -> tuple[np.ndarray, list]:
    """
    Apply an augmentation pipeline to one image and its YOLO bboxes.

    Args:
        image:       HxWxC numpy array (BGR from OpenCV).
        yolo_bboxes: List of [class_id, x_c, y_c, w, h] rows.
        pipeline:    Albumentations Compose pipeline.

    Returns:
        (augmented_image, augmented_yolo_bboxes)
    """
    alb_bboxes, class_labels = yolo_to_albumentations(yolo_bboxes)

    result = pipeline(
        image=image,
        bboxes=alb_bboxes,
        class_labels=class_labels,
    )

    aug_image      = result["image"]
    aug_yolo_boxes = albumentations_to_yolo(
        result["bboxes"],
        result["class_labels"],
    )
    return aug_image, aug_yolo_boxes


# Bulk augmentation

def augment_dataset(
    images_dir: str,
    labels_dir: str,
    output_images_dir: str,
    output_labels_dir: str,
    copies_per_image: int = 2,
    image_size: int = 640,
) -> dict:
    """
    Augment every image in a dataset folder and write new copies to disk.

    The original images are NOT modified — augmented copies are written
    alongside them with a suffix like _aug0, _aug1, etc.

    Args:
        images_dir:         Folder containing source images (.jpg / .png).
        labels_dir:         Folder containing YOLO .txt label files.
        output_images_dir:  Where to write augmented images.
        output_labels_dir:  Where to write augmented label files.
        copies_per_image:   How many augmented versions to generate per image.
        image_size:         Target size for resize (default 640).

    Returns:
        dict with keys: processed, augmented, skipped_no_label, skipped_no_image
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    out_img     = Path(output_images_dir)
    out_lbl     = Path(output_labels_dir)

    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    pipeline = build_train_pipeline(image_size)

    stats = {
        "processed":       0,
        "augmented":       0,
        "skipped_no_label":0,
        "skipped_no_image":0,
    }

    image_files = sorted(list(images_path.glob("*.jpg")) +
                         list(images_path.glob("*.png")))

    print(f"Found {len(image_files)} images — generating "
          f"{copies_per_image} augmented copies each ...")

    from tqdm import tqdm
    for img_file in tqdm(image_files, desc="Augmenting"):

        label_file = labels_path / (img_file.stem + ".txt")
        if not label_file.exists():
            stats["skipped_no_label"] += 1
            continue

        image = cv2.imread(str(img_file))
        if image is None:
            stats["skipped_no_image"] += 1
            continue

        # Parse YOLO label rows
        raw_lines  = label_file.read_text().strip().splitlines()
        yolo_boxes = []
        for line in raw_lines:
            parts = line.strip().split()
            if len(parts) == 5:
                yolo_boxes.append([float(p) for p in parts])

        stats["processed"] += 1

        for i in range(copies_per_image):
            aug_img, aug_boxes = augment_sample(image, yolo_boxes, pipeline)

            stem     = f"{img_file.stem}_aug{i}"
            out_img_path = out_img / f"{stem}.jpg"
            out_lbl_path = out_lbl / f"{stem}.txt"

            cv2.imwrite(str(out_img_path), aug_img)

            label_lines = [
                f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}"
                for b in aug_boxes
            ]
            out_lbl_path.write_text("\n".join(label_lines))
            stats["augmented"] += 1

    return stats
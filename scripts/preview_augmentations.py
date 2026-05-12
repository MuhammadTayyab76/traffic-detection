"""
scripts/preview_augmentations.py

Opens a window showing original vs augmented side-by-side for random samples.
Press any key to see the next sample. Press Q to quit.

Usage:
    python scripts/preview_augmentations.py --images data/bdd100k/train/images \
                                             --labels data/bdd100k/train/labels
"""

import argparse
import sys
import random
import cv2
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.processing.augmentor import build_train_pipeline, augment_sample

# Class colours for drawing boxes
CLASS_COLOURS = {
    0: (0,   255, 0),    # car          → green
    1: (255, 0,   0),    # pedestrian   → blue
    2: (0,   255, 255),  # traffic light→ yellow
    3: (0,   0,   255),  # traffic sign → red
}
CLASS_NAMES = {0: "car", 1: "pedestrian", 2: "traffic light", 3: "traffic sign"}


def draw_yolo_boxes(image: np.ndarray, yolo_boxes: list) -> np.ndarray:
    """Draw YOLO bounding boxes onto a copy of the image."""
    img   = image.copy()
    h, w  = img.shape[:2]
    for box in yolo_boxes:
        cls, x_c, y_c, bw, bh = box
        cls = int(cls)
        x1  = int((x_c - bw / 2) * w)
        y1  = int((y_c - bh / 2) * h)
        x2  = int((x_c + bw / 2) * w)
        y2  = int((y_c + bh / 2) * h)
        colour = CLASS_COLOURS.get(cls, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(img, CLASS_NAMES.get(cls, str(cls)),
                    (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--size",   type=int, default=640)
    args = parser.parse_args()

    images_path = Path(args.images)
    labels_path = Path(args.labels)
    pipeline    = build_train_pipeline(args.size)

    image_files = sorted(
        list(images_path.glob("*.jpg")) +
        list(images_path.glob("*.png"))
    )
    random.shuffle(image_files)

    print("Press any key → next sample    |    Q → quit")

    for img_file in image_files:
        lbl_file = labels_path / (img_file.stem + ".txt")
        if not lbl_file.exists():
            continue

        image = cv2.imread(str(img_file))
        if image is None:
            continue

        raw_lines  = lbl_file.read_text().strip().splitlines()
        yolo_boxes = []
        for line in raw_lines:
            parts = line.strip().split()
            if len(parts) == 5:
                yolo_boxes.append([float(p) for p in parts])

        aug_img, aug_boxes = augment_sample(image, yolo_boxes, pipeline)

        orig_drawn = draw_yolo_boxes(
            cv2.resize(image, (args.size, args.size)), yolo_boxes
        )
        aug_drawn  = draw_yolo_boxes(aug_img, aug_boxes)

        # Put labels at the top
        cv2.putText(orig_drawn, "ORIGINAL", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(aug_drawn,  "AUGMENTED", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        combined = np.hstack([orig_drawn, aug_drawn])
        cv2.imshow("Augmentation Preview", combined)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
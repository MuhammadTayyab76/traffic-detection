"""
PASCAL VOC dataset loader for SSDLite training.
Reads Roboflow VOC exports from data/bdd100k_voc/{train,valid}/
Each split folder contains paired .jpg + .xml files.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset

CLASS_TO_IDX: Dict[str, int] = {
    "bike":          1,
    "bus":           2,
    "car":           3,
    "motor":         4,
    "person":        5,
    "rider":         6,
    "traffic light": 7,
    "traffic sign":  8,
    "train":         9,
    "truck":         10,
}
NUM_CLASSES = len(CLASS_TO_IDX) + 1 

class VOCDetectionDataset(Dataset):
    """
    Loads Roboflow-exported PASCAL VOC annotations.

    Expected layout (Roboflow default):
        <root>/<split>/image_001.jpg
        <root>/<split>/image_001.xml
        ...
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transforms: Optional[Callable] = None,
        image_size: int = 320,
    ) -> None:
        self.root = Path(root) / split
        self.transforms = transforms
        self.image_size = image_size

        self.samples: List[Tuple[Path, Path]] = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img_path in sorted(self.root.glob(ext)):
                xml_path = img_path.with_suffix(".xml")
                if xml_path.exists():
                    self.samples.append((img_path, xml_path))

        if not self.samples:
            raise FileNotFoundError(
                f"No image+XML pairs found in {self.root}.\n"
                "Make sure you ran the Roboflow VOC download into data/bdd100k_voc/."
            )

        print(f"[VOCDataset] {split}: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        img_path, xml_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image = image.resize((self.image_size, self.image_size))
        image_tensor = F.to_tensor(image)  # float32 [0,1], shape [3,H,W]

        boxes, labels = self._parse_xml(xml_path, orig_w, orig_h)

        target: Dict = {
            "boxes":  torch.as_tensor(boxes,  dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }

        if self.transforms:
            image_tensor, target = self.transforms(image_tensor, target)

        return image_tensor, target

    def _parse_xml(
        self, xml_path: Path, orig_w: int, orig_h: int
    ) -> Tuple[List[List[float]], List[int]]:
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h

        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes: List[List[float]] = []
        labels: List[int] = []

        for obj in root.findall("object"):
            name = obj.find("name").text.strip().lower()
            if name not in CLASS_TO_IDX:
                continue

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text) * scale_x
            ymin = float(bndbox.find("ymin").text) * scale_y
            xmax = float(bndbox.find("xmax").text) * scale_x
            ymax = float(bndbox.find("ymax").text) * scale_y

            # Clamp to image bounds and skip degenerate boxes
            xmin = max(0.0, min(xmin, float(self.image_size - 1)))
            ymin = max(0.0, min(ymin, float(self.image_size - 1)))
            xmax = max(0.0, min(xmax, float(self.image_size)))
            ymax = max(0.0, min(ymax, float(self.image_size)))

            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_TO_IDX[name])

        # torchvision SSD crashes on empty targets — insert a dummy background box
        if not boxes:
            boxes  = [[0.0, 0.0, 1.0, 1.0]]
            labels = [0]

        return boxes, labels
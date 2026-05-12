"""
annotation_converter.py

Converts COCO JSON annotations to YOLO .txt format and vice versa.

COCO bbox format : [x_top_left, y_top_left, width, height]  (absolute pixels)
YOLO bbox format : [x_center, y_center, width, height]       (normalised 0-1)
"""

import json
from pathlib import Path
from tqdm import tqdm


# Maps COCO category names → YOLO class IDs
# names should be according to dataset category names
DEFAULT_CLASS_MAP = {
    "car":           0,
    "pedestrian":    1,
    "traffic light": 2,
    "traffic sign":  3,
}


# COCO → YOLO

def coco_bbox_to_yolo(
    bbox: list,
    img_width: int,
    img_height: int,
) -> tuple:
    """
    Convert a single COCO bounding box to YOLO normalised format.

    Args:
        bbox:       [x_top_left, y_top_left, width, height] in pixels.
        img_width:  Width of the image in pixels.
        img_height: Height of the image in pixels.

    Returns:
        (x_center, y_center, width, height) all normalised to [0, 1].
    """
    x_tl, y_tl, w, h = bbox

    x_center = (x_tl + w / 2) / img_width
    y_center  = (y_tl + h / 2) / img_height
    norm_w    = w / img_width
    norm_h    = h / img_height

    # Clamp to [0, 1] to handle any annotation boundary overflows
    x_center = max(0.0, min(1.0, x_center))
    y_center  = max(0.0, min(1.0, y_center))
    norm_w    = max(0.0, min(1.0, norm_w))
    norm_h    = max(0.0, min(1.0, norm_h))

    return x_center, y_center, norm_w, norm_h


def convert_coco_to_yolo(
    coco_json_path: str,
    output_dir: str,
    class_map: dict = None,
) -> dict:
    """
    Convert an entire COCO JSON annotation file to per-image YOLO .txt files.

    Args:
        coco_json_path: Path to the COCO-format .json file.
        output_dir:     Folder where .txt files will be written.
                        One .txt per image, named to match the image filename.
        class_map:      Dict mapping category name → YOLO class ID.
                        Defaults to DEFAULT_CLASS_MAP if not provided.

    Returns:
        dict with keys: converted, skipped_no_annotations,
                        skipped_unknown_class, total_boxes
    """
    if class_map is None:
        class_map = DEFAULT_CLASS_MAP

    coco_path = Path(coco_json_path)
    if not coco_path.exists():
        raise FileNotFoundError(f"COCO JSON not found: {coco_json_path}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading {coco_path.name} ...")
    with open(coco_path, "r") as f:
        coco = json.load(f)

    # image_id → image info
    images = {img["id"]: img for img in coco["images"]}

    # category_id → YOLO class_id  (skip unknowns)
    cat_id_to_yolo = {}
    skipped_cats   = set()
    for cat in coco["categories"]:
        name = cat["name"].lower()
        if name in class_map:
            cat_id_to_yolo[cat["id"]] = class_map[name]
        else:
            skipped_cats.add(name)

    if skipped_cats:
        print(f"  Warning: These categories are not in class_map "
              f"and will be skipped: {skipped_cats}")

    # image_id → list of annotations
    ann_by_image: dict[int, list] = {img_id: [] for img_id in images}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id in ann_by_image:
            ann_by_image[img_id].append(ann)

    # Convert 
    stats = {
        "converted":               0,
        "skipped_no_annotations":  0,
        "skipped_unknown_class":   0,
        "total_boxes":             0,
    }

    for img_id, img_info in tqdm(images.items(), desc="Converting annotations"):
        annotations = ann_by_image.get(img_id, [])

        if not annotations:
            stats["skipped_no_annotations"] += 1
            continue

        img_w = img_info["width"]
        img_h = img_info["height"]

        # Output .txt filename matches the image filename (without extension)
        stem     = Path(img_info["file_name"]).stem
        txt_file = out_path / f"{stem}.txt"

        lines = []
        for ann in annotations:
            cat_id = ann["category_id"]

            if cat_id not in cat_id_to_yolo:
                stats["skipped_unknown_class"] += 1
                continue

            yolo_class = cat_id_to_yolo[cat_id]
            x_c, y_c, w, h = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)

            # Skip degenerate boxes (zero area)
            if w <= 0 or h <= 0:
                continue

            lines.append(f"{yolo_class} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
            stats["total_boxes"] += 1

        if lines:
            txt_file.write_text("\n".join(lines))
            stats["converted"] += 1

    return stats


# YOLO → COCO

def yolo_bbox_to_coco(
    x_center: float,
    y_center: float,
    norm_w: float,
    norm_h: float,
    img_width: int,
    img_height: int,
) -> list:
    """
    Convert a single YOLO normalised bbox back to COCO absolute pixel format.

    Returns:
        [x_top_left, y_top_left, width, height] in pixels.
    """
    w     = norm_w * img_width
    h     = norm_h * img_height
    x_tl  = (x_center * img_width)  - w / 2
    y_tl  = (y_center * img_height) - h / 2
    return [round(x_tl, 2), round(y_tl, 2), round(w, 2), round(h, 2)]


def convert_yolo_to_coco(
    yolo_labels_dir: str,
    images_dir: str,
    output_json_path: str,
    class_map: dict = None,
) -> dict:
    """
    Convert a folder of YOLO .txt labels back into a single COCO JSON file.
    Useful for running COCO evaluation tools on YOLO-trained models.

    Args:
        yolo_labels_dir:  Folder containing YOLO .txt files.
        images_dir:       Folder containing corresponding image files.
        output_json_path: Where to write the output COCO JSON.
        class_map:        Dict mapping category name → YOLO class ID.

    Returns:
        dict with keys: images_processed, total_boxes
    """
    if class_map is None:
        class_map = DEFAULT_CLASS_MAP

    # Reverse map: YOLO class ID → category name
    id_to_name = {v: k for k, v in class_map.items()}

    labels_path = Path(yolo_labels_dir)
    images_path = Path(images_dir)

    coco_out = {
        "images":      [],
        "annotations": [],
        "categories":  [
            {"id": yolo_id, "name": name}
            for name, yolo_id in class_map.items()
        ],
    }

    ann_id   = 1
    img_id   = 1
    stats    = {"images_processed": 0, "total_boxes": 0}

    for txt_file in tqdm(sorted(labels_path.glob("*.txt")),
                         desc="Converting YOLO → COCO"):
        # Find matching image
        img_file = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = images_path / (txt_file.stem + ext)
            if candidate.exists():
                img_file = candidate
                break

        if img_file is None:
            continue

        import cv2
        frame = cv2.imread(str(img_file))
        if frame is None:
            continue
        img_h, img_w = frame.shape[:2]

        coco_out["images"].append({
            "id":        img_id,
            "file_name": img_file.name,
            "width":     img_w,
            "height":    img_h,
        })

        for line in txt_file.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id         = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:])
            bbox = yolo_bbox_to_coco(x_c, y_c, w, h, img_w, img_h)

            coco_out["annotations"].append({
                "id":          ann_id,
                "image_id":    img_id,
                "category_id": cls_id,
                "bbox":        bbox,
                "area":        round(bbox[2] * bbox[3], 2),
                "iscrowd":     0,
            })
            ann_id += 1
            stats["total_boxes"] += 1

        img_id += 1
        stats["images_processed"] += 1

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(coco_out, f, indent=2)

    return stats


def save_class_map(class_map: dict, output_path: str) -> None:
    """Save class map to a YAML file for reference."""
    import yaml
    with open(output_path, "w") as f:
        yaml.dump({"class_map": class_map}, f, default_flow_style=False)
    print(f"Class map saved to {output_path}")


def load_class_map(yaml_path: str) -> dict:
    """Load a previously saved class map from YAML."""
    import yaml
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data["class_map"]
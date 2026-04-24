"""
tests/test_annotation_converter.py
Run with: pytest tests/test_annotation_converter.py -v
"""

import json
import pytest
from pathlib import Path
from src.processing.annotation_converter import (
    coco_bbox_to_yolo,
    yolo_bbox_to_coco,
    convert_coco_to_yolo,
    DEFAULT_CLASS_MAP,
)


def test_coco_to_yolo_basic():
    """A box at top-left (0,0) with full image size should give center (0.5, 0.5)."""
    x_c, y_c, w, h = coco_bbox_to_yolo([0, 0, 640, 640], 640, 640)
    assert x_c == pytest.approx(0.5)
    assert y_c == pytest.approx(0.5)
    assert w   == pytest.approx(1.0)
    assert h   == pytest.approx(1.0)


def test_coco_to_yolo_quarter_box():
    """A box in the top-left quarter of a 1280x720 image."""
    x_c, y_c, w, h = coco_bbox_to_yolo([0, 0, 640, 360], 1280, 720)
    assert x_c == pytest.approx(0.25)
    assert y_c == pytest.approx(0.25)
    assert w   == pytest.approx(0.5)
    assert h   == pytest.approx(0.5)


def test_coco_to_yolo_clamping():
    """Bounding boxes that overflow image boundaries should be clamped to [0,1]."""
    x_c, y_c, w, h = coco_bbox_to_yolo([600, 600, 200, 200], 640, 640)
    assert 0.0 <= x_c <= 1.0
    assert 0.0 <= y_c <= 1.0
    assert 0.0 <= w   <= 1.0
    assert 0.0 <= h   <= 1.0


def test_roundtrip_conversion():
    """Converting COCO → YOLO → COCO should return (approximately) the original bbox."""
    original_bbox = [100, 150, 200, 300]
    img_w, img_h  = 1280, 720

    x_c, y_c, w, h  = coco_bbox_to_yolo(original_bbox, img_w, img_h)
    recovered        = yolo_bbox_to_coco(x_c, y_c, w, h, img_w, img_h)

    assert recovered[0] == pytest.approx(original_bbox[0], abs=0.1)
    assert recovered[1] == pytest.approx(original_bbox[1], abs=0.1)
    assert recovered[2] == pytest.approx(original_bbox[2], abs=0.1)
    assert recovered[3] == pytest.approx(original_bbox[3], abs=0.1)


def test_zero_area_box_ignored(tmp_path):
    """Boxes with zero width or height must not be written to the .txt file."""
    coco_data = {
        "images": [
            {"id": 1, "file_name": "frame_000001.png", "width": 640, "height": 640}
        ],
        "annotations": [
            {"image_id": 1, "category_id": 1, "bbox": [100, 100, 0, 0]},  # zero area
            {"image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 50]},  # valid
        ],
        "categories": [{"id": 1, "name": "car"}],
    }
    json_path = tmp_path / "test.json"
    json_path.write_text(json.dumps(coco_data))

    stats = convert_coco_to_yolo(str(json_path), str(tmp_path / "labels"))

    txt = (tmp_path / "labels" / "frame_000001.txt").read_text().strip().splitlines()
    assert len(txt) == 1  # only the valid box


def test_unknown_category_skipped(tmp_path):
    """Annotations with category names not in class_map should be skipped."""
    coco_data = {
        "images": [
            {"id": 1, "file_name": "frame_000001.png", "width": 640, "height": 640}
        ],
        "annotations": [
            {"image_id": 1, "category_id": 99, "bbox": [10, 10, 100, 100]},
        ],
        "categories": [{"id": 99, "name": "spaceship"}],  # not in class_map
    }
    json_path = tmp_path / "test.json"
    json_path.write_text(json.dumps(coco_data))

    stats = convert_coco_to_yolo(str(json_path), str(tmp_path / "labels"))
    assert stats["skipped_unknown_class"] >= 1


def test_image_with_no_annotations_skipped(tmp_path):
    """Images that have no annotations should not produce a .txt file."""
    coco_data = {
        "images": [
            {"id": 1, "file_name": "empty_frame.png", "width": 640, "height": 640}
        ],
        "annotations": [],
        "categories":  [{"id": 1, "name": "car"}],
    }
    json_path = tmp_path / "test.json"
    json_path.write_text(json.dumps(coco_data))

    stats = convert_coco_to_yolo(str(json_path), str(tmp_path / "labels"))

    txt_file = tmp_path / "labels" / "empty_frame.txt"
    assert not txt_file.exists()
    assert stats["skipped_no_annotations"] == 1
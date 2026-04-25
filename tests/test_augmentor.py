"""
tests/test_augmentor.py
Run with: pytest tests/test_augmentor.py -v
"""

import numpy as np
import pytest
from src.processing.augmentor import (
    yolo_to_albumentations,
    albumentations_to_yolo,
    build_train_pipeline,
    build_val_pipeline,
    augment_sample,
)


# Fixtures 

@pytest.fixture
def dummy_image():
    """640x640 random colour image."""
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_boxes():
    """Three valid YOLO boxes in a 640x640 image."""
    return [
        [0, 0.5,  0.5,  0.2, 0.2],   # car, centre
        [1, 0.2,  0.3,  0.1, 0.15],  # pedestrian, left
        [2, 0.75, 0.6,  0.1, 0.1],   # traffic light, right
    ]


# Format conversion tests 

def test_yolo_to_alb_center_box():
    """A centered box covering half the image should convert correctly."""
    alb, cls = yolo_to_albumentations([[0, 0.5, 0.5, 0.5, 0.5]])
    assert len(alb) == 1
    assert alb[0] == pytest.approx([0.25, 0.25, 0.75, 0.75])
    assert cls == [0]


def test_roundtrip_format_conversion(sample_boxes):
    """YOLO → Albumentations → YOLO should preserve values."""
    alb_boxes, class_labels = yolo_to_albumentations(sample_boxes)
    recovered = albumentations_to_yolo(alb_boxes, class_labels)

    assert len(recovered) == len(sample_boxes)
    for orig, rec in zip(sample_boxes, recovered):
        assert rec[0] == int(orig[0])              # class id
        assert rec[1] == pytest.approx(orig[1], abs=1e-4)  # x_c
        assert rec[2] == pytest.approx(orig[2], abs=1e-4)  # y_c


def test_degenerate_box_filtered():
    """Zero-area boxes must be dropped during conversion."""
    alb, cls = yolo_to_albumentations([[0, 0.5, 0.5, 0.0, 0.0]])
    assert len(alb) == 0


def test_out_of_bounds_box_clamped():
    """Boxes that overflow image boundaries must be clamped to [0,1]."""
    alb, cls = yolo_to_albumentations([[0, 0.99, 0.99, 0.5, 0.5]])
    if alb:  # may be dropped if zero-size after clamping
        x_min, y_min, x_max, y_max = alb[0]
        assert 0.0 <= x_min <= 1.0
        assert 0.0 <= y_min <= 1.0
        assert 0.0 <= x_max <= 1.0
        assert 0.0 <= y_max <= 1.0


# Pipeline tests

def test_train_pipeline_output_size(dummy_image, sample_boxes):
    """Train pipeline must output images of exactly the target size."""
    pipeline = build_train_pipeline(image_size=640)
    aug_img, aug_boxes = augment_sample(dummy_image, sample_boxes, pipeline)
    assert aug_img.shape == (640, 640, 3)


def test_val_pipeline_is_deterministic(dummy_image, sample_boxes):
    """Validation pipeline must produce identical results on repeated calls."""
    pipeline = build_val_pipeline(image_size=640)
    img1, boxes1 = augment_sample(dummy_image, sample_boxes, pipeline)
    img2, boxes2 = augment_sample(dummy_image, sample_boxes, pipeline)
    assert np.array_equal(img1, img2)


def test_augmented_boxes_within_bounds(dummy_image, sample_boxes):
    """All augmented box coordinates must stay within [0, 1]."""
    pipeline = build_train_pipeline(image_size=640)
    for _ in range(5):   # run several times since transforms are random
        _, aug_boxes = augment_sample(dummy_image, sample_boxes, pipeline)
        for box in aug_boxes:
            _, x_c, y_c, w, h = box
            assert 0.0 <= x_c <= 1.0
            assert 0.0 <= y_c <= 1.0
            assert 0.0 <= w   <= 1.0
            assert 0.0 <= h   <= 1.0


def test_augment_sample_returns_correct_types(dummy_image, sample_boxes):
    """augment_sample must return (np.ndarray, list)."""
    pipeline = build_train_pipeline(image_size=640)
    aug_img, aug_boxes = augment_sample(dummy_image, sample_boxes, pipeline)
    assert isinstance(aug_img,   np.ndarray)
    assert isinstance(aug_boxes, list)
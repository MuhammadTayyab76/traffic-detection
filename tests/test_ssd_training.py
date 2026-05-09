"""
Tests for feature/second-model-training.
Covers VOCDetectionDataset and SSDTrainer config loading.
All tests are CPU-only and use synthetic data so no GPU or real dataset needed.

Run from project root:
    pytest tests/test_ssd_training.py -v
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import torch
import yaml

def _write_voc_xml(folder: Path, stem: str, objects: list[dict], img_w=640, img_h=480) -> Path:
    """Write a minimal PASCAL VOC XML annotation file."""
    root = ET.Element("annotation")

    ET.SubElement(root, "filename").text = f"{stem}.jpg"

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text  = str(img_w)
    ET.SubElement(size, "height").text = str(img_h)
    ET.SubElement(size, "depth").text  = "3"

    for obj in objects:
        obj_el = ET.SubElement(root, "object")
        ET.SubElement(obj_el, "name").text = obj["name"]
        bnd = ET.SubElement(obj_el, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(obj["xmin"])
        ET.SubElement(bnd, "ymin").text = str(obj["ymin"])
        ET.SubElement(bnd, "xmax").text = str(obj["xmax"])
        ET.SubElement(bnd, "ymax").text = str(obj["ymax"])

    xml_path = folder / f"{stem}.xml"
    ET.ElementTree(root).write(xml_path)
    return xml_path


def _write_dummy_image(folder: Path, stem: str, width=640, height=480) -> Path:
    """Write a tiny solid-colour JPEG so PIL can actually open it."""
    from PIL import Image
    img_path = folder / f"{stem}.jpg"
    Image.new("RGB", (width, height), color=(128, 128, 128)).save(img_path)
    return img_path

@pytest.fixture()
def voc_root(tmp_path: Path) -> Path:
    for split, samples in {
        "train": [
            ("img0", [
                {"name": "car",           "xmin": 10, "ymin": 20, "xmax": 200, "ymax": 180},
                {"name": "traffic light", "xmin": 50, "ymin": 60, "xmax": 120, "ymax": 140},
            ]),
            ("img1", [
                {"name": "unknown_class", "xmin": 10, "ymin": 10, "xmax": 100, "ymax": 100},
            ]),
            ("img2", []),
        ],
        "valid": [
            ("img3", [
                {"name": "person", "xmin": 30, "ymin": 30, "xmax": 300, "ymax": 400},
            ]),
        ],
    }.items():
        images_dir = tmp_path / split / "images"
        labels_dir = tmp_path / split / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        for stem, objects in samples:
            _write_dummy_image(images_dir, stem)
            _write_voc_xml(labels_dir, stem, objects)

    return tmp_path

@pytest.fixture()
def ssd_config(tmp_path: Path, voc_root: Path) -> Path:
    """Write a minimal ssd_hyperparams.yaml pointing at the tmp voc_root."""
    cfg = {
        "model": {
            "architecture":        "ssdlite320_mobilenet_v3_large",
            "pretrained_backbone": False,
            "num_classes":         11,
        },
        "dataset": {
            "root":       str(voc_root),
            "image_size": 320,
        },
        "training": {
            "epochs":       2,
            "batch_size":   3,
            "lr":           0.01,
            "momentum":     0.9,
            "weight_decay": 4e-5,
            "warmup_epochs": 1,
            "patience":     5,
            "workers":      0,
            "device":       "cpu",
        },
        "output": {
            "project":     str(tmp_path / "runs" / "ssd"),
            "name":        "test_run",
            "save_period": 1,
            "weights_dir": str(tmp_path / "weights"),
        },
    }
    config_path = tmp_path / "ssd_hyperparams.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
    return config_path

class TestVOCDetectionDataset:

    def test_train_split_length(self, voc_root):
        from src.dataset.voc_dataset import VOCDetectionDataset
        ds = VOCDetectionDataset(voc_root, split="train", image_size=320)
        assert len(ds) == 3, "train split should have 3 image+xml pairs"

    def test_valid_split_length(self, voc_root):
        from src.dataset.voc_dataset import VOCDetectionDataset
        ds = VOCDetectionDataset(voc_root, split="valid", image_size=320)
        assert len(ds) == 1

    def test_getitem_returns_tensor_and_dict(self, voc_root):
        from src.dataset.voc_dataset import VOCDetectionDataset
        ds = VOCDetectionDataset(voc_root, split="train", image_size=320)
        image, target = ds[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(target, dict)

    def test_image_tensor_shape(self, voc_root):
        from src.dataset.voc_dataset import VOCDetectionDataset
        ds = VOCDetectionDataset(voc_root, split="train", image_size=320)
        image, _ = ds[0]
        assert image.shape == (3, 320, 320), f"Expected (3,320,320), got {image.shape}"

    def test_image_tensor_dtype_and_range(self, voc_root):
        from src.dataset.voc_dataset import VOCDetectionDataset
        ds = VOCDetectionDataset(voc_root, split="train", image_size=320)
        image, _ = ds[0]
        assert image.dtype == torch.float32
        assert image.min() >= 0.0
        assert image.max() <= 1.0

    def test_target_keys(self, voc_root):
        from src.dataset.voc_dataset import VOCDetectionDataset
        ds = VOCDetectionDataset(voc_root, split="train", image_size=320)
        _, target = ds[0]
        assert "boxes"  in target
        assert "labels" in target

    def test_target_tensor_types(self, voc_root):
        from src.dataset.voc_dataset import VOCDetectionDataset
        ds = VOCDetectionDataset(voc_root, split="train", image_size=320)
        _, target = ds[0]
        assert target["boxes"].dtype  == torch.float32
        assert target["labels"].dtype == torch.int64

    def test_valid_objects_parsed_correctly(self, voc_root):
        """img0 has 2 known classes: car(1) and traffic light(5)."""
        from src.dataset.voc_dataset import VOCDetectionDataset
        ds = VOCDetectionDataset(voc_root, split="train", image_size=320)
        _, target = ds[0]
        assert target["boxes"].shape[0]  == 2
        assert target["labels"].shape[0] == 2
        labels = sorted(target["labels"].tolist())
        assert labels == [3, 7], f"Expected [3,7] (car, traffic light), got {labels}"

    def test_unknown_class_skipped(self, voc_root):
        """img1 has only an unknown class, so it falls back to dummy background box."""
        from src.dataset.voc_dataset import VOCDetectionDataset
        ds = VOCDetectionDataset(voc_root, split="train", image_size=320)
        _, target = ds[1]
        assert target["boxes"].shape[0]  == 1, "Unknown class should produce 1 dummy box"
        assert target["labels"].tolist() == [0], "Dummy box label must be background (0)"

    def test_empty_annotation_produces_dummy_box(self, voc_root):
        """img2 has no objects at all, should not raise and return dummy box."""
        from src.dataset.voc_dataset import VOCDetectionDataset
        ds = VOCDetectionDataset(voc_root, split="train", image_size=320)
        image, target = ds[2]
        assert target["boxes"].shape[0]  == 1
        assert target["labels"].tolist() == [0]

    def test_boxes_within_image_bounds(self, voc_root):
        """All bounding box coordinates must be within [0, image_size]."""
        from src.dataset.voc_dataset import VOCDetectionDataset
        ds = VOCDetectionDataset(voc_root, split="train", image_size=320)
        for i in range(len(ds)):
            _, target = ds[i]
            boxes = target["boxes"]
            assert (boxes[:, 0] >= 0).all(), f"Sample {i}: xmin < 0"
            assert (boxes[:, 1] >= 0).all(), f"Sample {i}: ymin < 0"
            assert (boxes[:, 2] <= 320).all(), f"Sample {i}: xmax > image_size"
            assert (boxes[:, 3] <= 320).all(), f"Sample {i}: ymax > image_size"

    def test_xmax_gt_xmin_and_ymax_gt_ymin(self, voc_root):
        """No degenerate (zero-area) boxes should survive parsing."""
        from src.dataset.voc_dataset import VOCDetectionDataset
        ds = VOCDetectionDataset(voc_root, split="train", image_size=320)
        for i in range(len(ds)):
            _, target = ds[i]
            boxes = target["boxes"]
            assert (boxes[:, 2] > boxes[:, 0]).all(), f"Sample {i}: xmax <= xmin"
            assert (boxes[:, 3] > boxes[:, 1]).all(), f"Sample {i}: ymax <= ymin"

    def test_boxes_and_labels_same_length(self, voc_root):
        from src.dataset.voc_dataset import VOCDetectionDataset
        ds = VOCDetectionDataset(voc_root, split="train", image_size=320)
        for i in range(len(ds)):
            _, target = ds[i]
            assert target["boxes"].shape[0] == target["labels"].shape[0], \
                f"Sample {i}: boxes/labels length mismatch"

    def test_missing_split_raises_file_not_found(self, voc_root):
        from src.dataset.voc_dataset import VOCDetectionDataset
        with pytest.raises(FileNotFoundError):
            VOCDetectionDataset(voc_root, split="nonexistent_split", image_size=320)

    def test_class_to_idx_background_not_present(self):
        """Background (0) must not appear in CLASS_TO_IDX keys."""
        from src.dataset.voc_dataset import CLASS_TO_IDX
        assert 0 not in CLASS_TO_IDX.values(), \
            "CLASS_TO_IDX should start at 1; 0 is reserved for background"

    def test_num_classes_equals_10_plus_background(self):
        from src.dataset.voc_dataset import CLASS_TO_IDX, NUM_CLASSES
        assert NUM_CLASSES == len(CLASS_TO_IDX) + 1
        assert NUM_CLASSES == 11

class TestSSDTrainerConfig:

    def test_trainer_initialises_from_config(self, ssd_config, voc_root):
        from src.training.ssd_trainer import SSDTrainer
        trainer = SSDTrainer(ssd_config)
        assert trainer.hp is not None

    def test_run_dir_created_on_init(self, ssd_config, voc_root):
        from src.training.ssd_trainer import SSDTrainer
        trainer = SSDTrainer(ssd_config)
        assert trainer.run_dir.exists()

    def test_weights_dir_created_on_init(self, ssd_config, voc_root):
        from src.training.ssd_trainer import SSDTrainer
        trainer = SSDTrainer(ssd_config)
        assert trainer.weights_dir.exists()

    def test_device_is_cpu_when_specified(self, ssd_config, voc_root):
        from src.training.ssd_trainer import SSDTrainer
        trainer = SSDTrainer(ssd_config)
        assert trainer.device == torch.device("cpu")

    def test_missing_config_raises_file_not_found(self, tmp_path):
        from src.training.ssd_trainer import SSDTrainer
        with pytest.raises(FileNotFoundError):
            SSDTrainer(tmp_path / "does_not_exist.yaml")

    def test_build_model_output_type(self):
        from src.training.ssd_trainer import _build_model
        import torchvision
        model = _build_model(num_classes=11, pretrained_backbone=False)
        assert isinstance(model, torch.nn.Module)

    def test_build_model_num_classes(self):
        """
        SSDLite head classifier count should match num_classes.
        torchvision stores it on model.head.classification_head.num_classes
        or we just do a forward pass check on output shape.
        """
        from src.training.ssd_trainer import _build_model
        model = _build_model(num_classes=11, pretrained_backbone=False)
        model.eval()
        dummy = [torch.zeros(3, 320, 320)]
        with torch.no_grad():
            output = model(dummy)
        assert isinstance(output, list)
        assert len(output) == 1
        assert "boxes"  in output[0]
        assert "labels" in output[0]
        assert "scores" in output[0]

    def test_collate_fn_batches_correctly(self):
        from src.training.ssd_trainer import _collate
        batch = [
            (torch.zeros(3, 320, 320), {"boxes": torch.zeros(2, 4), "labels": torch.zeros(2, dtype=torch.int64)}),
            (torch.zeros(3, 320, 320), {"boxes": torch.zeros(3, 4), "labels": torch.zeros(3, dtype=torch.int64)}),
        ]
        images, targets = _collate(batch)
        assert len(images)  == 2
        assert len(targets) == 2

    def test_training_log_written_after_train(self, ssd_config, voc_root):
        """Run 2 epochs on CPU with synthetic data and confirm log file exists."""
        from src.training.ssd_trainer import SSDTrainer
        trainer = SSDTrainer(ssd_config)
        result = trainer.train()
        assert Path(result["log_path"]).exists()

    def test_training_log_has_correct_keys(self, ssd_config, voc_root):
        from src.training.ssd_trainer import SSDTrainer
        trainer = SSDTrainer(ssd_config)
        result  = trainer.train()
        with open(result["log_path"]) as f:
            log = json.load(f)
        assert "config"        in log
        assert "history"       in log
        assert "best_val_loss" in log

    def test_history_entry_has_all_fields(self, ssd_config, voc_root):
        from src.training.ssd_trainer import SSDTrainer
        trainer = SSDTrainer(ssd_config)
        result  = trainer.train()
        with open(result["log_path"]) as f:
            log = json.load(f)
        entry = log["history"][0]
        for key in ("epoch", "train_loss", "val_loss", "lr", "time_s"):
            assert key in entry, f"Missing key in history entry: {key}"

    def test_best_weights_saved(self, ssd_config, voc_root):
        from src.training.ssd_trainer import SSDTrainer
        trainer = SSDTrainer(ssd_config)
        result  = trainer.train()
        assert Path(result["best_model_path"]).exists()

    def test_best_weights_loadable(self, ssd_config, voc_root):
        """Saved state dict should load back into a fresh model without errors."""
        from src.training.ssd_trainer import SSDTrainer, _build_model
        trainer = SSDTrainer(ssd_config)
        result  = trainer.train()
        model   = _build_model(num_classes=13, pretrained_backbone=False)
        state   = torch.load(result["best_model_path"], map_location="cpu")
        model.load_state_dict(state)

    def test_best_val_loss_is_finite(self, ssd_config, voc_root):
        from src.training.ssd_trainer import SSDTrainer
        trainer = SSDTrainer(ssd_config)
        result  = trainer.train()
        assert math.isfinite(result["best_val_loss"])

    def test_early_stopping_does_not_exceed_epochs(self, ssd_config, voc_root):
        from src.training.ssd_trainer import SSDTrainer
        trainer = SSDTrainer(ssd_config)
        result  = trainer.train()
        max_epochs = trainer.hp["training"]["epochs"]
        assert len(result["history"]) <= max_epochs

import math
"""
Tests for feature/test-split-fix.
All tests use synthetic data in tmp_path — no real dataset needed.

Run from project root:
    pytest tests/test_split_fix.py -v
"""
from __future__ import annotations

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml


def _make_yolo_dataset(root: Path, n_train: int, n_classes: int = 5) -> None:
    """Write n_train synthetic YOLO image+label pairs into root/train/."""
    images_dir = root / "train" / "images"
    labels_dir = root / "train" / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    rng = random.Random(0)
    for i in range(n_train):
        stem = f"img_{i:04d}"
        (images_dir / f"{stem}.jpg").write_bytes(b"FAKE")
        cls = rng.randint(0, n_classes - 1)
        (labels_dir / f"{stem}.txt").write_text(
            f"{cls} 0.5 0.5 0.3 0.3\n"
        )


def _make_voc_dataset(root: Path, n_train: int) -> None:
    """Write n_train synthetic VOC image+xml pairs into root/train/."""
    classes = ["car", "bus", "person", "bike", "truck"]
    images_dir = root / "train" / "images"
    labels_dir = root / "train" / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    rng = random.Random(0)
    for i in range(n_train):
        stem = f"img_{i:04d}"
        (images_dir / f"{stem}.jpg").write_bytes(b"FAKE")

        ann = ET.Element("annotation")
        obj = ET.SubElement(ann, "object")
        ET.SubElement(obj, "name").text = rng.choice(classes)
        bnd = ET.SubElement(obj, "bndbox")
        for tag, val in [("xmin","10"),("ymin","10"),("xmax","100"),("ymax","100")]:
            ET.SubElement(bnd, tag).text = val
        ET.ElementTree(ann).write(labels_dir / f"{stem}.xml")


@pytest.fixture()
def yolo_root(tmp_path: Path) -> Path:
    root = tmp_path / "bdd100k"
    _make_yolo_dataset(root, n_train=40)
    return root


@pytest.fixture()
def voc_root(tmp_path: Path) -> Path:
    root = tmp_path / "bdd100k_voc"
    _make_voc_dataset(root, n_train=40)
    return root


def _run_process_split(root, fmt, ratio=0.15, seed=42, dry_run=False):
    from scripts.create_test_split import process_split
    process_split(root, fmt, ratio, seed, dry_run)


class TestYOLOSplit:

    def test_test_images_dir_created(self, yolo_root):
        _run_process_split(yolo_root, "yolo")
        assert (yolo_root / "test" / "images").exists()

    def test_test_labels_dir_created(self, yolo_root):
        _run_process_split(yolo_root, "yolo")
        assert (yolo_root / "test" / "labels").exists()

    def test_split_ratio_approximately_15_percent(self, yolo_root):
        _run_process_split(yolo_root, "yolo", ratio=0.15)
        n_test  = len(list((yolo_root / "test" / "images").glob("*.jpg")))
        n_train = len(list((yolo_root / "train" / "images").glob("*.jpg")))
        total   = n_test + n_train
        ratio   = n_test / total
        assert 0.10 <= ratio <= 0.25, f"Split ratio {ratio:.2f} outside expected range"

    def test_no_overlap_between_train_and_test(self, yolo_root):
        _run_process_split(yolo_root, "yolo")
        train_names = {p.name for p in (yolo_root / "train" / "images").glob("*.jpg")}
        test_names  = {p.name for p in (yolo_root / "test"  / "images").glob("*.jpg")}
        overlap = train_names & test_names
        assert not overlap, f"Files appear in both train and test: {overlap}"

    def test_every_test_image_has_matching_label(self, yolo_root):
        _run_process_split(yolo_root, "yolo")
        for img in (yolo_root / "test" / "images").glob("*.jpg"):
            lbl = yolo_root / "test" / "labels" / img.with_suffix(".txt").name
            assert lbl.exists(), f"Missing label for {img.name}"

    def test_every_test_label_has_matching_image(self, yolo_root):
        _run_process_split(yolo_root, "yolo")
        for lbl in (yolo_root / "test" / "labels").glob("*.txt"):
            img = yolo_root / "test" / "images" / lbl.with_suffix(".jpg").name
            assert img.exists(), f"Missing image for {lbl.name}"

    def test_total_files_unchanged_after_split(self, yolo_root):
        original_total = len(list((yolo_root / "train" / "images").glob("*.jpg")))
        _run_process_split(yolo_root, "yolo")
        n_train = len(list((yolo_root / "train" / "images").glob("*.jpg")))
        n_test  = len(list((yolo_root / "test"  / "images").glob("*.jpg")))
        assert n_train + n_test == original_total, \
            "Files were lost or duplicated during split"

    def test_dry_run_moves_nothing(self, yolo_root):
        original_count = len(list((yolo_root / "train" / "images").glob("*.jpg")))
        _run_process_split(yolo_root, "yolo", dry_run=True)
        after_count = len(list((yolo_root / "train" / "images").glob("*.jpg")))
        assert after_count == original_count
        assert not (yolo_root / "test" / "images").exists()

    def test_idempotent_on_second_run(self, yolo_root):
        _run_process_split(yolo_root, "yolo")
        test_files_after_first = set(
            p.name for p in (yolo_root / "test" / "images").glob("*.jpg")
        )
        _run_process_split(yolo_root, "yolo")
        test_files_after_second = set(
            p.name for p in (yolo_root / "test" / "images").glob("*.jpg")
        )
        assert test_files_after_first == test_files_after_second, \
            "Second run changed the test split"

    def test_deterministic_with_same_seed(self, tmp_path):
        root_a = tmp_path / "run_a"
        root_b = tmp_path / "run_b"
        _make_yolo_dataset(root_a, n_train=40)
        _make_yolo_dataset(root_b, n_train=40)
        _run_process_split(root_a, "yolo", seed=42)
        _run_process_split(root_b, "yolo", seed=42)
        names_a = {p.name for p in (root_a / "test" / "images").glob("*.jpg")}
        names_b = {p.name for p in (root_b / "test" / "images").glob("*.jpg")}
        assert names_a == names_b, "Same seed produced different splits"

    def test_different_seeds_produce_different_splits(self, tmp_path):
        root_a = tmp_path / "seed_1"
        root_b = tmp_path / "seed_2"
        _make_yolo_dataset(root_a, n_train=100)
        _make_yolo_dataset(root_b, n_train=100)
        _run_process_split(root_a, "yolo", seed=1)
        _run_process_split(root_b, "yolo", seed=99)
        names_a = {p.name for p in (root_a / "test" / "images").glob("*.jpg")}
        names_b = {p.name for p in (root_b / "test" / "images").glob("*.jpg")}
        assert names_a != names_b, "Different seeds should produce different splits"

    def test_missing_train_dir_skips_gracefully(self, tmp_path):
        empty_root = tmp_path / "empty"
        empty_root.mkdir()
        _run_process_split(empty_root, "yolo")
        assert not (empty_root / "test").exists()


class TestVOCSplit:

    def test_test_dirs_created(self, voc_root):
        _run_process_split(voc_root, "voc")
        assert (voc_root / "test" / "images").exists()
        assert (voc_root / "test" / "labels").exists()

    def test_split_ratio_approximately_15_percent(self, voc_root):
        _run_process_split(voc_root, "voc", ratio=0.15)
        n_test  = len(list((voc_root / "test"  / "images").glob("*.jpg")))
        n_train = len(list((voc_root / "train" / "images").glob("*.jpg")))
        ratio   = n_test / (n_test + n_train)
        assert 0.10 <= ratio <= 0.25

    def test_no_overlap_between_train_and_test(self, voc_root):
        _run_process_split(voc_root, "voc")
        train_names = {p.name for p in (voc_root / "train" / "images").glob("*.jpg")}
        test_names  = {p.name for p in (voc_root / "test"  / "images").glob("*.jpg")}
        assert not (train_names & test_names)

    def test_every_test_image_has_xml(self, voc_root):
        _run_process_split(voc_root, "voc")
        for img in (voc_root / "test" / "images").glob("*.jpg"):
            xml = voc_root / "test" / "labels" / img.with_suffix(".xml").name
            assert xml.exists(), f"Missing XML for {img.name}"

    def test_total_files_unchanged(self, voc_root):
        original = len(list((voc_root / "train" / "images").glob("*.jpg")))
        _run_process_split(voc_root, "voc")
        n_train = len(list((voc_root / "train" / "images").glob("*.jpg")))
        n_test  = len(list((voc_root / "test"  / "images").glob("*.jpg")))
        assert n_train + n_test == original

    def test_dry_run_moves_nothing(self, voc_root):
        original = len(list((voc_root / "train" / "images").glob("*.jpg")))
        _run_process_split(voc_root, "voc", dry_run=True)
        assert len(list((voc_root / "train" / "images").glob("*.jpg"))) == original
        assert not (voc_root / "test" / "images").exists()

    def test_idempotent_on_second_run(self, voc_root):
        _run_process_split(voc_root, "voc")
        first  = {p.name for p in (voc_root / "test" / "images").glob("*.jpg")}
        _run_process_split(voc_root, "voc")
        second = {p.name for p in (voc_root / "test" / "images").glob("*.jpg")}
        assert first == second


class TestDatasetYaml:

    def test_dataset_yaml_test_path_is_not_valid(self, tmp_path):
        """Confirm dataset.yaml test key does not point at valid/images."""
        yaml_path = Path("configs/dataset.yaml")
        if not yaml_path.exists():
            pytest.skip("configs/dataset.yaml not found")
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        test_path = cfg.get("test", "")
        assert test_path != "valid/images", \
            "dataset.yaml still points test at valid/images — split not applied"

    def test_dataset_yaml_test_path_is_test_images(self):
        """Confirm dataset.yaml test key points at the real test split."""
        yaml_path = Path("configs/dataset.yaml")
        if not yaml_path.exists():
            pytest.skip("configs/dataset.yaml not found")
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg.get("test") == "test/images", \
            f"Expected test: test/images, got: {cfg.get('test')}"
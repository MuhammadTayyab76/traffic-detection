"""
Carves a held-out test split from the training data for both datasets.

Moves 15% of training images+labels into a new test/ folder.
Uses a fixed random seed and stratifies by the most frequent class
per image so the class distribution is preserved.

Usage (from project root, conda env active):
    python scripts/create_test_split.py
    python scripts/create_test_split.py --ratio 0.15 --seed 42 --dry-run

DESTRUCTIVE: files are moved out of train/, not copied.
Run once. Idempotent: skips if test/ already contains files.
"""
from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import List, Tuple


YOLO_ROOT = Path("data/bdd100k")
VOC_ROOT  = Path("data/bdd100k_voc")


def _primary_class_yolo(label_path: Path) -> str:
    """Return the most frequent class ID string in a YOLO .txt file."""
    lines = label_path.read_text().strip().splitlines()
    classes = [ln.split()[0] for ln in lines if ln.strip()]
    if not classes:
        return "empty"
    return Counter(classes).most_common(1)[0][0]


def _primary_class_voc(label_path: Path) -> str:
    """Return the most frequent class name in a PASCAL VOC .xml file."""
    try:
        tree = ET.parse(label_path)
        names = [obj.find("name").text.strip().lower()
                 for obj in tree.getroot().findall("object")]
    except ET.ParseError:
        return "empty"
    if not names:
        return "empty"
    return Counter(names).most_common(1)[0][0]


def _stratified_sample(
    pairs: List[Tuple[Path, Path]],
    strata: List[str],
    ratio: float,
    seed: int,
) -> List[Tuple[Path, Path]]:
    """
    Stratified random sample without sklearn dependency.
    Groups pairs by stratum, samples `ratio` from each group,
    guarantees at least 1 sample per group when possible.
    """
    rng = random.Random(seed)
    groups: dict[str, list] = {}
    for pair, stratum in zip(pairs, strata):
        groups.setdefault(stratum, []).append(pair)

    selected = []
    for stratum, group_pairs in groups.items():
        rng.shuffle(group_pairs)
        n = max(1, round(len(group_pairs) * ratio))
        n = min(n, len(group_pairs))
        selected.extend(group_pairs[:n])

    return selected


def _class_distribution(pairs: List[Tuple[Path, Path]], fmt: str) -> Counter:
    counts: Counter = Counter()
    for _, label_path in pairs:
        if fmt == "yolo":
            lines = label_path.read_text().strip().splitlines()
            for ln in lines:
                if ln.strip():
                    counts[ln.split()[0]] += 1
        else:
            try:
                tree = ET.parse(label_path)
                for obj in tree.getroot().findall("object"):
                    counts[obj.find("name").text.strip().lower()] += 1
            except ET.ParseError:
                pass
    return counts


def process_split(
    root: Path,
    fmt: str,
    ratio: float,
    seed: int,
    dry_run: bool,
) -> None:
    label_ext = ".txt" if fmt == "yolo" else ".xml"

    train_images = root / "train" / "images"
    train_labels = root / "train" / "labels"
    test_images  = root / "test"  / "images"
    test_labels  = root / "test"  / "labels"

    if not train_images.exists():
        print(f"  [SKIP] {train_images} does not exist.")
        return

    existing_test = list(test_images.glob("*")) if test_images.exists() else []
    if existing_test:
        print(f"  [SKIP] {test_images} already contains {len(existing_test)} files. "
              "Delete test/ manually to re-run.")
        return

    all_images = sorted(train_images.glob("*.jpg")) + \
                 sorted(train_images.glob("*.jpeg")) + \
                 sorted(train_images.glob("*.png"))

    pairs: List[Tuple[Path, Path]] = []
    for img in all_images:
        lbl = train_labels / img.with_suffix(label_ext).name
        if lbl.exists():
            pairs.append((img, lbl))

    if not pairs:
        print(f"  [SKIP] No image+label pairs found in {train_images}.")
        return

    strata = [
        _primary_class_yolo(lbl) if fmt == "yolo" else _primary_class_voc(lbl)
        for _, lbl in pairs
    ]

    test_pairs = _stratified_sample(pairs, strata, ratio, seed)
    n_test  = len(test_pairs)
    n_total = len(pairs)

    print(f"\n  Dataset  : {root.name} ({fmt.upper()})")
    print(f"  Total    : {n_total} train pairs")
    print(f"  Moving   : {n_test} to test ({n_test/n_total*100:.1f}%)")

    before_dist = _class_distribution(pairs, fmt)
    test_dist   = _class_distribution(test_pairs, fmt)
    print(f"  Class distribution (train full): {dict(before_dist.most_common())}")
    print(f"  Class distribution (test slice): {dict(test_dist.most_common())}")

    if dry_run:
        print("  [DRY RUN] No files moved.")
        return

    test_images.mkdir(parents=True, exist_ok=True)
    test_labels.mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path in test_pairs:
        shutil.move(str(img_path), str(test_images / img_path.name))
        shutil.move(str(lbl_path), str(test_labels / lbl_path.name))

    remaining = len(list(train_images.glob("*.jpg"))) + \
                len(list(train_images.glob("*.jpeg"))) + \
                len(list(train_images.glob("*.png")))
    print(f"  Done. Train remaining: {remaining} | Test: {n_test}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create held-out test splits")
    parser.add_argument("--ratio",   type=float, default=0.15)
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without moving files")
    args = parser.parse_args()

    print(f"Test split ratio : {args.ratio}")
    print(f"Random seed      : {args.seed}")
    print(f"Dry run          : {args.dry_run}")

    process_split(YOLO_ROOT, "yolo", args.ratio, args.seed, args.dry_run)
    process_split(VOC_ROOT,  "voc",  args.ratio, args.seed, args.dry_run)

    if not args.dry_run:
        print("\nDone. Update configs/dataset.yaml if not already pointing to test/images.")


if __name__ == "__main__":
    main()
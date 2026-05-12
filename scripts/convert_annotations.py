"""
scripts/convert_annotations.py

Usage examples:

  # COCO JSON → YOLO .txt
  python scripts/convert_annotations.py coco-to-yolo \
      --json data/annotations/instances_train.json \
      --output data/annotations/labels/

  # YOLO .txt → COCO JSON
  python scripts/convert_annotations.py yolo-to-coco \
      --labels data/annotations/labels/ \
      --images data/frames/ \
      --output data/annotations/coco_converted.json
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.processing.annotation_converter import (
    convert_coco_to_yolo,
    convert_yolo_to_coco,
    DEFAULT_CLASS_MAP,
)


def cmd_coco_to_yolo(args):
    print("\n── COCO JSON → YOLO .txt ──────────────────────")
    print(f"  Input  : {args.json}")
    print(f"  Output : {args.output}\n")

    stats = convert_coco_to_yolo(
        coco_json_path=args.json,
        output_dir=args.output,
    )

    print(f"\n── Results ─────────────────────────────────────")
    print(f"  Images converted      : {stats['converted']}")
    print(f"  Skipped (no labels)   : {stats['skipped_no_annotations']}")
    print(f"  Skipped (unknown cls) : {stats['skipped_unknown_class']}")
    print(f"  Total boxes written   : {stats['total_boxes']}")
    print(f"────────────────────────────────────────────────\n")


def cmd_yolo_to_coco(args):
    print("\n── YOLO .txt → COCO JSON ──────────────────────")
    print(f"  Labels : {args.labels}")
    print(f"  Images : {args.images}")
    print(f"  Output : {args.output}\n")

    stats = convert_yolo_to_coco(
        yolo_labels_dir=args.labels,
        images_dir=args.images,
        output_json_path=args.output,
    )

    print(f"\n── Results ─────────────────────────────────────")
    print(f"  Images processed : {stats['images_processed']}")
    print(f"  Total boxes      : {stats['total_boxes']}")
    print(f"────────────────────────────────────────────────\n")


def main():
    parser = argparse.ArgumentParser(
        description="Annotation format converter — COCO ↔ YOLO"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # coco-to-yolo subcommand
    p1 = sub.add_parser("coco-to-yolo", help="Convert COCO JSON to YOLO .txt")
    p1.add_argument("--json",   required=True, help="Path to COCO .json file")
    p1.add_argument("--output", required=True, help="Output folder for .txt files")

    # yolo-to-coco subcommand
    p2 = sub.add_parser("yolo-to-coco", help="Convert YOLO .txt to COCO JSON")
    p2.add_argument("--labels", required=True, help="Folder containing YOLO .txt files")
    p2.add_argument("--images", required=True, help="Folder containing image files")
    p2.add_argument("--output", required=True, help="Output path for COCO .json")

    args = parser.parse_args()

    if args.command == "coco-to-yolo":
        cmd_coco_to_yolo(args)
    elif args.command == "yolo-to-coco":
        cmd_yolo_to_coco(args)


if __name__ == "__main__":
    main()
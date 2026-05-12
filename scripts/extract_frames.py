"""
scripts/extract_frames.py

Usage:
    python scripts/extract_frames.py --source data/raw/clip01.mp4
    python scripts/extract_frames.py --source data/raw/clip01.mp4 --fps 3 --blur 80
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ingestion.video_loader import load_video, get_video_info
from src.processing.frame_extractor import extract_frames


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from traffic video")
    parser.add_argument("--source",    required=True,       help="Path to video file or RTSP URL")
    parser.add_argument("--output",    default="data/frames", help="Output folder for frames")
    parser.add_argument("--fps",       type=float, default=5.0, help="Frames per second to extract")
    parser.add_argument("--blur",      type=float, default=100.0, help="Blur rejection threshold")
    parser.add_argument("--max",       type=int,   default=None, help="Max frames to save")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*50}")
    print(f"  Traffic Detection — Frame Extractor")
    print(f"{'='*50}")
    print(f"  Source : {args.source}")
    print(f"  Output : {args.output}")
    print(f"  FPS    : {args.fps}")
    print(f"  Blur θ : {args.blur}")
    print(f"{'='*50}\n")

    cap  = load_video(args.source)
    info = get_video_info(cap)
    print(f"Video info: {info}\n")

    stats = extract_frames(
        cap=cap,
        output_dir=args.output,
        target_fps=args.fps,
        blur_threshold=args.blur,
        max_frames=args.max,
    )
    cap.release()

    print(f"\n{'='*50}")
    print(f"  Done!")
    print(f"  Frames saved    : {stats['saved']}")
    print(f"  Rejected (blur) : {stats['rejected_blur']}")
    print(f"  Rejected (corrupt): {stats['rejected_corrupt']}")
    print(f"  Total read      : {stats['total_read']}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
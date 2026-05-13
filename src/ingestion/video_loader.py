"""
video_loader.py
Validates and opens a video source (local file or RTSP stream).
"""

import cv2
from pathlib import Path


SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}


def load_video(source: str) -> cv2.VideoCapture:
    # Local file validation
    if not source.startswith("rtsp://"):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {source}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{path.suffix}'. "
                f"Supported: {SUPPORTED_EXTENSIONS}"
            )

    # Open with OpenCV
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open source: {source}")

    return cap


def get_video_info(cap: cv2.VideoCapture) -> dict:
    return {
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":          cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
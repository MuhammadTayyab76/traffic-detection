"""
frame_extractor.py
Extracts, filters, and saves frames from a video source.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def _is_blurry(frame: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Returns True if the frame is too blurry to be useful.
    Uses the variance of the Laplacian as a sharpness score.
    Lower variance = blurrier image.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < threshold


def extract_frames(
    cap: cv2.VideoCapture,
    output_dir: str,
    target_fps: float = 5.0,
    blur_threshold: float = 100.0,
    max_frames: int = None,
) -> dict:
    """
    Extract frames from a VideoCapture and save clean ones to disk.

    Args:
        cap:             Opened cv2.VideoCapture object.
        output_dir:      Folder where PNG frames will be saved.
        target_fps:      How many frames per second to extract.
                         E.g. 5.0 means one frame every 0.2 seconds.
        blur_threshold:  Laplacian variance below this = frame rejected.
        max_frames:      Stop after saving this many frames (None = no limit).

    Returns:
        dict with keys: saved, rejected_blur, rejected_corrupt, total_read
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    source_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # How many source frames to skip between each extraction
    frame_interval = max(1, int(round(source_fps / target_fps)))

    stats = {"saved": 0, "rejected_blur": 0,
             "rejected_corrupt": 0, "total_read": 0}

    print(f"Source FPS: {source_fps:.1f} | "
          f"Target FPS: {target_fps} | "
          f"Saving every {frame_interval} frames")

    with tqdm(total=total_frames, desc="Extracting frames", unit="fr") as pbar:
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            stats["total_read"] += 1
            pbar.update(1)

            # Only process frames at the chosen interval
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            if frame is None or frame.size == 0:
                stats["rejected_corrupt"] += 1
                frame_idx += 1
                continue

            if _is_blurry(frame, blur_threshold):
                stats["rejected_blur"] += 1
                frame_idx += 1
                continue

            # Save the frame
            filename = out_path / f"frame_{stats['saved']:06d}.png"
            cv2.imwrite(str(filename), frame)
            stats["saved"] += 1

            if max_frames and stats["saved"] >= max_frames:
                break

            frame_idx += 1

    return stats
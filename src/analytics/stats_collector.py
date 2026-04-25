"""
stats_collector.py

Collects, aggregates, and logs per-frame detection statistics.

Receives a list of Detection objects each frame and maintains:
  - Per-frame counts per class
  - Rolling average confidence per class
  - Session-level totals
  - JSON log of every frame (for post-analysis)

This module is intentionally stateful — instantiate once per
video session and call update() on every frame.
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# Detection dataclass 

@dataclass
class Detection:
    """
    Represents a single detected object in one frame.
    This is the currency passed between inference, analytics, and GUI.
    """
    class_id:    int
    class_name:  str
    confidence:  float
    x_center:    float    # normalised [0,1]
    y_center:    float
    width:       float
    height:      float
    track_id:    Optional[int] = None   # assigned by tracker, None if no tracker


# Per-frame snapshot 

@dataclass
class FrameStats:
    """Statistics snapshot for a single frame."""
    frame_idx:        int
    timestamp_ms:     float
    total_detections: int
    counts:           dict   # class_name → count
    avg_confidence:   dict   # class_name → avg confidence
    fps:              float


# Stats collector 

class StatsCollector:
    """
    Stateful per-session statistics collector.

    Usage:
        collector = StatsCollector(class_names=[...])
        for frame in video:
            detections = model.detect(frame)
            stats = collector.update(detections)
            gui.update_sidebar(stats)
        collector.save_log("session_log.json")
    """

    def __init__(
        self,
        class_names: list[str],
        log_every_n_frames: int = 1,
        rolling_window: int = 30,
    ):
        """
        Args:
            class_names:        List of class name strings matching model output.
            log_every_n_frames: Save to internal log every N frames.
                                1 = every frame, 30 = every second at 30fps.
            rolling_window:     Number of frames to use for rolling averages.
        """
        self.class_names        = class_names
        self.log_every_n_frames = log_every_n_frames
        self.rolling_window     = rolling_window

        # Session state
        self._frame_idx         = 0
        self._session_start     = time.time()
        self._last_frame_time   = time.time()

        # Counters
        self._session_counts    = defaultdict(int)   # total detections per class
        self._frame_log: list[dict] = []

        # Rolling window for FPS calculation
        self._frame_times: list[float] = []

        # Rolling window for confidence
        self._conf_history: dict[str, list[float]] = {
            name: [] for name in class_names
        }

    # Core Update Method

    def update(self, detections: list[Detection]) -> FrameStats:
        """
        Process detections for one frame and return a FrameStats snapshot.

        Args:
            detections: List of Detection objects from the inference engine.

        Returns:
            FrameStats for the current frame.
        """
        now = time.time()

        # FPS calculation using rolling window
        self._frame_times.append(now)
        if len(self._frame_times) > self.rolling_window:
            self._frame_times.pop(0)
        if len(self._frame_times) > 1:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            fps = (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0.0
        else:
            fps = 0.0

        # Per-frame counts & confidence
        frame_counts: dict[str, int]   = defaultdict(int)
        frame_confs:  dict[str, list]  = defaultdict(list)

        for det in detections:
            frame_counts[det.class_name] += 1
            frame_confs[det.class_name].append(det.confidence)
            self._session_counts[det.class_name] += 1

        # Update rolling confidence history
        for name in self.class_names:
            confs = frame_confs.get(name, [])
            self._conf_history[name].extend(confs)
            if len(self._conf_history[name]) > self.rolling_window * 10:
                self._conf_history[name] = self._conf_history[name][-self.rolling_window * 10:]

        # Average confidence per class this frame
        avg_conf = {}
        for name in self.class_names:
            confs = frame_confs.get(name, [])
            avg_conf[name] = round(sum(confs) / len(confs), 4) if confs else 0.0

        stats = FrameStats(
            frame_idx        = self._frame_idx,
            timestamp_ms     = round((now - self._session_start) * 1000, 2),
            total_detections = len(detections),
            counts           = dict(frame_counts),
            avg_confidence   = avg_conf,
            fps              = round(fps, 2),
        )

        # Log every N frames
        if self._frame_idx % self.log_every_n_frames == 0:
            self._frame_log.append(asdict(stats))

        self._frame_idx    += 1
        self._last_frame_time = now
        return stats

    # Session summary 

    def get_session_summary(self) -> dict:
        """
        Return aggregate statistics for the entire session so far.

        Returns:
            dict with total frames, session duration, per-class totals.
        """
        duration = time.time() - self._session_start
        return {
            "total_frames":    self._frame_idx,
            "duration_seconds": round(duration, 2),
            "avg_fps":         round(self._frame_idx / duration, 2) if duration > 0 else 0,
            "session_counts":  dict(self._session_counts),
            "most_detected":   max(self._session_counts, key=self._session_counts.get)
                               if self._session_counts else None,
        }

    # Persistence 

    def save_log(self, output_path: str) -> None:
        """
        Save the full frame log to a JSON file.

        Args:
            output_path: Where to write the JSON log.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "session_summary": self.get_session_summary(),
            "frames":          self._frame_log,
        }
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Session log saved → {out}")

    def reset(self) -> None:
        """Reset all state for a new session."""
        self._frame_idx       = 0
        self._session_start   = time.time()
        self._last_frame_time = time.time()
        self._session_counts  = defaultdict(int)
        self._frame_log       = []
        self._frame_times     = []
        self._conf_history    = {name: [] for name in self.class_names}
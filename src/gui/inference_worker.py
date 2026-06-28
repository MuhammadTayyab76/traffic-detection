import time
import cv2
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker

from src.ingestion.video_loader import load_video, get_video_info
from src.inference.tracker import SimpleTracker
from src.inference.postprocessor import Postprocessor
from src.inference.detector import YOLODetector, SSDDetector
from src.analytics.stats_collector import StatsCollector

CLASS_NAMES = [
    'background', 'bike', 'bus', 'car', 'motor', 'person',
    'rider', 'traffic light', 'traffic sign', 'train', 'truck'
]

WEIGHTS_PATHS = {
    "yolo": "weights/yolo_best.pt",
    "ssd":  "weights/ssd_best.pt",
}


class InferenceWorker(QThread):
    frame_ready        = pyqtSignal(object)
    stats_ready        = pyqtSignal(object)
    video_info_ready   = pyqtSignal(dict)
    frame_idx_changed  = pyqtSignal(int)
    error_occurred     = pyqtSignal(str)
    finished_processing = pyqtSignal()

    def __init__(self, video_path: str, model_name: str = "yolo", parent=None):
        super().__init__(parent)
        self.video_path          = video_path
        self._model_name         = model_name
        self._pending_model_name = model_name
        self._threshold          = 0.25
        self._speed              = 1.0
        self._running            = True
        self._paused             = False
        self._pending_seek       = None
        self._mutex              = QMutex()

    def toggle_pause(self):
        with QMutexLocker(self._mutex):
            self._paused = not self._paused

    def set_threshold(self, value: float):
        with QMutexLocker(self._mutex):
            self._threshold = value

    def set_speed(self, multiplier: float):
        with QMutexLocker(self._mutex):
            self._speed = multiplier

    def set_model(self, model_name: str):
        with QMutexLocker(self._mutex):
            self._pending_model_name = model_name

    def seek(self, frame_idx: int):
        with QMutexLocker(self._mutex):
            self._pending_seek = frame_idx

    def stop(self):
        with QMutexLocker(self._mutex):
            self._running = False

    def _build_detector(self, model_name: str):
        return (YOLODetector if model_name == "yolo" else SSDDetector)(
            weights_path=WEIGHTS_PATHS[model_name]
        )

    @staticmethod
    def _frames_to_skip(speed: float) -> int:
        """
        Number of frames to grab-and-discard between each decoded frame.
        0.5x / 1x -> 0 skips (slowdown handled by sleep alone)
        1.5x      -> 1 skip  (decode every 2nd frame  ~ 1.5x faster)
        2x        -> 1 skip  (decode every 2nd frame  = 2x faster)
        Works cleanly for the four options in the combo box.
        """
        if speed <= 1.0:
            return 0
        return max(1, round(speed) - 1)

    def run(self):
        try:
            cap = load_video(self.video_path)
        except Exception as e:
            self.error_occurred.emit(str(e))
            return

        video_info = get_video_info(cap)
        self.video_info_ready.emit(video_info)
        fps            = video_info["fps"] or 30.0
        frame_duration = 1.0 / fps          # seconds per frame at 1x

        detector      = self._build_detector(self._model_name)
        tracker       = SimpleTracker()
        postprocessor = Postprocessor()
        stats_collector = StatsCollector(class_names=CLASS_NAMES)

        while True:
            with QMutexLocker(self._mutex):
                running       = self._running
                paused        = self._paused
                threshold     = self._threshold
                speed         = self._speed
                pending_model = self._pending_model_name
                pending_seek  = self._pending_seek
                self._pending_seek = None

            if not running:
                break

            if pending_model != self._model_name:
                detector         = self._build_detector(pending_model)
                tracker          = SimpleTracker()
                self._model_name = pending_model

            if pending_seek is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pending_seek)
                tracker = SimpleTracker()

            if paused:
                self.msleep(50)
                continue

            # --- Frame skipping for speed > 1x ---
            # cap.grab() advances the decode position without returning pixels,
            # so it is much cheaper than cap.read() for discarded frames.
            skip = self._frames_to_skip(speed)
            for _ in range(skip):
                if not cap.grab():
                    break

            t_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            detections = [d for d in detector.detect(frame) if d.confidence >= threshold]
            tracked    = tracker.update(detections)
            stats      = stats_collector.update(tracked)
            out_frame  = postprocessor.draw_boxes(frame, tracked)

            self.frame_ready.emit(out_frame)
            self.stats_ready.emit(stats)
            self.frame_idx_changed.emit(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))

            elapsed   = time.perf_counter() - t_start
            target    = frame_duration * (1 + skip) / max(speed, 0.01)
            remaining = target - elapsed
            if remaining > 0.005:   # ignore sub-5ms remainders
                self.msleep(int(remaining * 1000))

        cap.release()
        stats_collector.save_log("evaluation_outputs/gui_session_stats.json")
        self.finished_processing.emit()
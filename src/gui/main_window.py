import os
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox
from src.gui.video_widget import VideoWidget
from src.gui.controls import ControlsPanel
from src.gui.stats_panel import StatsPanel
from src.gui.inference_worker import InferenceWorker, CLASS_NAMES


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Detection System")
        self.resize(1280, 760)
        self.worker = None
        self._is_playing = False
        self._current_video_path = None
        self._current_model_name = "yolo"
        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        central = QWidget()
        content_layout = QHBoxLayout(central)

        left_layout = QVBoxLayout()
        self.video_widget = VideoWidget()
        left_layout.addWidget(self.video_widget)

        self.controls_panel = ControlsPanel()
        left_layout.addWidget(self.controls_panel)

        content_layout.addLayout(left_layout, stretch=3)

        self.stats_panel = StatsPanel(class_names=CLASS_NAMES)
        content_layout.addWidget(self.stats_panel, stretch=1)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    def _connect_signals(self):
        self.controls_panel.video_selected.connect(self._on_video_selected)
        self.controls_panel.play_pause_clicked.connect(self._on_play_pause_clicked)
        self.controls_panel.model_changed.connect(self._on_model_changed)
        self.controls_panel.threshold_changed.connect(self._on_threshold_changed)
        self.controls_panel.seek_changed.connect(self._on_seek_changed)
        self.controls_panel.speed_changed.connect(self._on_speed_changed)

    def _on_video_selected(self, path: str):
        self._current_video_path = path
        self._current_model_name = self.controls_panel.model_combo.currentText()
        self._start_worker(seek_to=0)

    def _start_worker(self, seek_to: int = 0):
        """Stop any existing worker and start a fresh one from seek_to."""
        self._stop_worker()
        self.video_widget.clear_frame()
        self.stats_panel.reset()

        self.worker = InferenceWorker(
            video_path=self._current_video_path,
            model_name=self._current_model_name
        )

        self.worker.frame_ready.connect(self.video_widget.set_frame)
        self.worker.stats_ready.connect(self.stats_panel.update_stats)
        self.worker.video_info_ready.connect(self._on_video_info_ready)
        self.worker.frame_idx_changed.connect(self.controls_panel.update_seek_position)
        self.worker.error_occurred.connect(self._on_worker_error)
        self.worker.finished_processing.connect(self._on_worker_finished)

        if seek_to > 0:
            self.worker.seek(seek_to)

        self.worker.start()
        self._is_playing = True
        self.controls_panel.set_play_pause_text(True)
        self.statusBar().showMessage(
            f"Playing: {os.path.basename(self._current_video_path)}"
        )

    def _on_video_info_ready(self, info: dict):
        self.controls_panel.set_seek_range(info["total_frames"])

    def _on_play_pause_clicked(self):
        if self._current_video_path is None:
            return

        # Worker is dead (playback finished) — restart from the beginning
        if self.worker is None or not self.worker.isRunning():
            self._start_worker(seek_to=0)
            self.controls_panel.update_seek_position(0)
            return

        self.worker.toggle_pause()
        self._is_playing = not self._is_playing
        self.controls_panel.set_play_pause_text(self._is_playing)

    def _on_model_changed(self, model_name: str):
        self._current_model_name = model_name
        if self.worker is not None and self.worker.isRunning():
            self.worker.set_model(model_name)

    def _on_threshold_changed(self, value: float):
        if self.worker is not None and self.worker.isRunning():
            self.worker.set_threshold(value)

    def _on_seek_changed(self, frame_idx: int):
        if self._current_video_path is None:
            return

        # Worker is dead (playback finished) — restart from the seeked position
        if self.worker is None or not self.worker.isRunning():
            self._start_worker(seek_to=frame_idx)
            return

        self.worker.seek(frame_idx)

    def _on_speed_changed(self, multiplier: float):
        if self.worker is not None and self.worker.isRunning():
            self.worker.set_speed(multiplier)

    def _on_worker_error(self, message: str):
        self._stop_worker()
        QMessageBox.critical(self, "Video Error", message)
        self.statusBar().showMessage("Error loading video")

    def _on_worker_finished(self):
        # wait() blocks until run() fully returns — safe here since finished_processing
        # is emitted as the very last line of run(), so this is near-instant
        if self.worker is not None:
            self.worker.wait()
            self.worker = None
        self._is_playing = False
        self.controls_panel.set_play_pause_text(False)
        self.statusBar().showMessage(
            "Playback finished — press Play or seek to restart"
        )

    def _stop_worker(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        self.worker = None

    def closeEvent(self, event):
        self._stop_worker()
        event.accept()
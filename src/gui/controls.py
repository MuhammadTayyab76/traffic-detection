from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QComboBox,
    QSlider, QLabel, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal


class ControlsPanel(QWidget):
    """Emits signals only — MainWindow/InferenceWorker decide what to do with them."""

    video_selected = pyqtSignal(str)
    play_pause_clicked = pyqtSignal()
    model_changed = pyqtSignal(str)        # "yolo" or "ssd"
    threshold_changed = pyqtSignal(float)  # 0.0 - 1.0
    seek_changed = pyqtSignal(int)         # frame index, from slider release
    speed_changed = pyqtSignal(float)      # playback multiplier

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Row 1: upload, model select, play/pause ---
        row1 = QHBoxLayout()

        self.upload_btn = QPushButton("Upload Video")
        self.upload_btn.clicked.connect(self._on_upload_clicked)
        row1.addWidget(self.upload_btn)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolo", "ssd"])
        self.model_combo.currentTextChanged.connect(self.model_changed.emit)
        row1.addWidget(QLabel("Model:"))
        row1.addWidget(self.model_combo)

        self.play_pause_btn = QPushButton("Play")
        self.play_pause_btn.clicked.connect(self.play_pause_clicked.emit)
        row1.addWidget(self.play_pause_btn)

        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "1x", "1.5x", "2x"])
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)
        row1.addWidget(QLabel("Speed:"))
        row1.addWidget(self.speed_combo)

        main_layout.addLayout(row1)

        # --- Row 2: confidence threshold slider ---
        row2 = QHBoxLayout()
        self.threshold_label = QLabel("Confidence: 0.25")
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(25)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        row2.addWidget(self.threshold_label)
        row2.addWidget(self.threshold_slider)
        main_layout.addLayout(row2)

        # --- Row 3: seek slider ---
        row3 = QHBoxLayout()
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 0)  # set real max once video is loaded
        # sliderReleased avoids flooding seek requests while dragging
        self.seek_slider.sliderReleased.connect(
            lambda: self.seek_changed.emit(self.seek_slider.value())
        )
        row3.addWidget(QLabel("Seek:"))
        row3.addWidget(self.seek_slider)
        main_layout.addLayout(row3)

    def _on_upload_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov)"
        )
        if path:
            self.video_selected.emit(path)

    def _on_threshold_changed(self, value: int):
        conf = value / 100.0
        self.threshold_label.setText(f"Confidence: {conf:.2f}")
        self.threshold_changed.emit(conf)

    def _on_speed_changed(self, text: str):
        multiplier = float(text.replace("x", ""))
        self.speed_changed.emit(multiplier)

    def set_play_pause_text(self, is_playing: bool):
        self.play_pause_btn.setText("Pause" if is_playing else "Play")

    def set_seek_range(self, total_frames: int):
        self.seek_slider.setRange(0, max(total_frames - 1, 0))

    def update_seek_position(self, frame_idx: int):
        # blockSignals avoids re-triggering sliderReleased while we're just syncing position
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(frame_idx)
        self.seek_slider.blockSignals(False)
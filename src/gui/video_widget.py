import cv2
import numpy as np
from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt


class VideoWidget(QLabel):
    """Displays cv2 BGR frames inside a QLabel, scaled to fit while keeping aspect ratio."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.setMinimumSize(640, 360)
        self._current_pixmap = None

    def set_frame(self, frame: np.ndarray):
        if frame is None:
            return

        # cv2 frames are BGR — convert to RGB for QImage
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._current_pixmap = QPixmap.fromImage(qimg)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if self._current_pixmap is None:
            return
        scaled = self._current_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        # Re-scale on window resize so video doesn't stay stretched to old size
        self._update_scaled_pixmap()
        super().resizeEvent(event)

    def clear_frame(self):
        self._current_pixmap = None
        self.clear()
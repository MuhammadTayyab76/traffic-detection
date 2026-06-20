from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
from src.gui.video_widget import VideoWidget
from src.gui.controls import ControlsPanel
from src.gui.stats_panel import StatsPanel

# Mirrors the class list used in run_demo.py / SSDDetector until a shared class_map exists
CLASS_NAMES = [
    'background', 'bike', 'bus', 'car', 'motor', 'person',
    'rider', 'traffic light', 'traffic sign', 'train', 'truck'
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Detection System")
        self.resize(1280, 760)
        self._build_ui()

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
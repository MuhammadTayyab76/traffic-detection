from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from src.gui.video_widget import VideoWidget
from src.gui.controls import ControlsPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Detection System")
        self.resize(1280, 760)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)

        self.video_widget = VideoWidget()
        layout.addWidget(self.video_widget)

        self.controls_panel = ControlsPanel()
        
        layout.addWidget(self.controls_panel)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")
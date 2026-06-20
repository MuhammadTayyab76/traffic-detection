from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QHeaderView
)


class StatsPanel(QWidget):
    """Read-only sidebar — fed by StatsCollector.FrameStats objects via update_stats()."""

    def __init__(self, class_names: list[str], parent=None):
        super().__init__(parent)
        # 'background' is a placeholder label, never a real detection — drop it from the table
        self.class_names = [c for c in class_names if c != "background"]
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        self.setFixedWidth(280)

        self.fps_label = QLabel("FPS: --")
        self.frame_label = QLabel("Frame: --")
        self.total_label = QLabel("Total Detections: --")
        for lbl in (self.fps_label, self.frame_label, self.total_label):
            lbl.setStyleSheet("font-weight: bold;")
            layout.addWidget(lbl)

        self.table = QTableWidget(len(self.class_names), 3)
        self.table.setHorizontalHeaderLabels(["Class", "Count", "Avg Conf"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        for row, name in enumerate(self.class_names):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem("0"))
            self.table.setItem(row, 2, QTableWidgetItem("0.00"))

        layout.addWidget(self.table)

    def update_stats(self, stats):
        self.fps_label.setText(f"FPS: {stats.fps:.1f}")
        self.frame_label.setText(f"Frame: {stats.frame_idx}")
        self.total_label.setText(f"Total Detections: {stats.total_detections}")

        for row, name in enumerate(self.class_names):
            count = stats.counts.get(name, 0)
            conf = stats.avg_confidence.get(name, 0.0)
            self.table.item(row, 1).setText(str(count))
            self.table.item(row, 2).setText(f"{conf:.2f}")

    def reset(self):
        self.fps_label.setText("FPS: --")
        self.frame_label.setText("Frame: --")
        self.total_label.setText("Total Detections: --")
        for row in range(self.table.rowCount()):
            self.table.item(row, 1).setText("0")
            self.table.item(row, 2).setText("0.00")
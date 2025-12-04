# gui/main_window.py

from __future__ import annotations
from pathlib import Path

from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QComboBox,
    QLineEdit,
    QStatusBar,
    QProgressBar,
    QMessageBox,
    QSplitter,
    QListWidget,
)

from .worker import CalibrationWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CCD Calibration Pipeline")
        self.resize(1200, 700)

        self.config_path: Path | None = None
        self.source_path: Path | None = None
        self.source_type: str = "directory"

        self._init_ui()

        self.thread: QThread | None = None
        self.worker: CalibrationWorker | None = None

    def _init_ui(self):
        # --- Top controls: configs + data source ---
        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(8)

        self.config_label = QLabel("Config: (none)")
        btn_load_config = QPushButton("Load configs.ini")
        btn_load_config.clicked.connect(self.on_load_config)

        self.source_type_combo = QComboBox()
        self.source_type_combo.addItems(["Directory", "List file", "Archive"])
        self.source_type_combo.currentIndexChanged.connect(self.on_source_type_changed)

        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("Path to directory / list / archive...")
        self.source_edit.setReadOnly(True)

        btn_browse_source = QPushButton("Browse...")
        btn_browse_source.clicked.connect(self.on_browse_source)

        btn_run = QPushButton("Run pipeline")
        btn_run.clicked.connect(self.on_run_pipeline)

        top_layout.addWidget(self.config_label, stretch=2)
        top_layout.addWidget(btn_load_config)
        top_layout.addSpacing(16)
        top_layout.addWidget(self.source_type_combo)
        top_layout.addWidget(self.source_edit, stretch=2)
        top_layout.addWidget(btn_browse_source)
        top_layout.addWidget(btn_run)

        # --- Center: left = placeholder for file list, right = logs ---
        center_widget = QWidget()
        center_layout = QHBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)

        # Left: file list / future preview; for now just list widget
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)

        left_layout.addWidget(QLabel("Files (placeholder for future file tree):"))
        self.files_list = QListWidget()
        left_layout.addWidget(self.files_list)

        # Right: logs
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)

        right_layout.addWidget(QLabel("Log:"))
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QTextEdit.NoWrap)

        right_layout.addWidget(self.log_view)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        center_layout.addWidget(splitter)

        # --- Status bar ---
        status = QStatusBar()
        self.setStatusBar(status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_bar.setVisible(False)
        status.addPermanentWidget(self.progress_bar)

        # --- Main layout ---
        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        main_layout.addWidget(top_bar)
        main_layout.addWidget(center_widget)

        self.setCentralWidget(main)

    # -------------------------------------------------
    # Slots
    # -------------------------------------------------
    def on_load_config(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select configs.ini", "", "INI files (*.ini);;All files (*)"
        )
        if not file_path:
            return
        self.config_path = Path(file_path)
        self.config_label.setText(f"Config: {self.config_path.name}")

    def on_source_type_changed(self, idx: int):
        text = self.source_type_combo.currentText()
        if text == "Directory":
            self.source_type = "directory"
        elif text == "List file":
            self.source_type = "list"
        else:
            self.source_type = "archive"

    def on_browse_source(self):
        if self.source_type == "directory":
            dir_path = QFileDialog.getExistingDirectory(self, "Select directory with FITS")
            if not dir_path:
                return
            self.source_path = Path(dir_path)
        elif self.source_type == "list":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select list file", "", "List files (*.lst *.txt);;All files (*)"
            )
            if not file_path:
                return
            self.source_path = Path(file_path)
        else:  # archive
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select archive", "", "Archives (*.zip *.tar *.tar.gz);;All files (*)"
            )
            if not file_path:
                return
            self.source_path = Path(file_path)

        self.source_edit.setText(str(self.source_path))
        self.populate_file_list_placeholder()

    def populate_file_list_placeholder(self):
        """For now we just show a single entry; later we can parse FITS list."""
        self.files_list.clear()
        if self.source_path:
            self.files_list.addItem(str(self.source_path))

    def on_run_pipeline(self):
        if not self.config_path:
            QMessageBox.warning(self, "Missing configs", "Please load a configs.ini first.")
            return
        if not self.source_path:
            QMessageBox.warning(self, "Missing data", "Please choose data source (directory/list/archive).")
            return

        # Start background worker
        self.progress_bar.setVisible(True)
        self.log_view.clear()
        self.statusBar().showMessage("Running pipeline...")

        self.thread = QThread(self)
        self.worker = CalibrationWorker(
            str(self.config_path),
            str(self.source_path),
            self.source_type,
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.append_log)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def append_log(self, text: str):
        self.log_view.append(text)

    def on_worker_finished(self, ok: bool):
        self.progress_bar.setVisible(False)
        if ok:
            self.statusBar().showMessage("Pipeline finished successfully.", 5000)
        else:
            self.statusBar().showMessage("Pipeline finished with errors.", 5000)
            QMessageBox.warning(self, "Error", "Pipeline finished with errors. See log.")



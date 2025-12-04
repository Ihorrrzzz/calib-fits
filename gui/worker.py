# gui/worker.py

from PySide6.QtCore import QObject, QThread, Signal
from calib_core import CalibrationPipeline


class CalibrationWorker(QObject):
    progress = Signal(str)   # log messages
    finished = Signal(bool)  # success flag

    def __init__(self, config_path: str, source: str, source_type: str):
        super().__init__()
        self.config_path = config_path
        self.source = source
        self.source_type = source_type

    def run(self):
        try:
            pipeline = CalibrationPipeline(self.config_path)
            # Wrap pipeline.run_full_calibration and forward prints to progress signal
            self.progress.emit("Starting calibration...")
            pipeline.run_full_calibration(
                self.source,
                source_type=self.source_type,
                verbose=True,
            )
            self.progress.emit("Calibration finished.")
            self.finished.emit(True)
        except Exception as e:
            self.progress.emit(f"Error: {e}")
            self.finished.emit(False)

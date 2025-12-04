# gui/worker.py

from __future__ import annotations

import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    """
    Signals available from a running worker thread.

    finished: job is done (success or failure).
    error:    string with traceback if an exception occurred.
    result:   any object returned by the function.
    log:      log / status messages as strings.
    """

    finished = Signal()
    error = Signal(str)
    result = Signal(object)
    log = Signal(str)


class Worker(QRunnable):
    """
    Generic worker to run a function in another thread.

    Example:
        def task():
            return expensive_thing()

        worker = Worker(task)
        worker.signals.result.connect(handle_result)
        QThreadPool.globalInstance().start(worker)
    """

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        """Run the function with its arguments."""
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            tb = traceback.format_exc()
            self.signals.error.emit(tb)
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()
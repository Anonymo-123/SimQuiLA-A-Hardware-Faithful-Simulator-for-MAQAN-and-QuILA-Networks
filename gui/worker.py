"""Background worker that runs the DPS-QKD backend off the GUI thread."""

from __future__ import annotations

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from backend.adapter import SimulationParameters, SimulationResult, run_simulation


class SimulationWorker(QObject):
    """Runs :func:`backend.adapter.run_simulation` on a worker thread.

    Signals
    -------
    finished(SimulationResult)
        Emitted once the (fast, event-driven) backend run completes.
    failed(str)
        Emitted if the backend raises an exception.
    """

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, params: SimulationParameters) -> None:
        super().__init__()
        self._params = params

    def run(self) -> None:
        try:
            result: SimulationResult = run_simulation(self._params)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001 - surface any backend error
            self.failed.emit(str(exc))


def start_simulation(
    params: SimulationParameters,
    on_finished,
    on_failed,
) -> tuple[QThread, SimulationWorker]:
    """Create and start a worker thread, wiring up callbacks.

    Returns the ``(thread, worker)`` pair; the caller must keep references
    to both alive until ``finished``/``failed`` fires or Qt will garbage
    collect them mid-run.
    """

    thread = QThread()
    worker = SimulationWorker(params)
    worker.moveToThread(thread)

    thread.started.connect(worker.run)
    worker.finished.connect(on_finished)
    worker.failed.connect(on_failed)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    thread.start()
    return thread, worker

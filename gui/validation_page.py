"""GUI page: run the MultiVeStA-style statistical validation layer and show
mean / std / 95% CI results in the app itself.

Uses the same pattern as gui/worker.py (QThread + signals) so the GUI never
blocks while running dozens of independent-seed trials. Talks only to
backend.mvesta_adapter / backend.statistical_validation -- never touches
backend.adapter, sequence/, or any other GUI page's internals.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from backend.adapter import SimulationParameters, total_channel_loss_db
from backend.mvesta_adapter import DPSQKDSimulationModel
from backend.statistical_validation import BatchStats, run_batch

_OBSERVABLE_LABELS = [
    ("QBER", "QBER"),
    ("SIFTED_KEY_LEN", "Sifted key length"),
    ("THROUGHPUT_BPS", "Sifted key rate (bps)"),
    ("DETECTOR_CLICKS", "Detector clicks"),
    ("EARLY_LATE_DISCARDS", "Early/late discards"),
    ("OUTSIDE_WINDOW_CLICKS", "Clicks outside port windows"),
    ("LEAK_CLICKS", "IM-leakage clicks"),
    ("EVE_INTERCEPTIONS", "Bits from Eve"),
    ("PHOTON_LOSS_RATE", "Pulses without a click (fraction)"),
]


class _ValidationWorker(QObject):
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, params: SimulationParameters, base_seed: int,
                 n_seeds: int, observables: list[str]):
        super().__init__()
        self._params = params
        self._base_seed = base_seed
        self._n_seeds = n_seeds
        self._observables = observables

    def run(self) -> None:
        try:
            model = DPSQKDSimulationModel()
            settings = {f.name: getattr(self._params, f.name) for f in fields(self._params)
                        if not f.name.endswith("_seed")}
            settings["max_trace_events"] = 0      # statistics only
            model.configure(**settings)
            seeds = range(self._base_seed, self._base_seed + self._n_seeds)
            results = run_batch(model, seeds, self._observables)
            self.finished.emit(results)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ValidationPage(QWidget):
    """MultiVeStA-style statistical validation panel.

    Runs N independent-seed trials of the current dashboard parameters
    through backend.mvesta_adapter.DPSQKDSimulationModel (reset/run_step/
    rval), the same interface a real MultiVeStA Java engine would drive
    over py4j, and reports mean / std / 95% CI per observable.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._parameters_provider = None
        self._thread: Optional[QThread] = None
        self._worker: Optional[_ValidationWorker] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(12)

        title = QLabel("MultiVeStA Statistical Validation")
        title.setObjectName("PageTitle")
        outer.addWidget(title)

        subtitle = QLabel(
            "Runs the current Dashboard parameters through independent, "
            "seeded trials (reset(seed) / run_step() / rval(observable)) "
            "and reports mean, std, and 95% confidence intervals -- the "
            "same statistic a MultiVeStA E[...] query computes."
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("Subtitle")
        outer.addWidget(subtitle)

        controls = QFrame()
        controls.setObjectName("Panel")
        cl = QHBoxLayout(controls)
        cl.setContentsMargins(14, 12, 14, 12)
        cl.setSpacing(16)

        cl.addWidget(QLabel("Independent seeds:"))
        self.n_seeds_spin = QSpinBox()
        self.n_seeds_spin.setRange(5, 2000)
        self.n_seeds_spin.setValue(50)
        cl.addWidget(self.n_seeds_spin)

        cl.addWidget(QLabel("Base seed:"))
        self.base_seed_spin = QSpinBox()
        self.base_seed_spin.setRange(0, 1_000_000)
        self.base_seed_spin.setValue(1)
        cl.addWidget(self.base_seed_spin)

        cl.addStretch(1)

        self.run_button = QPushButton("Run Validation")
        self.run_button.setObjectName("PrimaryButton")
        self.run_button.clicked.connect(self._on_run_clicked)
        cl.addWidget(self.run_button)

        outer.addWidget(controls)

        obs_frame = QFrame()
        obs_frame.setObjectName("Panel")
        obs_layout = QGridLayout(obs_frame)
        obs_layout.setContentsMargins(14, 10, 14, 10)
        self._obs_checkboxes: dict[str, QCheckBox] = {}
        for i, (key, label) in enumerate(_OBSERVABLE_LABELS):
            box = QCheckBox(label)
            box.setChecked(key in ("QBER", "SIFTED_KEY_LEN"))
            obs_layout.addWidget(box, i // 3, i % 3)
            self._obs_checkboxes[key] = box
        outer.addWidget(obs_frame)

        self.results_table = QTableWidget(0, 6)
        self.results_table.setHorizontalHeaderLabels(
            ["Observable", "n", "Mean", "Std", "95% CI low", "95% CI high"]
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.results_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        outer.addWidget(self.results_table, 1)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        self.log.setObjectName("LogView")
        outer.addWidget(self.log)

        self.status_label = QLabel("Ready.")
        self.status_label.setObjectName("Subtitle")
        outer.addWidget(self.status_label)

    def set_parameters_provider(self, provider) -> None:
        self._parameters_provider = provider

    # ------------------------------------------------------------------ #

    def _on_run_clicked(self) -> None:
        if self._parameters_provider is None:
            return
        params = self._parameters_provider()
        observables = [k for k, box in self._obs_checkboxes.items() if box.isChecked()]
        if not observables:
            self.status_label.setText("Select at least one observable.")
            return

        n_seeds = self.n_seeds_spin.value()
        base_seed = self.base_seed_spin.value()

        self.run_button.setEnabled(False)
        self.status_label.setText(
            f"Running {n_seeds} independent-seed trials "
            f"(seeds {base_seed}..{base_seed + n_seeds - 1})..."
        )
        self.log.append(
            f"Starting validation batch: {n_seeds} seeds, {params.num_pulses:,} pulses each, "
            f"engine={params.engine}, channel loss={total_channel_loss_db(params):.2f} dB, "
            f"visibility={params.interferometer_visibility}, eve_enabled={params.eve_enabled}"
        )

        self._thread = QThread()
        self._worker = _ValidationWorker(params, base_seed, n_seeds, observables)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

    def _on_finished(self, results: dict) -> None:
        self.results_table.setRowCount(0)
        for name, stats in results.items():
            assert isinstance(stats, BatchStats)
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            values = [
                name,
                str(stats.n),
                f"{stats.mean:.5f}",
                f"{stats.std:.5f}",
                f"{stats.ci95_low:.5f}",
                f"{stats.ci95_high:.5f}",
            ]
            for col, value in enumerate(values):
                self.results_table.setItem(row, col, QTableWidgetItem(value))
        self.log.append("Validation batch complete.")
        self.status_label.setText("Done.")
        self.run_button.setEnabled(True)

    def _on_failed(self, message: str) -> None:
        self.log.append(f"ERROR: {message}")
        self.status_label.setText("Validation failed -- see log.")
        self.run_button.setEnabled(True)

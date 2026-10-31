"""GUI page: Cascade classical error-correction, run downstream of the
DPS-QKD backend on the actual sifted Alice/Bob keys.

Follows the same pattern as gui/validation_page.py (QThread + signals so
the GUI never blocks) and, like that page, talks only to its own backend
module (backend.cascade) -- never to backend.adapter, sequence/, or any
other GUI page's internals. Nothing here changes any existing simulation
parameter, formula, or statistic; Cascade only ever reads
SimulationResult.alice_key_bits / .bob_key_bits / .sifted_key_bits after a
run has already completed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from backend.cascade import CascadeConfig, CascadeResult, run_cascade
from gui.widgets import StatCard

if TYPE_CHECKING:
    # Only needed for the type hint below; avoids making this page's import
    # depend on the full sequence/ stack (backend.adapter's own import
    # chain), keeping Cascade a genuinely independent downstream layer.
    from backend.adapter import SimulationResult


def _parse_bit_string(text: str) -> List[int]:
    """Parse a user-typed bit string into a list of 0/1 ints.

    Accepts digits with arbitrary separators (spaces, commas) so both
    "0101 1100" and "0,1,0,1,1,1,0,0" and "01011100" work. Raises
    ValueError with a human-readable message on anything else.
    """

    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Key field is empty.")

    # Strip common separators, keep only characters that could plausibly
    # be bits so we can give a precise error on the first bad character.
    stripped = cleaned.replace(",", " ").replace("\t", " ")
    tokens = stripped.split()
    bits: List[int] = []
    if len(tokens) > 1 or (len(tokens) == 1 and len(tokens[0]) <= 1):
        # space/comma separated tokens, each expected to be a single 0/1
        for tok in tokens:
            if tok not in ("0", "1"):
                raise ValueError(f"Invalid bit token '{tok}' (expected 0 or 1).")
            bits.append(int(tok))
    else:
        # one contiguous run of digits, e.g. "0101100"
        run = tokens[0]
        for ch in run:
            if ch not in ("0", "1"):
                raise ValueError(
                    f"Invalid character '{ch}' in key (expected only 0/1)."
                )
            bits.append(int(ch))

    if not bits:
        raise ValueError("No bits parsed from input.")
    return bits


class _CascadeWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, alice_bits: List[int], bob_bits: List[int], config: CascadeConfig):
        super().__init__()
        self._alice_bits = alice_bits
        self._bob_bits = bob_bits
        self._config = config

    def run(self) -> None:
        try:
            result: CascadeResult = run_cascade(
                self._alice_bits, self._bob_bits, self._config
            )
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001 - surface any error to the GUI
            self.failed.emit(str(exc))


class CascadePage(QWidget):
    """Downstream classical error-correction panel.

    Two key sources are supported:

    * "Last Simulation Result" -- uses the real, actual sifted
      ``alice_key_bits`` / ``bob_key_bits`` produced by the most recent
      DPS-QKD backend run (via a result-provider callback set by
      MainWindow, mirroring the ``set_parameters_provider`` pattern
      already used by ValidationPage/AnalyticsPage).
    * "Custom Test Keys" -- lets you type in controlled Alice/Bob bit
      strings directly, for testing Cascade's edge-case handling
      independently of the QKD simulation (identical keys, a single
      flipped bit, adversarial patterns, tiny keys, etc).
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._result_provider: Optional[Callable[[], Optional["SimulationResult"]]] = None
        self._thread: Optional[QThread] = None
        self._worker: Optional[_CascadeWorker] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(12)

        title = QLabel("Cascade Error Correction")
        title.setObjectName("PageTitle")
        outer.addWidget(title)

        subtitle = QLabel(
            "Independent downstream classical layer: takes Alice's and "
            "Bob's raw sifted keys (from the last DPS-QKD run, or a "
            "manually entered controlled test key) and runs the real "
            "Cascade protocol -- QBER sampling, block-size calculation, "
            "multi-pass parity checks with random shuffling, binary-"
            "search bit correction, and cascading back-propagation."
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("Subtitle")
        outer.addWidget(subtitle)

        # -- Key source controls ---------------------------------------- #
        source_frame = QFrame()
        source_frame.setObjectName("Panel")
        sl = QHBoxLayout(source_frame)
        sl.setContentsMargins(14, 12, 14, 12)
        sl.setSpacing(16)

        sl.addWidget(QLabel("Key source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Last Simulation Result", "Custom Test Keys"])
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        sl.addWidget(self.source_combo)
        sl.addStretch(1)
        outer.addWidget(source_frame)

        self.source_stack = QStackedWidget()

        # Page 0: last-simulation-result info line
        sim_info_frame = QFrame()
        sim_info_layout = QVBoxLayout(sim_info_frame)
        sim_info_layout.setContentsMargins(0, 0, 0, 0)
        self.sim_key_label = QLabel(
            "No simulation result yet -- run a simulation on the Dashboard first."
        )
        self.sim_key_label.setObjectName("Subtitle")
        self.sim_key_label.setWordWrap(True)
        sim_info_layout.addWidget(self.sim_key_label)
        self.source_stack.addWidget(sim_info_frame)

        # Page 1: custom key entry
        custom_frame = QFrame()
        custom_layout = QGridLayout(custom_frame)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        custom_layout.setSpacing(8)
        custom_layout.addWidget(QLabel("Alice's bits:"), 0, 0)
        self.alice_bits_edit = QLineEdit()
        self.alice_bits_edit.setPlaceholderText(
            "e.g. 0110101100 or 0,1,1,0,1,0,1,1,0,0"
        )
        custom_layout.addWidget(self.alice_bits_edit, 0, 1)
        custom_layout.addWidget(QLabel("Bob's bits:"), 1, 0)
        self.bob_bits_edit = QLineEdit()
        self.bob_bits_edit.setPlaceholderText(
            "e.g. 0110001100 (must be same length as Alice's)"
        )
        custom_layout.addWidget(self.bob_bits_edit, 1, 1)
        self.source_stack.addWidget(custom_frame)

        outer.addWidget(self.source_stack)

        # -- Cascade parameter controls ----------------------------------- #
        params_frame = QFrame()
        params_frame.setObjectName("Panel")
        pl = QHBoxLayout(params_frame)
        pl.setContentsMargins(14, 12, 14, 12)
        pl.setSpacing(16)

        pl.addWidget(QLabel("Passes (w):"))
        self.passes_spin = QSpinBox()
        self.passes_spin.setRange(1, 10)
        self.passes_spin.setValue(4)
        pl.addWidget(self.passes_spin)

        pl.addWidget(QLabel("QBER sample fraction:"))
        self.sample_fraction_spin = QDoubleSpinBox()
        self.sample_fraction_spin.setRange(0.05, 0.90)
        self.sample_fraction_spin.setSingleStep(0.05)
        self.sample_fraction_spin.setDecimals(2)
        self.sample_fraction_spin.setValue(0.20)
        self.sample_fraction_spin.setSuffix(" (fraction publicly sacrificed)")
        pl.addWidget(self.sample_fraction_spin)

        pl.addWidget(QLabel("Shuffle seed:"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 1_000_000)
        self.seed_spin.setValue(1)
        pl.addWidget(self.seed_spin)

        pl.addStretch(1)

        self.run_button = QPushButton("Run Cascade")
        self.run_button.setObjectName("PrimaryButton")
        self.run_button.clicked.connect(self._on_run_clicked)
        pl.addWidget(self.run_button)

        outer.addWidget(params_frame)

        # -- Summary stat cards -------------------------------------------- #
        stats_frame = QFrame()
        stats_grid = QGridLayout(stats_frame)
        stats_grid.setContentsMargins(0, 0, 0, 0)
        stats_grid.setSpacing(10)

        self.stat_input_len = StatCard("Sifted Key Used")
        self.stat_ref_qber = StatCard("Reference QBER (ground truth)")
        self.stat_sample = StatCard("QBER Sample")
        self.stat_est_qber = StatCard("Estimated QBER")
        self.stat_k1 = StatCard("Block Size k1")
        self.stat_passes_run = StatCard("Passes Run")
        self.stat_comparisons = StatCard("Total Comparisons")
        self.stat_disclosed = StatCard("Disclosed Bits")
        self.stat_secure_est = StatCard("Est. Secure Key Length")
        self.stat_final_mismatch = StatCard("Final Mismatches")
        self.stat_agreement = StatCard("Final Agreement")

        cards = [
            self.stat_input_len, self.stat_ref_qber, self.stat_sample,
            self.stat_est_qber, self.stat_k1, self.stat_passes_run,
            self.stat_comparisons, self.stat_disclosed, self.stat_secure_est,
            self.stat_final_mismatch, self.stat_agreement,
        ]
        for i, card in enumerate(cards):
            stats_grid.addWidget(card, i // 4, i % 4)
        outer.addWidget(stats_frame)

        # -- Pass-wise table -------------------------------------------- #
        outer.addWidget(QLabel("Pass-wise breakdown:"))
        self.pass_table = QTableWidget(0, 6)
        self.pass_table.setHorizontalHeaderLabels(
            ["Pass", "Block Size", "Blocks", "Comparisons", "Corrections", "Disclosed Bits"]
        )
        self.pass_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.pass_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.pass_table.setMaximumHeight(160)
        outer.addWidget(self.pass_table)

        # -- Correction / warning log -------------------------------------- #
        outer.addWidget(QLabel("Correction log:"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setObjectName("LogView")
        self.log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        outer.addWidget(self.log, 1)

        self.status_label = QLabel("Ready.")
        self.status_label.setObjectName("Subtitle")
        outer.addWidget(self.status_label)

        self._on_source_changed(0)

    # ------------------------------------------------------------------ #
    # Wiring from MainWindow
    # ------------------------------------------------------------------ #

    def set_result_provider(self, provider: Callable[[], Optional[SimulationResult]]) -> None:
        """MainWindow calls this once at startup, mirroring
        ``set_parameters_provider`` on the other pages. ``provider()``
        returns the most recent ``SimulationResult`` (or ``None``)."""

        self._result_provider = provider

    def notify_new_result(self, result: SimulationResult) -> None:
        """MainWindow calls this after every completed simulation so the
        info line stays current without this page having to poll."""

        if result.sifted_key_bits > 0:
            self.sim_key_label.setText(
                f"Latest run: {result.sifted_key_bits} sifted key bits "
                f"available (backend QBER {result.qber * 100:.2f}%). "
                f"Click 'Run Cascade' to error-correct this key."
            )
        else:
            self.sim_key_label.setText(
                "Latest run produced no sifted key bits -- Cascade needs "
                "at least a couple of bits to run. Try different "
                "simulation parameters (e.g. lower fibre loss / more "
                "pulses) or switch to Custom Test Keys."
            )

    # ------------------------------------------------------------------ #
    # UI events
    # ------------------------------------------------------------------ #

    def _on_source_changed(self, index: int) -> None:
        self.source_stack.setCurrentIndex(index)

    def _on_run_clicked(self) -> None:
        try:
            alice_bits, bob_bits = self._collect_keys()
        except ValueError as exc:
            self.status_label.setText(f"Input error: {exc}")
            self.log.append(f"ERROR: {exc}")
            return

        config = CascadeConfig(
            num_passes=self.passes_spin.value(),
            qber_sample_fraction=self.sample_fraction_spin.value(),
            shuffle_base_seed=self.seed_spin.value(),
        )

        self.run_button.setEnabled(False)
        self.status_label.setText("Running Cascade\u2026")
        self.log.append(
            f"Starting Cascade: {len(alice_bits)} input bits, "
            f"w={config.num_passes}, sample_fraction="
            f"{config.qber_sample_fraction:.2f}, seed={config.shuffle_base_seed}"
        )

        self._thread = QThread()
        self._worker = _CascadeWorker(alice_bits, bob_bits, config)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

    def _collect_keys(self) -> tuple[List[int], List[int]]:
        if self.source_combo.currentIndex() == 0:
            if self._result_provider is None:
                raise ValueError("No result provider configured.")
            result = self._result_provider()
            if result is None:
                raise ValueError("Run a simulation first (Dashboard \u2192 Start).")
            n = result.sifted_key_bits
            if n <= 0:
                raise ValueError(
                    "The last simulation produced no sifted key bits."
                )
            return list(result.alice_key_bits[:n]), list(result.bob_key_bits[:n])
        else:
            alice_bits = _parse_bit_string(self.alice_bits_edit.text())
            bob_bits = _parse_bit_string(self.bob_bits_edit.text())
            return alice_bits, bob_bits

    # ------------------------------------------------------------------ #
    # Worker callbacks
    # ------------------------------------------------------------------ #

    def _on_finished(self, result: CascadeResult) -> None:
        self.run_button.setEnabled(True)

        if not result.success:
            self.status_label.setText(f"Cascade failed: {result.error}")
            self.log.append(f"ERROR: {result.error}")
            return

        for w in result.warnings:
            self.log.append(f"WARNING: {w}")

        self.stat_input_len.set_value(str(result.input_key_length))
        self.stat_ref_qber.set_value(f"{result.reference_initial_qber * 100:.2f}%")

        if result.qber_estimation_skipped:
            self.stat_sample.set_value("skipped")
            self.stat_est_qber.set_value("n/a (assumed)")
        else:
            self.stat_sample.set_value(
                f"{result.sample_mismatches}/{result.sample_size}"
            )
            self.stat_est_qber.set_value(f"{result.estimated_qber * 100:.2f}%")

        self.stat_k1.set_value(str(result.k1_block_size))
        self.stat_passes_run.set_value(str(len(result.passes)))
        self.stat_comparisons.set_value(str(result.total_comparisons))
        self.stat_disclosed.set_value(str(result.total_disclosed_bits))
        self.stat_secure_est.set_value(str(result.estimated_secure_key_length))

        mismatch_color = "#22c55e" if result.final_mismatches == 0 else "#ef4444"
        self.stat_final_mismatch.set_value(str(result.final_mismatches), mismatch_color)
        self.stat_agreement.set_value(
            "\u2713 MATCH" if result.final_agreement else "\u2717 MISMATCH",
            "#22c55e" if result.final_agreement else "#ef4444",
        )

        self.pass_table.setRowCount(0)
        for p in result.passes:
            row = self.pass_table.rowCount()
            self.pass_table.insertRow(row)
            values = [
                str(p.pass_id), str(p.block_size), str(p.num_blocks),
                str(p.comparisons), str(p.corrections), str(p.disclosed_bits),
            ]
            for col, val in enumerate(values):
                self.pass_table.setItem(row, col, QTableWidgetItem(val))

        self.log.append(
            f"QBER sample: {result.sample_mismatches}/{result.sample_size} "
            f"mismatches -> estimated QBER "
            f"{'n/a' if result.estimated_qber is None else f'{result.estimated_qber:.4f}'}"
        )
        self.log.append(f"Block size k1 = {result.k1_block_size}")
        for corr in result.corrections:
            origin = "back-propagation" if corr.via_backpropagation else "direct scan"
            self.log.append(
                f"  Pass {corr.found_in_pass}: flipped bit at working index "
                f"{corr.working_index} ({origin})"
            )
        if result.residual_error_note:
            self.log.append(f"NOTE: {result.residual_error_note}")

        self.log.append(
            f"Done. {result.total_corrections} correction(s), "
            f"{result.total_disclosed_bits} bit(s) of public information "
            f"disclosed, final agreement: "
            f"{'YES' if result.final_agreement else 'NO'}."
        )

        self.status_label.setText(
            "Done \u2014 "
            + ("keys fully agree." if result.final_agreement
               else f"{result.final_mismatches} bit(s) still disagree (see note above).")
        )

    def _on_failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.log.append(f"ERROR: {message}")
        self.status_label.setText("Cascade run failed \u2014 see log.")

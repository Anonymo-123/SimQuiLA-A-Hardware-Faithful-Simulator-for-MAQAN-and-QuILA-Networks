"""Analytics page -- post-run charts and four parameter-sweep experiments.

Everything plotted comes from :mod:`backend.adapter` results; nothing is
synthesised. The last run is shown as a click classification and as Bob's
time-stamp histogram -- for the MAQAN time-bin hardware this shows the
interferometer's 1/6 : 1/3 : 1/3 : 1/6 slot structure in the port-0 and
port-1 time windows. Each sweep re-runs the real backend at a handful of
parameter values in a worker thread and plots the simulated points together
with the closed-form expectation of the same model (dashed line):

* Distance vs. sifted key rate      -- loss from distance x dB/km.
* Mean photon number vs. sifted key rate -- detector hold-off saturates it.
* Interferometer visibility vs. QBER.
* Temporal filter (guard band) vs. QBER -- trades key rate for QBER, as in
  Shaw et al. (2022), Fig. 11.
"""

from __future__ import annotations

import html
from dataclasses import replace
from typing import Callable, List, Optional

import pyqtgraph as pg
from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from backend.adapter import SimulationParameters, SimulationResult, run_simulation

pg.setConfigOption("background", "#171c24")
pg.setConfigOption("foreground", "#8a93a3")

#: Pulses per sweep point (each point is a full simulation run).
SWEEP_PULSES = 3_000_000


# --------------------------------------------------------------------------- #
# Generic parameter-sweep worker
# --------------------------------------------------------------------------- #

class _SweepWorker(QObject):
    """Runs the real backend once per value of a single swept parameter.

    ``param_name`` must be a field of :class:`SimulationParameters`;
    ``to_param`` converts the plotted x value into that field's units.
    ``y_extractor`` returns (simulated, expected) from a result.
    """

    progress = pyqtSignal(float, float, float)  # x, simulated y, expected y
    finished = pyqtSignal()

    def __init__(
        self,
        base_params: SimulationParameters,
        param_name: str,
        values: List[float],
        y_extractor: Callable[[SimulationResult], tuple],
        to_param: Callable[[float], float],
        num_pulses_override: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._base = base_params
        self._param_name = param_name
        self._values = values
        self._y_extractor = y_extractor
        self._to_param = to_param
        self._num_pulses_override = num_pulses_override

    def run(self) -> None:
        for v in self._values:
            kwargs = {self._param_name: self._to_param(v), "max_trace_events": 0}
            if self._param_name in ("distance_km", "fiber_loss_db_km"):
                # A distance or dB/km sweep only makes sense with loss computed
                # from distance x dB/km, not a fixed measured loss.
                kwargs["channel_loss_db"] = None
            if self._num_pulses_override is not None:
                kwargs["num_pulses"] = self._num_pulses_override
            p = replace(self._base, **kwargs)
            if p.engine == "event":
                p = replace(p, engine="timebin")    # sweeps need many points: fast engine
            result = run_simulation(p)
            y_sim, y_exp = self._y_extractor(result)
            self.progress.emit(v, y_sim, y_exp)
        self.finished.emit()


class _SweepChart(QGroupBox):
    """One self-contained sweep experiment: button + plot + its own thread."""

    def __init__(
        self,
        title: str,
        button_label: str,
        param_name: str,
        values: List[float],
        y_extractor: Callable[[SimulationResult], tuple],
        x_label: str,
        y_label: str,
        color: str,
        to_param: Callable[[float], float] = lambda x: x,
        num_pulses_override: Optional[int] = SWEEP_PULSES,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(title, parent)
        self._param_name = param_name
        self._values = values
        self._y_extractor = y_extractor
        self._to_param = to_param
        self._color = color
        self._num_pulses_override = num_pulses_override
        self._get_params_callback: Optional[Callable[[], SimulationParameters]] = None
        self._original_label = button_label

        self._thread: Optional[QThread] = None
        self._worker: Optional[_SweepWorker] = None
        self._x: List[float] = []
        self._y: List[float] = []
        self._y_exp: List[float] = []

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 14, 10, 10)
        layout.setSpacing(8)
        controls = QHBoxLayout()
        self.run_button = QPushButton(button_label)
        self.run_button.clicked.connect(self._run)
        controls.addWidget(self.run_button)
        controls.addStretch(1)
        legend = QLabel("● simulated   - - closed form")
        legend.setProperty("class", "dim")
        controls.addWidget(legend)
        layout.addLayout(controls)

        self.plot = pg.PlotWidget()
        self.plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plot.setMinimumHeight(220)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setLabel("bottom", x_label)
        self.plot.setLabel("left", y_label)
        layout.addWidget(self.plot, 1)

    def set_parameters_provider(self, callback: Callable[[], SimulationParameters]) -> None:
        self._get_params_callback = callback

    def _run(self) -> None:
        if self._get_params_callback is None:
            return
        base_params = self._get_params_callback()

        self._x, self._y, self._y_exp = [], [], []
        self.plot.clear()
        self.run_button.setEnabled(False)
        self.run_button.setText("Running…")

        self._thread = QThread()
        self._worker = _SweepWorker(
            base_params, self._param_name, self._values, self._y_extractor,
            self._to_param, self._num_pulses_override,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_point)
        self._worker.finished.connect(self._on_done)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()

    def _on_point(self, x: float, y: float, y_exp: float) -> None:
        self._x.append(x)
        self._y.append(y)
        self._y_exp.append(y_exp)
        self.plot.clear()
        self.plot.plot(self._x, self._y_exp,
                       pen=pg.mkPen(self._color, width=1.5, style=Qt.PenStyle.DashLine))
        self.plot.plot(self._x, self._y, pen=None, symbol="o",
                       symbolBrush=self._color, symbolSize=7)

    def _on_done(self) -> None:
        self.run_button.setEnabled(True)
        self.run_button.setText(self._original_label)


# --------------------------------------------------------------------------- #
# Extractors: (simulated, expected) from a SimulationResult
# --------------------------------------------------------------------------- #

def _rate_kbps(result: SimulationResult) -> tuple:
    return result.throughput_bps / 1e3, result.expected_sifted_rate_bps / 1e3


def _qber_percent(result: SimulationResult) -> tuple:
    return result.qber * 100.0, result.expected_qber * 100.0


# --------------------------------------------------------------------------- #
# Analytics page
# --------------------------------------------------------------------------- #

class AnalyticsPage(QWidget):
    """Charts summarising the most recent run plus four sweep experiments."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._get_params_callback = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        top_row = QHBoxLayout()
        top_row.setSpacing(14)

        hist_box = QGroupBox("Bob's Time Stamps Within a Pulse Period (Last Run)")
        hist_box.setMinimumHeight(250)
        hist_layout = QVBoxLayout(hist_box)
        self.hist_plot = pg.PlotWidget()
        self.hist_plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.hist_plot.showGrid(x=True, y=True, alpha=0.15)
        self.hist_plot.setLabel("bottom", "Time after bin 0 of the pulse reaches Bob (ns)")
        self.hist_plot.setLabel("left", "Clicks per 50 ps")
        hist_layout.addWidget(self.hist_plot)
        self.hist_note = QLabel("")
        self.hist_note.setWordWrap(True)
        self.hist_note.setProperty("class", "dim")
        hist_layout.addWidget(self.hist_note)
        top_row.addWidget(hist_box, 3)

        clicks_box = QGroupBox("Click Classification (Last Run)")
        clicks_box.setMinimumHeight(250)
        clicks_layout = QVBoxLayout(clicks_box)
        self.clicks_plot = pg.PlotWidget()
        self.clicks_plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.clicks_plot.showGrid(x=False, y=True, alpha=0.15)
        clicks_layout.addWidget(self.clicks_plot)
        top_row.addWidget(clicks_box, 2)

        summary_box = QGroupBox("Run Summary")
        summary_box.setMinimumHeight(250)
        summary_layout = QVBoxLayout(summary_box)
        self.summary_label = QLabel("Run a simulation on the Dashboard to see results here.")
        self.summary_label.setWordWrap(True)
        self.summary_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        summary_layout.addWidget(self.summary_label)
        top_row.addWidget(summary_box, 2)

        layout.addLayout(top_row, 0)

        sweeps_grid = QGridLayout()
        sweeps_grid.setSpacing(14)
        sweeps_grid.setRowStretch(0, 1)
        sweeps_grid.setRowStretch(1, 1)
        sweeps_grid.setColumnStretch(0, 1)
        sweeps_grid.setColumnStretch(1, 1)

        self.distance_vs_rate = _SweepChart(
            title="Distance vs. Sifted Key Rate",
            button_label="Run Sweep (1–100 km)",
            param_name="distance_km",
            values=[1, 5, 10, 20, 30, 50, 75, 100],
            y_extractor=_rate_kbps,
            x_label="Distance (km), loss = distance × dB/km",
            y_label="Sifted key rate (kbps)",
            color="#3b82f6",
        )
        sweeps_grid.addWidget(self.distance_vs_rate, 0, 0)

        self.mu_vs_rate = _SweepChart(
            title="Mean Photon Number vs. Sifted Key Rate",
            button_label="Run Sweep (μ 0.05–1.0)",
            param_name="mean_photon_number",
            values=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
            y_extractor=_rate_kbps,
            x_label="Mean photon number per pulse μ",
            y_label="Sifted key rate (kbps)",
            color="#2ecc71",
        )
        sweeps_grid.addWidget(self.mu_vs_rate, 0, 1)

        self.visibility_vs_qber = _SweepChart(
            title="Interferometer Visibility vs. QBER",
            button_label="Run Sweep (0.80–1.00)",
            param_name="interferometer_visibility",
            values=[1.0, 0.98, 0.96, 0.94, 0.92, 0.90, 0.86, 0.80],
            y_extractor=_qber_percent,
            x_label="Interferometer visibility V",
            y_label="QBER (%)",
            color="#e74c3c",
        )
        sweeps_grid.addWidget(self.visibility_vs_qber, 1, 0)

        self.guard_vs_qber = _SweepChart(
            title="Temporal Filter vs. QBER (cf. Shaw et al. Fig. 11)",
            button_label="Run Sweep (guard 0–400 ps)",
            param_name="guard_s",
            values=[0, 50, 100, 150, 200, 250, 300, 400],
            y_extractor=_qber_percent,
            x_label="Guard band at the key-slot edges (ps)",
            y_label="QBER (%)",
            color="#f39c12",
            to_param=lambda ps: ps * 1e-12,
        )
        sweeps_grid.addWidget(self.guard_vs_qber, 1, 1)

        self._sweep_charts: List[_SweepChart] = [
            self.distance_vs_rate,
            self.mu_vs_rate,
            self.visibility_vs_qber,
            self.guard_vs_qber,
        ]

        sweeps_container = QWidget()
        sweeps_container.setMinimumHeight(560)
        sweeps_container.setLayout(sweeps_grid)
        layout.addWidget(sweeps_container, 1)

        scroll.setWidget(content)

    def set_parameters_provider(self, callback: Callable[[], SimulationParameters]) -> None:
        """Callback returning the current :class:`SimulationParameters`."""
        self._get_params_callback = callback
        for chart in self._sweep_charts:
            chart.set_parameters_provider(callback)

    def display_result(self, result: SimulationResult) -> None:
        # -- time-stamp histogram -------------------------------------------
        self.hist_plot.clear()
        if result.arrival_hist and len(result.hist_edges_ns) == len(result.arrival_hist) + 1:
            for start, end in result.port_windows_ns:
                region = pg.LinearRegionItem(values=(start, end), movable=False,
                                             brush=pg.mkBrush(59, 130, 246, 28),
                                             pen=pg.mkPen(59, 130, 246, 110))
                region.setZValue(-10)
                self.hist_plot.addItem(region)
            self.hist_plot.plot(result.hist_edges_ns, result.arrival_hist, stepMode="center",
                                fillLevel=0, brush=pg.mkBrush(46, 204, 113, 110),
                                pen=pg.mkPen("#2ecc71", width=1))
            n = result.bins_per_pulse
            ratio = " : ".join(["1"] + ["2"] * (n - 1) + ["1"])
            windows = ", ".join(f"{a:g}–{b:g} ns" for a, b in result.port_windows_ns)
            self.hist_note.setText(
                f"Shaded: port windows ({windows}). Each window holds {n + 1} slots in the "
                f"ratio {ratio}: early, {n - 1} interference (key) slots, late. "
                "Clicks between the windows come from IM leakage and dark counts."
            )
        else:
            self.hist_note.setText("No time-stamp histogram for this engine.")

        # -- click classification ---------------------------------------------
        self.clicks_plot.clear()
        categories = ["Bit 0", "Bit 1", "Early", "Late", "Guard", "Outside"]
        values = [result.d0_clicks, result.d1_clicks, result.early_clicks, result.late_clicks,
                  result.guard_discards, result.outside_window_clicks]
        colors = ["#3b82f6", "#f39c12", "#5c6773", "#5c6773", "#8e44ad", "#e74c3c"]
        bar = pg.BarGraphItem(x=list(range(len(categories))), height=values, width=0.6,
                              brushes=colors)
        self.clicks_plot.addItem(bar)
        axis = self.clicks_plot.getAxis("bottom")
        axis.setTicks([[(i, c) for i, c in enumerate(categories)]])

        eve_rate_pct = result.eve_actual_interception_rate * 100
        self.summary_label.setText(
            f"<b>Model:</b> {html.escape(result.model)}<br>"
            f"<b>Pulses sent:</b> {result.pulses_sent:,} "
            f"({result.duration_s * 1e3:.2f} ms)<br>"
            f"<b>Photons emitted:</b> {result.photons_emitted:,}<br>"
            f"<b>Eve:</b> {result.pulses_intercepted:,} pulses, "
            f"{result.photons_intercepted:,} photons ({eve_rate_pct:.1f}% of photons)<br>"
            f"<b>Detector clicks:</b> {result.raw_detector_clicks:,} "
            f"(signal {result.photons_detected:,}, IM leakage {result.leak_clicks:,}, "
            f"dark {result.dark_counts:,})<br>"
            f"<b>Lost in hold-off:</b> {result.holdoff_blocked:,}<br>"
            f"<b>Sifted key bits:</b> {result.sifted_key_bits:,}<br>"
            f"<b>Bit errors:</b> {result.bit_errors:,}<br>"
            f"<b>QBER:</b> {result.qber * 100:.2f}% "
            f"(expected {result.expected_qber * 100:.2f}%)<br>"
            f"<b>Sifted key rate:</b> {result.throughput_bps / 1000:.2f} kbps "
            f"(expected {result.expected_sifted_rate_bps / 1000:.2f} kbps)<br>"
            f"<b>Latency:</b> {result.latency_s * 1000:.3f} ms<br>"
            f"<b>Security:</b> {html.escape(result.security_message)}"
        )

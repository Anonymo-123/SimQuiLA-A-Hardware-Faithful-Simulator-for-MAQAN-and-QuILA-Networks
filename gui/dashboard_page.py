"""Dashboard page -- the primary live-simulation view.

The parameters describe the MAQAN / IIT Madras time-bin DPS hardware
("Model C"): Alice's pulse format (one N-bin pulse per period), the link,
and Bob's interferometer and gated detector. "MAQAN hardware" restores the
published values; "Ideal components" keeps the MAQAN format but switches
off every imperfection.
"""

from __future__ import annotations

import math
import time
from dataclasses import fields
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from backend.adapter import (
    EVENT_ENGINE_MAX_PULSES,
    LEGACY_ENGINES,
    SECURITY_THRESHOLDS,
    PulseEvent,
    SimulationParameters,
    SimulationResult,
    frame_layout,
)
from gui.bloch_sphere_view import BlochSphereView
from gui.loss_budget_dialog import LossBudgetDialog
from gui.topology_view import TopologyView
from gui.widgets import EventLogTable, KeyStreamWidget, LabeledSlider, StatRow, ToggleSwitch
from quila_sim.core import LossBudget, load_preset


def _load_maqan_links() -> list:
    """MAQAN links with a known loss, for the preset menu (empty if unavailable)."""
    try:
        return [lk for lk in load_preset("maqan").links if lk.effective_loss_db is not None]
    except (OSError, ValueError, ImportError):
        return []


#: Sifted bits shown in the live key streams (the full key is in the result).
KEY_DISPLAY_LIMIT = 2000

#: Bin widths offered in the GUI: (seconds, label).
BIN_WIDTHS = (
    (1.0e-9, "1.0 ns (DLI FSR 1 GHz, MAQAN)"),
    (0.5e-9, "0.5 ns (DLI FSR 2 GHz)"),
    (0.4e-9, "0.4 ns (DLI FSR 2.5 GHz)"),
    (2.0e-9, "2.0 ns (DLI FSR 500 MHz)"),
)

#: Hidden parameters: kept from loaded configurations, MAQAN values otherwise.
_HIDDEN = ("tdm_delay_s", "combiner_loss_db", "balance_loss_db", "gate_width_s")


def _compact_combo() -> QComboBox:
    """A combo box that fits the narrow parameter panel instead of widening it."""
    combo = QComboBox()
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(12)
    return combo


def _section(layout: QVBoxLayout, text: str) -> None:
    title = QLabel(text)
    title.setObjectName("SectionTitle")
    layout.addSpacing(8)
    layout.addWidget(title)


def _dim_label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setProperty("class", "dim")
    return label


def _fmt_duration(seconds: float) -> str:
    if seconds >= 1.0:
        return f"{seconds:.2f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds * 1e6:.1f} µs"


class DashboardPage(QWidget):
    """Live simulation dashboard: controls, topology animation, stats, log."""

    run_requested = pyqtSignal()
    reset_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._events: List[PulseEvent] = []
        self._result: Optional[SimulationResult] = None
        self._replay_index = 0
        self._playback_speed = 1.0
        self._d0_running = 0
        self._d1_running = 0
        self._loss_running = 0
        self._clicks_running = 0
        self._hidden: Dict[str, Any] = {k: getattr(SimulationParameters(), k) for k in _HIDDEN}

        self._replay_timer = QTimer(self)
        self._replay_timer.timeout.connect(self._advance_replay)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        splitter.addWidget(self._build_control_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([300, 700, 340])
        self.restore_defaults()

    # ------------------------------------------------------------------ #
    # Control panel (left)
    # ------------------------------------------------------------------ #

    def _build_control_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setFixedWidth(300)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(4)

        title = QLabel("SIMULATION PARAMETERS")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        layout.addSpacing(4)

        # -- Engine and hardware profile -----------------------------------
        layout.addWidget(QLabel("Simulation engine"))
        self.engine_combo = _compact_combo()
        self.engine_combo.addItem("Time-bin DPS, fast (MAQAN model)", "timebin")
        self.engine_combo.addItem("Time-bin DPS, SeQUeNCe per photon", "event")
        self.engine_combo.addItem("Continuous train (Model B, comparison)", "continuous")
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        layout.addWidget(self.engine_combo)
        self.engine_note = _dim_label()
        layout.addWidget(self.engine_note)
        layout.addSpacing(4)

        profile_row = QHBoxLayout()
        self.maqan_button = QPushButton("MAQAN hardware")
        self.maqan_button.setToolTip("Published MAQAN / IIT Madras values on the IITM-ERNET link")
        self.maqan_button.clicked.connect(self.restore_defaults)
        self.ideal_button = QPushButton("Ideal components")
        self.ideal_button.setToolTip(
            "MAQAN pulse format and receiver, but V = 1, no jitter, instantaneous phase "
            "modulator, perfect intensity modulator and no dark counts"
        )
        self.ideal_button.clicked.connect(self._apply_ideal_components)
        profile_row.addWidget(self.maqan_button)
        profile_row.addWidget(self.ideal_button)
        layout.addLayout(profile_row)

        self.pulses_slider = LabeledSlider(
            "Number of Pulses", 3.0, 8.5, 7.0, step=0.05, decimals=0,
            formatter=self._format_pulses,
        )
        layout.addWidget(self.pulses_slider)

        # -- Link ----------------------------------------------------------
        _section(layout, "LINK")
        layout.addWidget(QLabel("Preset"))
        self.preset_combo = _compact_combo()
        self.preset_combo.addItem("Custom link", None)
        for link in _load_maqan_links():
            self.preset_combo.addItem(
                f"MAQAN {link.name} · {link.effective_loss_db:g} dB", link
            )
            self.preset_combo.setItemData(
                self.preset_combo.count() - 1,
                f"{link.a} → {link.b}, {link.length_km:g} km, measured loss "
                f"{link.effective_loss_db:g} dB",
                Qt.ItemDataRole.ToolTipRole,
            )
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        layout.addWidget(self.preset_combo)

        self.distance_slider = LabeledSlider("Fiber Distance (km)", 0.1, 150.0, 4.8,
                                             step=0.1, decimals=1)
        layout.addWidget(self.distance_slider)

        measured_row = QHBoxLayout()
        measured_row.addWidget(QLabel("Use measured link loss"))
        measured_row.addStretch(1)
        self.measured_switch = ToggleSwitch(checked=True)
        measured_row.addWidget(self.measured_switch)
        layout.addLayout(measured_row)

        self.loss_slider = LabeledSlider("Fiber Loss (dB/km)", 0.0, 1.0, 0.20,
                                         step=0.01, decimals=2)
        self.measured_loss_slider = LabeledSlider(
            "Measured Link Loss (dB)", 0.0, 40.0, 7.5, step=0.1, decimals=1
        )
        layout.addWidget(self.loss_slider)
        layout.addWidget(self.measured_loss_slider)

        self.budget_button = QPushButton("Loss budget…")
        self.budget_button.clicked.connect(self._open_loss_budget)
        layout.addWidget(self.budget_button)
        self.rx_loss_slider = LabeledSlider(
            "Bob's Internal Loss (dB)", 0.0, 10.0, 0.0, step=0.1, decimals=1
        )
        self.rx_loss_slider.setToolTip("Filters, WDM and DLI insertion loss inside Bob")
        layout.addWidget(self.rx_loss_slider)
        self.link_summary = _dim_label()
        layout.addWidget(self.link_summary)

        # -- Transmitter -----------------------------------------------------
        _section(layout, "TRANSMITTER (ALICE)")
        self.period_slider = LabeledSlider(
            "Pulse Period (ns)", math.log10(4.0), 3.0, math.log10(32.0), step=0.002,
            formatter=lambda x: f"{self._period_ns(x):g}",
        )
        self.bins_slider = LabeledSlider("Time Bins per Pulse", 2.0, 8.0, 3.0, step=1.0, decimals=0)
        layout.addWidget(self.period_slider)
        layout.addWidget(self.bins_slider)
        layout.addWidget(QLabel("Time-bin width"))
        self.bin_combo = _compact_combo()
        for seconds, label in BIN_WIDTHS:
            self.bin_combo.addItem(label, seconds)
        layout.addWidget(self.bin_combo)
        self.mu_slider = LabeledSlider("Mean Photon Number per Pulse (μ)", 0.01, 1.0, 0.20,
                                       step=0.01, decimals=2)
        self.er_slider = LabeledSlider("IM Extinction Ratio (dB)", 5.0, 40.0, 14.8,
                                       step=0.1, decimals=1)
        self.er_slider.setToolTip("40 dB means a perfect intensity modulator (no leakage)")
        self.pm_slider = LabeledSlider("PM Transition Time (ps)", 0.0, 300.0, 42.0,
                                       step=1.0, decimals=0)
        for w in (self.mu_slider, self.er_slider, self.pm_slider):
            layout.addWidget(w)
        self.format_summary = _dim_label()
        layout.addWidget(self.format_summary)

        # -- Receiver --------------------------------------------------------
        _section(layout, "RECEIVER (BOB)")
        layout.addWidget(QLabel("Detectors"))
        self.layout_combo = _compact_combo()
        self.layout_combo.addItem("One SPD, time-multiplexed (MAQAN)", "tdm")
        self.layout_combo.addItem("Two SPDs, one per port", "dual")
        layout.addWidget(self.layout_combo)
        self.visibility_slider = LabeledSlider("Interferometer Visibility", 0.80, 1.0, 0.92,
                                               step=0.001, decimals=3)
        self.efficiency_slider = LabeledSlider("Detector Efficiency", 0.01, 1.0, 0.10,
                                               step=0.01, decimals=2)
        self.dark_slider = LabeledSlider(
            "Dark Counts per Gate (\u00d710\u207b\u2076)", 0.0, 100.0, 3.0, step=0.1, decimals=1,
        )
        self.holdoff_slider = LabeledSlider(
            "Detector Hold-off (\u00b5s)", -2.0, 2.0, 1.0, step=0.01,
            formatter=lambda x: "none" if x <= -2.0 else f"{10 ** x:.3g}",
        )
        self.jitter_slider = LabeledSlider("Timing Jitter (ps rms)", 0.0, 500.0, 162.0,
                                           step=1.0, decimals=0)
        self.guard_slider = LabeledSlider("Temporal Filter / Guard (ps)", 0.0, 500.0, 0.0,
                                          step=10.0, decimals=0)
        for w in (self.visibility_slider, self.efficiency_slider, self.dark_slider,
                  self.holdoff_slider, self.jitter_slider, self.guard_slider):
            layout.addWidget(w)
        self.receiver_summary = _dim_label()
        layout.addWidget(self.receiver_summary)

        _section(layout, "EVE AND SECURITY CHECK")
        layout.addWidget(QLabel("QBER threshold for \u201cSECURE\u201d"))
        self.threshold_combo = _compact_combo()
        for value, label in SECURITY_THRESHOLDS:
            self.threshold_combo.addItem(label, value)
        layout.addWidget(self.threshold_combo)
        layout.addWidget(_dim_label(
            "A check, not a security proof. MAQAN's DPS links run at a QBER of about "
            "11\u201312%."
        ))
        layout.addSpacing(6)

        eve_row = QHBoxLayout()
        eve_row.addWidget(QLabel("Eve on channel"))
        eve_row.addStretch(1)
        self.eve_checkbox = ToggleSwitch(checked=False, on_color="#ef4444")
        eve_row.addWidget(self.eve_checkbox)
        layout.addLayout(eve_row)
        layout.addSpacing(4)

        self.eve_prob_slider = LabeledSlider(
            "Eve Intercept Probability (per pulse)", 0.0, 1.0, 0.5, step=0.01, decimals=2
        )
        layout.addWidget(self.eve_prob_slider)
        layout.addSpacing(12)

        layout.addStretch(1)
        scroll.setWidget(container)

        self._loss_budget: Optional[LossBudget] = None
        self._applying_preset = False
        for w in (self.distance_slider, self.loss_slider, self.measured_loss_slider,
                  self.rx_loss_slider):
            w.valueChanged.connect(self._on_link_edited)
        self.measured_switch.toggled.connect(self._on_link_edited)
        for w in (self.period_slider, self.bins_slider, self.mu_slider, self.er_slider,
                  self.pm_slider, self.visibility_slider, self.efficiency_slider,
                  self.dark_slider, self.holdoff_slider, self.jitter_slider, self.guard_slider,
                  self.pulses_slider):
            w.valueChanged.connect(self._refresh_hardware_summary)
        self.bin_combo.currentIndexChanged.connect(self._refresh_hardware_summary)
        self.layout_combo.currentIndexChanged.connect(self._refresh_hardware_summary)
        self._on_engine_changed()
        self._refresh_link_controls()
        return scroll

    # -- Helpers ----------------------------------------------------------
    @staticmethod
    def _period_ns(x: float) -> float:
        """Pulse period shown for slider position x (log scale): whole ns, or
        tenths of a ns below 10 ns."""
        period = 10 ** x
        return round(period, 1) if period < 10 else float(round(period))

    def _pulse_rate_hz(self) -> float:
        return 1e9 / self._period_ns(self.period_slider.value())

    def _holdoff_s(self) -> float:
        x = self.holdoff_slider.value()
        return 0.0 if x <= -2.0 else float(f"{10 ** x:.3g}") / 1e6

    def _format_pulses(self, x: float) -> str:
        n = int(round(10 ** x))
        if hasattr(self, "period_slider"):
            period_s = self._period_ns(self.period_slider.value()) * 1e-9
            return f"{n:,} ({_fmt_duration(n * period_s)})"
        return f"{n:,}"

    def _num_pulses(self) -> int:
        n = int(round(10 ** self.pulses_slider.value()))
        if self.engine_combo.currentData() == "event":
            n = min(n, EVENT_ENGINE_MAX_PULSES)
        return max(2, n)

    def _set_num_pulses(self, n: int) -> None:
        self.pulses_slider.set_value(math.log10(max(2, int(n))))

    def _set_bin_width(self, seconds: float) -> None:
        best = min(range(self.bin_combo.count()),
                   key=lambda i: abs(self.bin_combo.itemData(i) - seconds))
        self.bin_combo.setCurrentIndex(best)

    def _on_engine_changed(self, *_args) -> None:
        engine = self.engine_combo.currentData()
        if engine == "event":
            self.engine_note.setText(
                "Every photon, interferometer exit and detector gate as a SeQUeNCe event; "
                f"slower, limited to {EVENT_ENGINE_MAX_PULSES:,} pulses."
            )
        elif engine == "continuous":
            self.engine_note.setText(
                "Textbook DPS with a gap-free pulse train: the same hardware run as one 1-bin "
                "pulse per time bin. NOT how MAQAN works – for comparison only."
            )
        else:
            self.engine_note.setText(
                "MAQAN time-bin DPS: one N-bin pulse per period, one-bin-delay interferometer, "
                "gated detector. Millions of pulses per second; statistics use the whole run."
            )
        self._refresh_hardware_summary()

    def _on_preset_changed(self, *_args) -> None:
        link = self.preset_combo.currentData()
        if link is None:
            self._refresh_link_controls()
            return
        self._applying_preset = True
        try:
            self.distance_slider.set_value(float(link.length_km))
            self.measured_loss_slider.set_value(float(link.effective_loss_db))
            self.measured_switch.setChecked(True)
            self._loss_budget = link.loss_budget
        finally:
            self._applying_preset = False
        self._refresh_link_controls()

    def _on_link_edited(self, *_args) -> None:
        if not self._applying_preset:
            if self.preset_combo.currentIndex() != 0:
                self.preset_combo.blockSignals(True)
                self.preset_combo.setCurrentIndex(0)
                self.preset_combo.blockSignals(False)
            self._loss_budget = None
        self._refresh_link_controls()

    def _open_loss_budget(self) -> None:
        dialog = LossBudgetDialog(self.distance_slider.value(), self._loss_budget, self)
        if dialog.exec():
            budget = dialog.budget()
            self._applying_preset = True
            try:
                if budget.fibre_length_km:
                    self.distance_slider.set_value(budget.fibre_length_km)
                self.measured_loss_slider.set_value(budget.total_db)
                self.measured_switch.setChecked(True)
            finally:
                self._applying_preset = False
            if self.preset_combo.currentIndex() != 0:
                self.preset_combo.blockSignals(True)
                self.preset_combo.setCurrentIndex(0)
                self.preset_combo.blockSignals(False)
            self._loss_budget = budget
            self._refresh_link_controls()

    def _channel_loss_db(self) -> float:
        if self.measured_switch.isChecked():
            return self.measured_loss_slider.value()
        return self.distance_slider.value() * self.loss_slider.value()

    def _refresh_link_controls(self) -> None:
        measured = self.measured_switch.isChecked()
        self.loss_slider.setEnabled(not measured)
        self.measured_loss_slider.setEnabled(measured)
        self.loss_slider.set_title("Fiber Loss (dB/km)" + ("  – not used" if measured else ""))
        self.measured_loss_slider.set_title(
            "Measured Link Loss (dB)" + ("" if measured else "  – off")
        )
        loss = self._channel_loss_db()
        if measured:
            text = f"Channel loss {loss:.2f} dB (measured)."
            if self._loss_budget is not None:
                text += " " + self._loss_budget.summary() + "."
        else:
            text = (f"Channel loss {loss:.2f} dB = {self.distance_slider.value():.1f} km "
                    f"× {self.loss_slider.value():.2f} dB/km (no splices or connectors).")
        rx = self.rx_loss_slider.value()
        if rx > 0:
            text += f" Plus {rx:.1f} dB inside Bob."
        self.link_summary.setText(text)

    def _refresh_hardware_summary(self, *_args) -> None:
        """Describe the pulse format and receiver, or say what is inconsistent."""
        if not hasattr(self, "receiver_summary"):
            return
        self.pulses_slider.refresh_label()
        params = self.read_parameters()
        n = int(params.bins_per_pulse)
        bin_ns = params.bin_width_s * 1e9
        period_ns = 1e9 / params.pulse_rate_hz
        try:
            lay = frame_layout(params)
        except ValueError as exc:
            self.format_summary.setText(f"<span style='color:#ef4444'>⚠ {exc}</span>")
            self.receiver_summary.setText("")
            return
        weights = " : ".join(["1"] + ["2"] * (n - 1) + ["1"])
        self.format_summary.setText(
            f"One {n}-bin pulse ({n * bin_ns:g} ns of light) every {period_ns:g} ns "
            f"= {params.pulse_rate_hz / 1e6:.4g} MHz; {lay.frame_bins - n} empty bins. "
            f"The interferometer gives {n + 1} slots in the ratio {weights}; "
            f"{n - 1} carry key bits."
        )
        if params.receiver_layout == "tdm":
            rx_text = (f"One gated SPD. Port 1 delayed {params.tdm_delay_s * 1e9:g} ns; "
                       f"{params.combiner_loss_db + params.balance_loss_db:g} dB coupler + "
                       f"attenuator loss. Gate {lay.gate_width * bin_ns:g} ns per period: "
                       f"port-0 window 0–{(n + 1) * bin_ns:g} ns, port-1 window "
                       f"{lay.port_offset[1] * bin_ns:g}–"
                       f"{(lay.port_offset[1] + n + 1) * bin_ns:g} ns.")
        else:
            rx_text = (f"Two gated SPDs, one per port; gate {lay.gate_width * bin_ns:g} ns "
                       "per period.")
        blind = lay.blind_gates(params.holdoff_s, lay.gate_start + lay.gate_width / 2)
        if blind:
            rx_text += f" After a click the detector skips {blind:,} gates (hold-off)."
        self.receiver_summary.setText(rx_text)

    def _apply_ideal_components(self) -> None:
        self.visibility_slider.set_value(1.0)
        self.jitter_slider.set_value(0.0)
        self.pm_slider.set_value(0.0)
        self.er_slider.set_value(40.0)
        self.dark_slider.set_value(0.0)
        self._refresh_hardware_summary()

    def restore_defaults(self) -> None:
        """Reset every control to the published MAQAN hardware (IITM-ERNET link)."""
        self.apply_parameters(SimulationParameters())
        maqan_index = next((i for i in range(1, self.preset_combo.count())
                            if self.preset_combo.itemData(i).name == "IITM-ERNET"), 0)
        self.preset_combo.setCurrentIndex(maqan_index)

    def read_parameters(self) -> SimulationParameters:
        link = self.preset_combo.currentData()
        er = round(self.er_slider.value(), 3)
        return SimulationParameters(
            num_pulses=self._num_pulses(),
            engine=self.engine_combo.currentData(),
            distance_km=round(self.distance_slider.value(), 4),
            fiber_loss_db_km=round(self.loss_slider.value(), 4),
            channel_loss_db=(round(self.measured_loss_slider.value(), 4)
                             if self.measured_switch.isChecked() else None),
            receiver_loss_db=round(self.rx_loss_slider.value(), 4),
            link_name=link.name if link is not None else "",
            pulse_rate_hz=self._pulse_rate_hz(),
            bins_per_pulse=int(round(self.bins_slider.value())),
            bin_width_s=self.bin_combo.currentData(),
            mean_photon_number=round(self.mu_slider.value(), 4),
            extinction_ratio_db=None if er >= 40.0 else er,
            pm_transition_s=round(self.pm_slider.value()) / 1e12,
            interferometer_visibility=round(self.visibility_slider.value(), 4),
            receiver_layout=self.layout_combo.currentData(),
            detector_efficiency=round(self.efficiency_slider.value(), 4),
            dark_count_prob=round(self.dark_slider.value(), 2) / 1e6,
            holdoff_s=self._holdoff_s(),
            timing_jitter_s=round(self.jitter_slider.value()) / 1e12,
            guard_s=round(self.guard_slider.value()) / 1e12,
            eve_enabled=self.eve_checkbox.isChecked(),
            eve_intercept_probability=round(self.eve_prob_slider.value(), 4),
            qber_threshold=self.threshold_combo.currentData(),
            **self._hidden,
        )

    def apply_parameters(self, params: Any) -> None:
        """Set the controls from parameters (a SimulationParameters, or a dict
        of its fields as saved in a configuration file)."""
        if isinstance(params, dict):
            from backend.adapter import params_from_dict

            params, _notes = params_from_dict(params)
        defaults = {f.name: getattr(params, f.name) for f in fields(SimulationParameters)}
        engine = LEGACY_ENGINES.get(defaults["engine"], defaults["engine"])
        self.engine_combo.setCurrentIndex(max(0, self.engine_combo.findData(engine)))
        self._set_num_pulses(int(defaults["num_pulses"]))
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.blockSignals(False)
        self._applying_preset = True
        try:
            self.distance_slider.set_value(float(defaults["distance_km"]))
            self.loss_slider.set_value(float(defaults["fiber_loss_db_km"]))
            measured = defaults["channel_loss_db"]
            self.measured_switch.setChecked(measured is not None)
            if measured is not None:
                self.measured_loss_slider.set_value(float(measured))
            self.rx_loss_slider.set_value(float(defaults["receiver_loss_db"]))
        finally:
            self._applying_preset = False
        self._loss_budget = None
        self.period_slider.set_value(math.log10(1e9 / float(defaults["pulse_rate_hz"])))
        self.bins_slider.set_value(int(defaults["bins_per_pulse"]))
        self._set_bin_width(float(defaults["bin_width_s"]))
        self.mu_slider.set_value(float(defaults["mean_photon_number"]))
        er = defaults["extinction_ratio_db"]
        self.er_slider.set_value(40.0 if er is None else float(er))
        self.pm_slider.set_value(float(defaults["pm_transition_s"]) * 1e12)
        self.layout_combo.setCurrentIndex(max(0, self.layout_combo.findData(
            defaults["receiver_layout"])))
        self.visibility_slider.set_value(float(defaults["interferometer_visibility"]))
        self.efficiency_slider.set_value(float(defaults["detector_efficiency"]))
        self.dark_slider.set_value(float(defaults["dark_count_prob"]) * 1e6)
        holdoff_us = float(defaults["holdoff_s"]) * 1e6
        self.holdoff_slider.set_value(math.log10(holdoff_us) if holdoff_us > 0.01 else -2.0)
        self.jitter_slider.set_value(float(defaults["timing_jitter_s"]) * 1e12)
        self.guard_slider.set_value(float(defaults["guard_s"]) * 1e12)
        self.eve_checkbox.setChecked(bool(defaults["eve_enabled"]))
        self.eve_prob_slider.set_value(float(defaults["eve_intercept_probability"]))
        threshold = float(defaults["qber_threshold"])
        index = self.threshold_combo.findData(threshold)
        if index < 0:
            self.threshold_combo.addItem(f"{threshold:.0%} \u2013 from the loaded configuration",
                                         threshold)
            index = self.threshold_combo.count() - 1
        self.threshold_combo.setCurrentIndex(index)
        for key in _HIDDEN:
            self._hidden[key] = defaults[key]
        name = defaults.get("link_name") or ""
        measured = defaults["channel_loss_db"]
        if name and measured is not None:
            for i in range(1, self.preset_combo.count()):
                link = self.preset_combo.itemData(i)
                same_loss = abs(float(link.effective_loss_db) - float(measured)) < 1e-6
                if link.name == name and same_loss:
                    self.preset_combo.blockSignals(True)
                    self.preset_combo.setCurrentIndex(i)
                    self.preset_combo.blockSignals(False)
                    self._loss_budget = link.loss_budget
                    break
        self._refresh_link_controls()
        self._refresh_hardware_summary()

    # ------------------------------------------------------------------ #
    # Center panel: topology + legend
    # ------------------------------------------------------------------ #

    def _build_center_panel(self) -> QWidget:
        container = QFrame()
        container.setObjectName("Panel")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self.topology_view = TopologyView()
        layout.addWidget(self.topology_view, 1)

        legend = QHBoxLayout()
        legend.setContentsMargins(14, 6, 14, 10)
        legend.addWidget(self._legend_dot("#3b82f6", "Key bit 0"))
        legend.addWidget(self._legend_dot("#f39c12", "Key bit 1"))
        legend.addWidget(self._legend_dot("#e74c3c", "Eve intercepted"))
        legend.addWidget(self._legend_dot("#94a3b8", "No key bit (early/late, outside)"))
        legend.addStretch(1)
        self.status_label = QLabel("Ready.")
        self.status_label.setProperty("class", "dim")
        legend.addWidget(self.status_label)
        layout.addLayout(legend)
        return container

    def _legend_dot(self, color: str, text: str) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 16, 0)
        dot = QLabel("●")
        dot.setStyleSheet(f"color: {color}; font-size: 14px;")
        label = QLabel(text)
        h.addWidget(dot)
        h.addWidget(label)
        return w

    # ------------------------------------------------------------------ #
    # Right panel: live stats, keys, event log
    # ------------------------------------------------------------------ #

    def _build_right_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setFixedWidth(340)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        stats_box = QGroupBox("Live Statistics")
        stats_layout = QVBoxLayout(stats_box)
        stats_layout.setContentsMargins(10, 6, 10, 8)
        stats_layout.setSpacing(0)

        self.stat_pulses = StatRow("Pulses Sent")
        self.stat_duration = StatRow("Simulated Time")
        self.stat_link_loss = StatRow("Channel Loss")
        self.stat_loss = StatRow("Pulses Without Click")
        self.stat_clicks = StatRow("Detector Clicks")
        self.stat_key_clicks = StatRow("Key-Slot Clicks")
        self.stat_early_late = StatRow("Early/Late Discards")
        self.stat_guard = StatRow("Guard-Band Discards")
        self.stat_outside = StatRow("Outside Port Windows")
        self.stat_leak = StatRow("IM-Leakage Clicks")
        self.stat_dark = StatRow("Dark Counts")
        self.stat_blocked = StatRow("Lost in Hold-off")
        self.stat_eve = StatRow("Bits From Eve")
        self.stat_sifted = StatRow("Sifted Key Bits")
        self.stat_rate = StatRow("Sifted Key Rate")
        self.stat_rate_expected = StatRow("  expected (closed form)")
        self.stat_errors = StatRow("Bit Errors")
        self.stat_qber = StatRow("QBER")
        self.stat_qber_expected = StatRow("  expected (closed form)")
        self.stat_d0 = StatRow("Bits 0 (port-0 window)")
        self.stat_d1 = StatRow("Bits 1 (port-1 window)")
        self._stat_rows = (
            self.stat_pulses, self.stat_duration, self.stat_link_loss, self.stat_loss,
            self.stat_clicks, self.stat_key_clicks, self.stat_early_late, self.stat_guard,
            self.stat_outside, self.stat_leak, self.stat_dark, self.stat_blocked,
            self.stat_eve, self.stat_sifted, self.stat_rate, self.stat_rate_expected,
            self.stat_errors, self.stat_qber, self.stat_qber_expected,
            self.stat_d0, self.stat_d1,
        )
        for row in self._stat_rows:
            stats_layout.addWidget(row)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #252c37; background-color: #252c37;")
        divider.setFixedHeight(1)
        stats_layout.addSpacing(4)
        stats_layout.addWidget(divider)
        stats_layout.addSpacing(4)

        self.stat_security = StatRow("Security Status")
        stats_layout.addWidget(self.stat_security)

        layout.addWidget(stats_box)

        keys_box = QGroupBox("Sifted Keys (live)")
        keys_layout = QVBoxLayout(keys_box)

        alice_row = QHBoxLayout()
        alice_tag = QLabel("Alice")
        alice_tag.setStyleSheet("color:#3b82f6; font-weight:700; min-width: 40px;")
        alice_row.addWidget(alice_tag)
        self.alice_key_stream = KeyStreamWidget(accent="#3b82f6")
        alice_row.addWidget(self.alice_key_stream, 1)
        keys_layout.addLayout(alice_row)

        bob_row = QHBoxLayout()
        bob_tag = QLabel("Bob")
        bob_tag.setStyleSheet("color:#22c55e; font-weight:700; min-width: 40px;")
        bob_row.addWidget(bob_tag)
        self.bob_key_stream = KeyStreamWidget(accent="#22c55e")
        bob_row.addWidget(self.bob_key_stream, 1)
        keys_layout.addLayout(bob_row)

        self.key_match_label = QLabel("")
        keys_layout.addWidget(self.key_match_label)
        layout.addWidget(keys_box, 0)

        log_box = QGroupBox("Event Log (pulses with a click)")
        log_layout = QVBoxLayout(log_box)
        self.event_log = EventLogTable()
        self.event_log.setMinimumHeight(200)
        log_layout.addWidget(self.event_log)
        layout.addWidget(log_box, 5)

        bloch_box = QGroupBox("Bloch Sphere (display of key clicks)")
        bloch_layout = QVBoxLayout(bloch_box)
        self.bloch_view = BlochSphereView()
        self.bloch_view.setMinimumHeight(180)
        bloch_layout.addWidget(self.bloch_view)
        layout.addWidget(bloch_box, 3)

        self._sim_start_time = 0.0
        self._sifted_reveal_index = 0

        scroll.setWidget(panel)
        return scroll

    # ------------------------------------------------------------------ #
    # Simulation lifecycle hooks (called by MainWindow)
    # ------------------------------------------------------------------ #

    def reset_view(self) -> None:
        self._replay_timer.stop()
        self._events = []
        self._result = None
        self._replay_index = 0
        self._d0_running = self._d1_running = self._loss_running = self._clicks_running = 0
        self.topology_view.reset()
        self.event_log.clear_log()
        self.bloch_view.clear_points()
        self.alice_key_stream.clear_bits()
        self.bob_key_stream.clear_bits()
        self._sifted_reveal_index = 0
        for row in self._stat_rows:
            row.set_value("—")
        self.stat_security.set_value("pending")
        self.key_match_label.setText("")
        self.status_label.setText("Ready.")

    def set_playback_speed(self, speed: float) -> None:
        self._playback_speed = max(0.1, speed)

    def load_result(self, result: SimulationResult) -> None:
        """Called when the backend finishes; prepares (but does not yet
        play) the replay of the recorded trace."""
        self.reset_view()
        self._result = result
        self._events = result.events
        self.topology_view.set_receiver_layout(result.receiver_layout or "tdm",
                                               continuous=result.engine == "continuous")
        self.topology_view.set_eve_visible(
            result.eve_intercepted_count > 0 or any(e.eve_intercepted for e in result.events)
        )
        what = "slots" if result.engine == "continuous" else "pulses with a click"
        if result.trace_truncated:
            self.status_label.setText(
                f"Simulation complete — {result.pulses_sent:,} pulses; replaying the "
                f"first {len(self._events):,} {what} (statistics use the whole run)."
            )
        else:
            self.status_label.setText(
                f"Simulation complete — {len(self._events):,} {what} recorded."
            )

    def start_replay(self) -> None:
        if not self._events:
            self._finalize_stats()
            return
        self._replay_index = 0
        self._d0_running = self._d1_running = self._loss_running = self._clicks_running = 0
        self._sifted_reveal_index = 0
        self.event_log.clear_log()
        self.alice_key_stream.clear_bits()
        self.bob_key_stream.clear_bits()
        self._sim_start_time = time.monotonic()
        self._replay_timer.start(16)

    def pause_replay(self) -> None:
        self._replay_timer.stop()

    def resume_replay(self) -> None:
        if self._events:
            self._replay_timer.start(16)

    def stop_replay(self) -> None:
        self._replay_timer.stop()

    # ------------------------------------------------------------------ #
    # Replay engine
    # ------------------------------------------------------------------ #

    def _reveal_key_bits(self, count: int) -> None:
        r = self._result
        if r is None:
            return
        limit = min(len(r.alice_key_bits), len(r.bob_key_bits), KEY_DISPLAY_LIMIT)
        for _ in range(count):
            if self._sifted_reveal_index >= limit:
                return
            i = self._sifted_reveal_index
            a_bit, b_bit = r.alice_key_bits[i], r.bob_key_bits[i]
            mismatch = a_bit != b_bit
            self.alice_key_stream.append_bit(a_bit, mismatch)
            self.bob_key_stream.append_bit(b_bit, mismatch)
            self._sifted_reveal_index += 1

    def _advance_replay(self) -> None:
        total = len(self._events)
        if total == 0 or self._replay_index >= total:
            self._replay_timer.stop()
            self._finalize_stats()
            return

        # Scale batch size to the trace length so long traces stay smooth.
        base_batch = max(1, round(total / 300))
        batch = max(1, round(base_batch * self._playback_speed))
        end = min(total, self._replay_index + batch)

        launch_this_tick = None
        log_rows = []
        for i in range(self._replay_index, end):
            evt = self._events[i]
            if evt.clicked:
                self._clicks_running += 1
                if evt.kind == "key" and evt.bob_bit is not None:
                    if evt.bob_bit == 0:
                        self._d0_running += 1
                    else:
                        self._d1_running += 1
                self._reveal_key_bits(evt.n_key)
            else:
                self._loss_running += 1
            launch_this_tick = evt
            log_rows.append(self._make_log_row(evt))
            self.bloch_view.add_point(
                evt.bloch_x, evt.bloch_y, evt.bloch_z,
                dark=evt.dark_count, lost=not (evt.clicked and evt.kind == "key"),
            )

        if log_rows:
            self.event_log.append_events_batch(log_rows)

        if launch_this_tick is not None:
            speed = 0.10 if total < 500 else 0.22
            evt = launch_this_tick
            self.topology_view.launch_photon(
                evt.phase, evt.eve_intercepted, speed=speed, keyless=evt.kind != "key"
            )
            if evt.clicked and evt.detector is not None:
                QTimer.singleShot(int(1 / speed * 16 * 0.9),
                                  lambda d=evt.detector: self.topology_view.flash_detector(d))

        self._replay_index = end
        self._update_live_stats()

    def _make_log_row(self, evt: PulseEvent) -> tuple:
        """Build one (pulse #, phases, click, event, color) row."""
        if evt.phases:
            phases = evt.phase_text
        elif evt.alice_bit is not None:
            phases = "Δφ = " + ("π" if evt.alice_bit else "0")
        else:
            phases = "—"
        if evt.clicked and evt.detector is not None and evt.slot is not None:
            click = f"P{evt.detector} s{evt.slot} {evt.time_ns:.2f}ns"
        elif evt.clicked and evt.detector is not None:
            click = f"D{evt.detector}"
        elif evt.clicked:
            click = f"{evt.time_ns:.2f} ns"
        else:
            click = "—"
        event_text, color = EventLogTable.classify(evt)
        return (evt.index, phases, click, event_text, color)

    def _update_live_stats(self) -> None:
        n = self._replay_index
        eve_count = sum(1 for e in self._events[:n] if e.eve_intercepted)
        self.stat_loss.set_value(f"{self._loss_running}")
        self.stat_clicks.set_value(f"{self._clicks_running}")
        self.stat_eve.set_value(f"{eve_count}")
        self.stat_d0.set_value(f"{self._d0_running}")
        self.stat_d1.set_value(f"{self._d1_running}")

    def _finalize_stats(self) -> None:
        if self._result is None:
            return
        r = self._result
        self.stat_pulses.set_value(f"{r.pulses_sent:,}")
        self.stat_duration.set_value(_fmt_duration(r.duration_s))
        self.stat_link_loss.set_value(f"{r.channel_loss_db_used:.2f} dB")
        self.stat_loss.set_value(f"{r.channel_loss_count:,}")
        self.stat_clicks.set_value(f"{r.raw_detector_clicks:,}")
        self.stat_key_clicks.set_value(f"{r.key_slot_clicks:,}")
        self.stat_early_late.set_value(f"{r.early_late_discards:,}")
        self.stat_guard.set_value(f"{r.guard_discards:,}")
        self.stat_outside.set_value(f"{r.outside_window_clicks:,}")
        self.stat_leak.set_value(f"{r.leak_clicks:,}")
        self.stat_dark.set_value(f"{r.dark_counts:,}")
        self.stat_blocked.set_value(f"{r.holdoff_blocked:,}")
        self.stat_eve.set_value(f"{r.eve_intercepted_count:,}")
        self.stat_d0.set_value(f"{r.d0_clicks:,}")
        self.stat_d1.set_value(f"{r.d1_clicks:,}")
        self.stat_sifted.set_value(f"{r.sifted_key_bits:,}")
        self.stat_rate.set_value(f"{r.throughput_bps / 1000:.2f} kbps")
        self.stat_rate_expected.set_value(f"{r.expected_sifted_rate_bps / 1000:.2f} kbps")
        self.stat_errors.set_value(f"{r.bit_errors:,}")
        qber_color = None
        if r.sifted_key_bits:
            qber_color = "#2ecc71" if r.security_secure else "#e74c3c"
        self.stat_qber.set_value(f"{r.qber * 100:.2f}%", qber_color)
        self.stat_qber_expected.set_value(f"{r.expected_qber * 100:.2f}%")

        if r.security_secure:
            self.stat_security.set_value(r.security_message, "#2ecc71")
        elif r.sifted_key_bits == 0:
            self.stat_security.set_value(r.security_message, "#f39c12")
        else:
            self.stat_security.set_value(r.security_message, "#e74c3c")

        if r.alice_key_bits and r.bob_key_bits:
            # Reveal any remaining bits the replay loop hadn't caught up to yet
            # (e.g. if the user paused mid-run then let it finish).
            shown_limit = min(len(r.alice_key_bits), KEY_DISPLAY_LIMIT)
            self._reveal_key_bits(shown_limit - self._sifted_reveal_index)
            match = r.bit_errors == 0
            shown = (f" (first {shown_limit:,} of {len(r.alice_key_bits):,} bits shown)"
                     if len(r.alice_key_bits) > shown_limit else "")
            self.key_match_label.setText(
                ("✓ Keys match exactly" if match else
                 f"✗ {r.bit_errors:,} mismatched bit(s)") + shown
            )
            self.key_match_label.setStyleSheet(
                "color: #22c55e; font-weight:600;" if match else "color: #ef4444; font-weight:600;"
            )
        else:
            self.key_match_label.setText("No sifted key generated.")
            self.key_match_label.setStyleSheet("color: #8a93a3;")

        self.status_label.setText("Replay complete.")

"""Settings page -- appearance, replay behaviour, and default simulation
parameters. Every control here is wired to something real: the theme
toggle drives the live stylesheet, the replay defaults drive the toolbar
sliders' starting position, and "Restore Factory Defaults" resets the
Dashboard's own parameter sliders on request.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class SettingsPage(QWidget):
    """Application preferences."""

    theme_changed = pyqtSignal(str)
    restore_defaults_requested = pyqtSignal()
    default_speed_changed = pyqtSignal(int)   # 1-50, matches toolbar speed slider
    default_zoom_changed = pyqtSignal(int)    # 5-20, matches toolbar zoom slider

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("SETTINGS")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        # ------------------------------------------------------------- #
        # Appearance
        # ------------------------------------------------------------- #
        appearance_box = QGroupBox("Appearance")
        form = QFormLayout(appearance_box)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.currentTextChanged.connect(
            lambda text: self.theme_changed.emit(text.lower())
        )
        form.addRow("Theme:", self.theme_combo)
        layout.addWidget(appearance_box)

        # ------------------------------------------------------------- #
        # Replay defaults -- these set the starting position of the
        # toolbar's Speed / Zoom sliders the next time the app runs a
        # replay; they are genuinely read by MainWindow, not decorative.
        # ------------------------------------------------------------- #
        replay_box = QGroupBox("Replay Defaults")
        replay_form = QFormLayout(replay_box)

        self.default_speed_spin = QSpinBox()
        self.default_speed_spin.setRange(1, 50)
        self.default_speed_spin.setValue(10)
        self.default_speed_spin.setSuffix("  (1 = slowest, 50 = fastest)")
        self.default_speed_spin.valueChanged.connect(self.default_speed_changed.emit)
        replay_form.addRow("Default playback speed:", self.default_speed_spin)

        self.default_zoom_spin = QSpinBox()
        self.default_zoom_spin.setRange(5, 20)
        self.default_zoom_spin.setValue(10)
        self.default_zoom_spin.valueChanged.connect(self.default_zoom_changed.emit)
        replay_form.addRow("Default topology zoom:", self.default_zoom_spin)

        layout.addWidget(replay_box)

        # ------------------------------------------------------------- #
        # Simulation defaults
        # ------------------------------------------------------------- #
        sim_box = QGroupBox("Simulation Parameters")
        sim_layout = QVBoxLayout(sim_box)
        sim_note = QLabel(
            "Resets every control on the Dashboard's Simulation Parameters panel to the "
            "published MAQAN / IIT Madras hardware on the IITM\u2013ERNET link: 10 million "
            "pulses, 4.8 km with 7.5 dB measured loss, one 3 \u00d7 1 ns pulse every 32 ns, "
            "\u03bc = 0.2 per pulse, IM extinction ratio 14.8 dB, PM transition 42 ps, "
            "visibility 0.92, one time-multiplexed gated detector (\u03b7 = 0.10, "
            "3\u00d710\u207b\u2076 dark counts per gate, 10 \u00b5s hold-off, 162 ps jitter), "
            "Eve off."
        )
        sim_note.setWordWrap(True)
        sim_note.setProperty("class", "dim")
        sim_layout.addWidget(sim_note)

        restore_btn = QPushButton("Restore Factory Defaults")
        restore_btn.clicked.connect(self.restore_defaults_requested.emit)
        sim_layout.addWidget(restore_btn)
        layout.addWidget(sim_box)

        # ------------------------------------------------------------- #
        # Physics model reference -- factual, not marketing copy.
        # ------------------------------------------------------------- #
        physics_box = QGroupBox("Physics Model")
        physics_layout = QVBoxLayout(physics_box)
        physics_text = QLabel(
            "Protocol: time-bin Differential Phase Shift QKD as built at IIT Madras for "
            "MAQAN (Shaw et al., IEEE Photonics J. 2022; Sharma et al., QIP 2026).<br>"
            "Alice: the intensity modulator carves one pulse of N = 3 time bins (1 ns each) "
            "every 32 ns; the phase modulator writes 0 or \u03c0 into each bin; the rest of "
            "the period is empty except for light leaking through the IM.<br>"
            "Bob: a one-bin-delay interferometer turns each pulse into N + 1 slots "
            "(1/6 : 1/3 : 1/3 : 1/6) per port; the two interference slots carry the phase "
            "differences \u03c6<sub>1</sub> XOR \u03c6<sub>0</sub> and "
            "\u03c6<sub>2</sub> XOR \u03c6<sub>1</sub>. Port 1 is delayed by 10 ns onto the "
            "same gated InGaAs detector, so the time window of a click is Bob's bit. The "
            "model includes detector jitter, phase-modulator transitions, IM leakage, "
            "dark counts per gate and the 10 \u00b5s hold-off; each imperfection reproduces "
            "its line of the published QBER budget (Shaw et al., Table III). Afterpulsing "
            "is not modelled yet.<br>"
            "Link loss: measured end-to-end loss (e.g. MAQAN preset) or distance "
            "\u00d7 dB/km.<br>"
            "Security check: \u201cSECURE\u201d when the QBER is below the threshold chosen on "
            "the Dashboard \u2013 20% (critical QBER of 3-pulse DPS against individual attacks, "
            "MAQAN analysis; default), 11% (BB84 demonstration value) or 5% (information-"
            "theoretic DPS bounds). A check, not a security proof.<br>"
            "Engines: a fast vectorised time-bin engine (default) and SeQUeNCe's "
            "event-driven per-photon kernel as a reference; both share the same pulse "
            "timing and sifting code and are cross-checked in the tests. A continuous-train "
            "engine (textbook DPS, \u201cModel B\u201d) is kept for comparison only."
        )
        physics_text.setWordWrap(True)
        physics_layout.addWidget(physics_text)
        layout.addWidget(physics_box)

        layout.addStretch(1)

    def set_theme_combo(self, theme_name: str) -> None:
        self.theme_combo.blockSignals(True)
        self.theme_combo.setCurrentText(theme_name.capitalize())
        self.theme_combo.blockSignals(False)

"""Main application window: toolbar, navigation, and page orchestration."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from typing import Optional

from PyQt6.QtCore import QThread, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QStackedWidget,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from backend.adapter import SimulationResult, params_from_dict, total_channel_loss_db
from gui.analytics_page import AnalyticsPage
from gui.cascade_page import CascadePage
from gui.dashboard_page import DashboardPage
from gui.settings_page import SettingsPage
from gui.theme import DARK, LIGHT, build_stylesheet
from gui.validation_page import ValidationPage
from gui.worker import start_simulation


class MainWindow(QMainWindow):
    """Top-level window hosting the toolbar, side navigation, and pages."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(
            "DPS-QKD Simulator \u2014 time-bin Differential Phase Shift QKD (MAQAN hardware)"
        )
        self.resize(1440, 900)

        self._theme_name = "dark"
        self._thread: Optional[QThread] = None
        self._worker = None
        self._last_result: Optional[SimulationResult] = None

        self.dashboard_page = DashboardPage()
        self.analytics_page = AnalyticsPage()
        self.analytics_page.set_parameters_provider(self.dashboard_page.read_parameters)
        self.validation_page = ValidationPage()
        self.validation_page.set_parameters_provider(self.dashboard_page.read_parameters)
        self.cascade_page = CascadePage()
        self.cascade_page.set_result_provider(lambda: self._last_result)
        self.settings_page = SettingsPage()
        self.settings_page.theme_changed.connect(self.apply_theme)
        self.settings_page.restore_defaults_requested.connect(self._on_restore_defaults)
        self.settings_page.default_speed_changed.connect(self._on_default_speed_changed)
        self.settings_page.default_zoom_changed.connect(self._on_default_zoom_changed)

        central = QWidget()
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_nav_rail())

        self.stack = QStackedWidget()
        self.stack.addWidget(self.dashboard_page)
        self.stack.addWidget(self.analytics_page)
        self.stack.addWidget(self.validation_page)
        self.stack.addWidget(self.cascade_page)
        self.stack.addWidget(self.settings_page)
        outer.addWidget(self.stack, 1)

        self.setCentralWidget(central)

        self._build_toolbar()

        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage("Ready.")

        self.apply_theme("dark")
        self._set_running_state(False)

    # ------------------------------------------------------------------ #
    # Navigation
    # ------------------------------------------------------------------ #

    def _build_nav_rail(self) -> QWidget:
        rail = QWidget()
        rail.setFixedWidth(150)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(2)

        self._nav_buttons = []
        for i, label in enumerate(["Dashboard", "Analytics", "Validation", "Cascade EC", "Settings"]):
            btn = QPushButton(label)
            btn.setObjectName("NavButton")
            btn.setCheckable(True)
            btn.setChecked(i == 0)
            btn.clicked.connect(lambda _checked, idx=i: self._switch_page(idx))
            layout.addWidget(btn)
            self._nav_buttons.append(btn)
        layout.addStretch(1)
        return rail

    def _switch_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self._nav_buttons):
            btn.setChecked(i == index)

    # ------------------------------------------------------------------ #
    # Toolbar
    # ------------------------------------------------------------------ #

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.start_btn = QPushButton("\u25b6 Start")
        self.start_btn.setObjectName("PrimaryButton")
        self.start_btn.clicked.connect(self._on_start)
        toolbar.addWidget(self.start_btn)

        self.pause_btn = QPushButton("\u23f8 Pause")
        self.pause_btn.clicked.connect(self._on_pause)
        toolbar.addWidget(self.pause_btn)

        self.resume_btn = QPushButton("\u25b6 Resume")
        self.resume_btn.clicked.connect(self._on_resume)
        toolbar.addWidget(self.resume_btn)

        self.stop_btn = QPushButton("\u25a0 Stop")
        self.stop_btn.clicked.connect(self._on_stop)
        toolbar.addWidget(self.stop_btn)

        self.reset_btn = QPushButton("\u21ba Reset")
        self.reset_btn.clicked.connect(self._on_reset)
        toolbar.addWidget(self.reset_btn)

        self.replay_btn = QPushButton("\u21b7 Replay")
        self.replay_btn.clicked.connect(self._on_replay)
        toolbar.addWidget(self.replay_btn)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("  Speed"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setFixedWidth(90)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(50)
        self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(
            lambda v: self.dashboard_page.set_playback_speed(v / 10.0)
        )
        toolbar.addWidget(self.speed_slider)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("  Zoom"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setFixedWidth(90)
        self.zoom_slider.setMinimum(5)
        self.zoom_slider.setMaximum(20)
        self.zoom_slider.setValue(10)
        self.zoom_slider.valueChanged.connect(
            lambda v: self.dashboard_page.topology_view.set_zoom(v / 10.0)
        )
        toolbar.addWidget(self.zoom_slider)

        toolbar.addSeparator()

        theme_btn = QPushButton("\u25d1 Theme")
        theme_btn.clicked.connect(self._toggle_theme)
        toolbar.addWidget(theme_btn)

        export_btn = QPushButton("\u2913 Export")
        export_btn.clicked.connect(self._on_export)
        toolbar.addWidget(export_btn)

        save_btn = QPushButton("\U0001f4be Save")
        save_btn.clicked.connect(self._on_save)
        toolbar.addWidget(save_btn)

        load_btn = QPushButton("\U0001f4c2 Load")
        load_btn.clicked.connect(self._on_load)
        toolbar.addWidget(load_btn)

    # ------------------------------------------------------------------ #
    # Simulation control
    # ------------------------------------------------------------------ #

    def _set_running_state(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.reset_btn.setEnabled(not running)

    def _on_start(self) -> None:
        params = self.dashboard_page.read_parameters()
        self.dashboard_page.reset_view()
        self.dashboard_page.topology_view.set_link_loss(total_channel_loss_db(params))
        engine = {"timebin": "fast time-bin engine", "event": "SeQUeNCe per-photon engine",
                  "continuous": "continuous-train comparison engine"}.get(params.engine, "")
        self.statusBar().showMessage(f"Running {params.num_pulses:,} pulses on the {engine}\u2026")
        self._set_running_state(True)

        self._thread, self._worker = start_simulation(
            params, self._on_simulation_finished, self._on_simulation_failed
        )

    def _on_simulation_finished(self, result: SimulationResult) -> None:
        self._last_result = result
        self._set_running_state(False)
        self.dashboard_page.load_result(result)
        self.analytics_page.display_result(result)
        self.cascade_page.notify_new_result(result)
        self.dashboard_page.start_replay()
        self.statusBar().showMessage(
            f"Done \u2014 {result.pulses_sent:,} pulses ({result.duration_s * 1e3:.2f} ms), "
            f"{result.sifted_key_bits:,} sifted bits at {result.throughput_bps / 1e3:.2f} kbps, "
            f"QBER {result.qber * 100:.2f}% (expected {result.expected_qber * 100:.2f}%), "
            f"{result.security_message}"
        )

    def _on_simulation_failed(self, message: str) -> None:
        self._set_running_state(False)
        self.statusBar().showMessage("Simulation failed.")
        QMessageBox.critical(self, "Backend Error", f"The DPS-QKD backend raised an error:\n\n{message}")

    def _on_pause(self) -> None:
        self.dashboard_page.pause_replay()
        self.statusBar().showMessage("Paused.")

    def _on_resume(self) -> None:
        self.dashboard_page.resume_replay()
        self.statusBar().showMessage("Resumed.")

    def _on_stop(self) -> None:
        self.dashboard_page.stop_replay()
        self.statusBar().showMessage("Stopped.")

    def _on_reset(self) -> None:
        self.dashboard_page.reset_view()
        self._last_result = None
        self.statusBar().showMessage("Ready.")

    def _on_replay(self) -> None:
        self.dashboard_page.start_replay()
        self.statusBar().showMessage("Replaying recorded run.")

    def _on_restore_defaults(self) -> None:
        self.dashboard_page.restore_defaults()
        self.statusBar().showMessage("Simulation parameters restored to factory defaults.")

    def _on_default_speed_changed(self, value: int) -> None:
        self.speed_slider.setValue(value)

    def _on_default_zoom_changed(self, value: int) -> None:
        self.zoom_slider.setValue(value)

    # ------------------------------------------------------------------ #
    # Theme
    # ------------------------------------------------------------------ #

    def _toggle_theme(self) -> None:
        self.apply_theme("light" if self._theme_name == "dark" else "dark")

    def apply_theme(self, name: str) -> None:
        self._theme_name = name
        palette = DARK if name == "dark" else LIGHT
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().setStyleSheet(build_stylesheet(palette))
        self.dashboard_page.topology_view.apply_palette(palette)
        self.settings_page.set_theme_combo(name)

    # ------------------------------------------------------------------ #
    # Export / Save / Load
    # ------------------------------------------------------------------ #

    def _on_export(self) -> None:
        if self._last_result is None:
            QMessageBox.information(self, "Export", "Run a simulation first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "dps_qkd_results.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        def blank(value):
            return "" if value is None else value

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"# {self._last_result.model}"])
            writer.writerow(["pulse", "phases", "click_kind", "click_source", "port_window",
                             "slot", "time_ns", "alice_bit", "bob_bit", "key_bits",
                             "eve_intercepted"])
            for e in self._last_result.events:
                writer.writerow([
                    e.index, "".join(str(b) for b in e.phases), e.kind, e.source,
                    blank(e.detector), blank(e.slot), f"{e.time_ns:.4f}" if e.kind else "",
                    blank(e.alice_bit), blank(e.bob_bit), e.n_key, e.eve_intercepted,
                ])
        self.statusBar().showMessage(f"Exported {len(self._last_result.events)} events to {path}")

    def _on_save(self) -> None:
        params = self.dashboard_page.read_parameters()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "dps_qkd_config.json", "JSON Files (*.json)"
        )
        if not path:
            return
        with open(path, "w") as f:
            json.dump(asdict(params), f, indent=2)
        self.statusBar().showMessage(f"Configuration saved to {path}")

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "", "JSON Files (*.json)"
        )
        if not path:
            return
        with open(path) as f:
            data = json.load(f)
        try:
            params, notes = params_from_dict(data)
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Load Configuration", f"Could not read {path}:\n\n{exc}")
            return
        self.dashboard_page.apply_parameters(params)
        message = f"Configuration loaded from {path}"
        if notes:
            message += " \u2014 " + "; ".join(notes)
        self.statusBar().showMessage(message)

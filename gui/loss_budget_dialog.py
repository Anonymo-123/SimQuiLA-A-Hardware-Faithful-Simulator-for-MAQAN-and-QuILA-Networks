"""Dialog for entering a link's loss as an itemised budget (roadmap item 3).

The user enters what is known about the link -- fibre length and
attenuation, splices, connectors, add-drop multiplexers and other
components, and optionally the measured end-to-end loss. The dialog shows
the itemised total and, when a measurement is given, how much of it the
items do not explain.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from quila_sim.core import LossBudget


def _dspin(value: float, maximum: float, step: float, decimals: int, suffix: str) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(0.0, maximum)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    box.setSuffix(suffix)
    return box


class LossBudgetDialog(QDialog):
    """Edit a :class:`LossBudget`; ``budget()`` returns the result."""

    def __init__(self, length_km: float, initial: LossBudget | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Link loss budget")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)

        intro = QLabel(
            "Field links lose most of their light at splices, connectors and "
            "add-drop multiplexers, not in the fibre itself. If you have a "
            "measured end-to-end loss (OTDR or insertion-loss meter), enter it: "
            "it is used as the link loss, and the items below explain it."
        )
        intro.setWordWrap(True)
        intro.setProperty("class", "dim")
        layout.addWidget(intro)

        fibre_box = QGroupBox("Fibre")
        fibre_form = QFormLayout(fibre_box)
        self.length = _dspin(length_km, 2000.0, 0.1, 2, " km")
        self.attenuation = _dspin(0.20, 2.0, 0.01, 3, " dB/km")
        fibre_form.addRow("Length", self.length)
        fibre_form.addRow("Attenuation", self.attenuation)
        layout.addWidget(fibre_box)

        events_box = QGroupBox("Splices, connectors and components")
        events_form = QFormLayout(events_box)
        self.splices = QSpinBox()
        self.splices.setRange(0, 1000)
        self.splice_loss = _dspin(0.10, 3.0, 0.01, 2, " dB each")
        self.connectors = QSpinBox()
        self.connectors.setRange(0, 1000)
        self.connector_loss = _dspin(0.30, 3.0, 0.05, 2, " dB each")
        self.adm = _dspin(0.0, 30.0, 0.1, 2, " dB")
        self.other = _dspin(0.0, 30.0, 0.1, 2, " dB")
        events_form.addRow("Splices", self.splices)
        events_form.addRow("  loss per splice", self.splice_loss)
        events_form.addRow("Connectors", self.connectors)
        events_form.addRow("  loss per connector", self.connector_loss)
        events_form.addRow("Add-drop multiplexers / filters", self.adm)
        events_form.addRow("Other losses", self.other)
        layout.addWidget(events_box)

        measured_box = QGroupBox("Measurement")
        measured_form = QFormLayout(measured_box)
        self.use_measured = QCheckBox("I have a measured end-to-end loss")
        self.measured = _dspin(0.0, 60.0, 0.1, 2, " dB")
        measured_form.addRow(self.use_measured)
        measured_form.addRow("Measured loss", self.measured)
        layout.addWidget(measured_box)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if initial is not None:
            self._load(initial)
        for w in (self.length, self.attenuation, self.splice_loss, self.connector_loss,
                  self.adm, self.other, self.measured):
            w.valueChanged.connect(self._refresh)
        for w in (self.splices, self.connectors):
            w.valueChanged.connect(self._refresh)
        self.use_measured.toggled.connect(self._refresh)
        self._refresh()

    def _load(self, budget: LossBudget) -> None:
        if budget.fibre_length_km:
            self.length.setValue(budget.fibre_length_km)
        self.attenuation.setValue(budget.fibre_attenuation_db_per_km)
        kinds = budget.by_kind()
        n_splices = sum(1 for e in budget.events if e.kind == "splice")
        n_connectors = sum(1 for e in budget.events if e.kind == "connector")
        self.splices.setValue(n_splices)
        if n_splices:
            self.splice_loss.setValue(kinds["splice"] / n_splices)
        self.connectors.setValue(n_connectors)
        if n_connectors:
            self.connector_loss.setValue(kinds["connector"] / n_connectors)
        self.adm.setValue(sum(v for k, v in kinds.items() if k in ("adm", "wdm", "filter")))
        self.other.setValue(sum(v for k, v in kinds.items()
                                if k in ("bend", "patch-panel", "other")))
        if budget.measured_total_db is not None:
            self.use_measured.setChecked(True)
            self.measured.setValue(budget.measured_total_db)

    def budget(self) -> LossBudget:
        """The budget currently described by the dialog."""
        components = {}
        if self.adm.value():
            components["adm"] = self.adm.value()
        if self.other.value():
            components["other"] = self.other.value()
        return LossBudget.from_counts(
            self.length.value(),
            self.attenuation.value(),
            splices=self.splices.value(),
            splice_loss_db=self.splice_loss.value(),
            connectors=self.connectors.value(),
            connector_loss_db=self.connector_loss.value(),
            components=components,
            measured_total_db=self.measured.value() if self.use_measured.isChecked() else None,
            source="entered in the loss budget dialog",
        )

    def _refresh(self) -> None:
        self.measured.setEnabled(self.use_measured.isChecked())
        b = self.budget()
        text = f"<b>Link loss used: {b.total_db:.2f} dB</b><br>Itemised: {b.itemised_db:.2f} dB"
        if b.unaccounted_db is not None:
            text += f"<br>Not explained by the items: {b.unaccounted_db:+.2f} dB"
        self.summary.setText(text)

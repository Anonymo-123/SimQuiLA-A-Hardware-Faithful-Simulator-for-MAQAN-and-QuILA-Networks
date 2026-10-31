"""Small reusable widgets shared across pages."""

from __future__ import annotations

from typing import Callable, List, Optional

from PyQt6.QtCore import QPropertyAnimation, QRectF, Qt, pyqtProperty, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


# Five-colour, professional event-log palette (blue / green / orange / red / gray).
LOG_COLOR_BLUE = "#3b82f6"    # other
LOG_COLOR_GREEN = "#22c55e"   # key bit
LOG_COLOR_ORANGE = "#f59e0b"  # Eve intercepted
LOG_COLOR_RED = "#ef4444"     # key bit with an error, or re-sent by Eve
LOG_COLOR_GRAY = "#8a93a3"    # click without a key bit, or no click


class ToggleSwitch(QWidget):
    """A modern pill-shaped on/off switch, e.g. for 'Eve on channel'."""

    toggled = pyqtSignal(bool)

    def __init__(self, checked: bool = False, on_color: str = "#3b82f6",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._checked = checked
        self._on_color = QColor(on_color)
        self._off_color = QColor("#2b3340")
        self._knob_pos = 1.0 if checked else 0.0
        self.setFixedSize(42, 22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._anim = QPropertyAnimation(self, b"knobPos", self)
        self._anim.setDuration(140)

    def _get_knob_pos(self) -> float:
        return self._knob_pos

    def _set_knob_pos(self, value: float) -> None:
        self._knob_pos = value
        self.update()

    knobPos = pyqtProperty(float, _get_knob_pos, _set_knob_pos)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool) -> None:
        if checked == self._checked:
            return
        self._checked = checked
        self._anim.stop()
        self._anim.setStartValue(self._knob_pos)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()
        self.toggled.emit(checked)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.MouseButton.LeftButton:
            self.setChecked(not self._checked)
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        track_color = QColor(self._off_color)
        track_color.setRed(int(self._off_color.red() + (self._on_color.red() - self._off_color.red()) * self._knob_pos))
        track_color.setGreen(int(self._off_color.green() + (self._on_color.green() - self._off_color.green()) * self._knob_pos))
        track_color.setBlue(int(self._off_color.blue() + (self._on_color.blue() - self._off_color.blue()) * self._knob_pos))
        rect = QRectF(0, 0, self.width(), self.height())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track_color)
        painter.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)

        knob_d = self.height() - 4
        x = 2 + self._knob_pos * (self.width() - knob_d - 4)
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(QRectF(x, 2, knob_d, knob_d))


class StatCard(QFrame):
    """One tile in the STATISTICS grid: small uppercase label over a big value."""

    def __init__(self, label: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("StatCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        self._label = QLabel(label.upper())
        self._label.setObjectName("StatCardLabel")
        self._value = QLabel("\u2014")
        self._value.setObjectName("StatCardValue")
        layout.addWidget(self._label)
        layout.addWidget(self._value)
        layout.addStretch(1)
        self.setMinimumHeight(64)

    def set_value(self, text: str, color: Optional[str] = None) -> None:
        self._value.setText(text)
        self._value.setStyleSheet(f"color: {color};" if color else "")


class StatRow(QWidget):
    """One line in the compact 'Live Statistics' list: label left, value right.

    Much shorter than StatCard -- no box/border, just a single text row --
    so ten of these plus a security line take roughly the vertical space of
    two StatCard tiles, leaving the rest of the panel for the event log.

    Has a fixed height so a crowded panel degrades by scrolling (handled by
    the parent QScrollArea) rather than by squeezing rows until the text
    clips or overlaps.
    """

    ROW_HEIGHT = 24

    def __init__(self, label: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(self.ROW_HEIGHT)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(8)

        self._label = QLabel(label + ":")
        self._label.setObjectName("StatRowLabel")
        self._label.setProperty("class", "dim")

        self._value = QLabel("\u2014")
        self._value.setObjectName("StatRowValue")
        font = self._value.font()
        font.setBold(True)
        self._value.setFont(font)
        self._value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(self._label)
        layout.addStretch(1)
        layout.addWidget(self._value)

    def set_value(self, text: str, color: Optional[str] = None) -> None:
        self._value.setText(text)
        self._value.setStyleSheet(f"color: {color};" if color else "")


class KeyStreamWidget(QScrollArea):
    """Horizontally-scrolling row of fixed-width bit cells.

    Fixes the classic "scrollbar covers the last bits" problem by giving the
    scrollbar its own reserved strip below the cell row instead of letting it
    float on top of the content. Auto-scrolls to the newest bit as they are
    appended, unless the user has manually scrolled away from the right edge
    (in which case we don't yank the view back while they're reading).
    """

    CELL_W = 22
    CELL_H = 26

    def __init__(self, accent: str = "#3b82f6", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._accent = accent
        self._bits: List[str] = []
        self._user_scrolled_away = False

        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Reserve a fixed height: cell row + scrollbar strip, so the
        # scrollbar never overlaps the bit cells.
        self.setFixedHeight(self.CELL_H + 16)
        self.setStyleSheet(
            f"""
            QScrollArea {{ background: transparent; border: none; }}
            QScrollBar:horizontal {{
                height: 10px;
                background: transparent;
                margin: 4px 0 0 0;
            }}
            QScrollBar::handle:horizontal {{
                background: #3a4250;
                border-radius: 4px;
                min-width: 24px;
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
            """
        )

        self._container = QWidget()
        self._row = QHBoxLayout(self._container)
        self._row.setContentsMargins(2, 0, 2, 0)
        self._row.setSpacing(2)
        self._row.addStretch(1)  # cells are inserted before this stretch
        self.setWidget(self._container)

        self.horizontalScrollBar().valueChanged.connect(self._on_scroll)

    def _on_scroll(self, value: int) -> None:
        bar = self.horizontalScrollBar()
        self._user_scrolled_away = value < bar.maximum() - 4

    def clear_bits(self) -> None:
        self._bits = []
        self._user_scrolled_away = False
        while self._row.count() > 1:
            item = self._row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def append_bit(self, bit: int, mismatch: bool = False) -> None:
        self._bits.append(str(bit))
        cell = QLabel(str(bit))
        cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cell.setFixedSize(self.CELL_W, self.CELL_H)
        bg = "#ef4444" if mismatch else "#1d232d"
        border = "#ef4444" if mismatch else "#2b3340"
        cell.setStyleSheet(
            f"""
            background-color: {bg};
            border: 1px solid {border};
            border-radius: 3px;
            font-family: 'Consolas', 'Cascadia Mono', monospace;
            font-weight: 600;
            color: {'#ffffff' if mismatch else self._accent};
            """
        )
        self._row.insertWidget(self._row.count() - 1, cell)
        if not self._user_scrolled_away:
            self._scroll_to_end()

    def append_bits(self, bits: List[int], mismatches: Optional[List[bool]] = None) -> None:
        self.setUpdatesEnabled(False)
        for i, b in enumerate(bits):
            m = bool(mismatches[i]) if mismatches else False
            self.append_bit(b, m)
        self.setUpdatesEnabled(True)

    def _scroll_to_end(self) -> None:
        bar = self.horizontalScrollBar()
        bar.setValue(bar.maximum())

    def bit_count(self) -> int:
        return len(self._bits)


class EventLogTable(QTableWidget):
    """Structured event log: one row per pulse that produced a detector click.

    Columns: Pulse # | Phases | Click | Event
    (Phases = Alice's phase per time bin; Click = "P<port window> s<slot>
    <time stamp within the pulse period>", e.g. "P1 s2 12.31ns".) Five colours
    keep it scannable: green = key bit, red = key bit with an error or from
    Eve, orange = Eve intercepted, gray = click without a key bit, blue = other.
    """

    HEADERS = ["Pulse #", "Phases", "Click", "Event"]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(0, len(self.HEADERS), parent)
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setShowGrid(False)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(22)
        header = self.horizontalHeader()
        for col in range(3):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._max_rows = 5000  # hard cap so long traces stay smooth; oldest trimmed

    def clear_log(self) -> None:
        self.setRowCount(0)

    @staticmethod
    def classify(evt) -> tuple[str, str]:
        """Return (event_text, color) for one recorded pulse (a PulseEvent)."""
        if not evt.clicked:
            if evt.eve_intercepted:
                return "Eve intercepted, no click", LOG_COLOR_ORANGE
            return "No click", LOG_COLOR_GRAY
        if evt.kind == "key":
            error = evt.alice_bit is not None and evt.bob_bit is not None \
                and evt.alice_bit != evt.bob_bit
            if evt.eve_intercepted:
                return f"Eve resent \u2192 key bit {evt.bob_bit}", LOG_COLOR_RED
            origin = {"dark": " (dark count)", "leak": " (IM leakage)"}.get(evt.source, "")
            if error:
                return f"Key bit {evt.bob_bit} \u2717 error{origin}", LOG_COLOR_RED
            return f"Key bit {evt.bob_bit}{origin}", LOG_COLOR_GREEN
        labels = {
            "early": "Early slot \u2013 discarded",
            "late": "Late slot \u2013 discarded",
            "guard": "Guard band \u2013 discarded",
            "outside": "Outside port windows",
        }
        text = labels.get(evt.kind, evt.kind)
        if evt.source == "dark":
            text += " (dark count)"
        elif evt.source == "leak":
            text += " (IM leakage)"
        if evt.eve_intercepted:
            return text + ", Eve", LOG_COLOR_ORANGE
        return text, LOG_COLOR_GRAY

    def append_event(self, pulse_index: int, phases: str, click: str,
                     event_text: str, color: str) -> None:
        row = self.rowCount()
        self.insertRow(row)
        vals = [f"{pulse_index}", phases, click, event_text]
        last = len(vals) - 1
        for col, text in enumerate(vals):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter if col != last
                                  else Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            if col == last:
                item.setForeground(QColor(color))
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self.setItem(row, col, item)

        if self.rowCount() > self._max_rows:
            self.removeRow(0)
        self.scrollToBottom()

    def append_events_batch(self, rows: List[tuple]) -> None:
        """rows: list of (pulse_index, phases, click, event_text, color)."""
        self.setUpdatesEnabled(False)
        self.setSortingEnabled(False)
        for r in rows:
            self.append_event(*r)
        self.setUpdatesEnabled(True)


class LabeledSlider(QWidget):
    """A titled slider with a live numeric readout, e.g. 'Fiber Loss (dB/km) 0.20'."""

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        title: str,
        minimum: float,
        maximum: float,
        default: float,
        step: float = 1.0,
        decimals: int = 0,
        parent: Optional[QWidget] = None,
        formatter: Optional[Callable[[float], str]] = None,
    ) -> None:
        super().__init__(parent)
        self._decimals = decimals
        self._formatter = formatter
        self._minimum = minimum
        self._step = step
        self._scale = 1.0 / step if step else 1.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(2)

        header = QHBoxLayout()
        self._title_label = QLabel(title)
        self._value_label = QLabel(self._fmt(default))
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        header.addWidget(self._title_label)
        header.addStretch(1)
        header.addWidget(self._value_label)
        layout.addLayout(header)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        steps = max(1, round((maximum - minimum) / step))
        self._slider.setMinimum(0)
        self._slider.setMaximum(steps)
        self._slider.setValue(round((default - minimum) / step))
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider)

    def _fmt(self, value: float) -> str:
        if self._formatter is not None:
            return self._formatter(value)
        return f"{value:.{self._decimals}f}"

    def set_title(self, title: str) -> None:
        self._title_label.setText(title)

    def refresh_label(self) -> None:
        """Re-format the value read-out (e.g. when the formatter depends on other controls)."""
        self._value_label.setText(self._fmt(self.value()))

    def _on_slider_changed(self, raw: int) -> None:
        value = self._minimum + raw * self._step
        self._value_label.setText(self._fmt(value))
        self.valueChanged.emit(value)

    def value(self) -> float:
        return self._minimum + self._slider.value() * self._step

    def set_value(self, value: float) -> None:
        self._slider.setValue(round((value - self._minimum) / self._step))

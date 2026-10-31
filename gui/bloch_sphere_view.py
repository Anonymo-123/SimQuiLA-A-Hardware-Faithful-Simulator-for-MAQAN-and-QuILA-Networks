"""Bloch-sphere view of the detected key clicks (display only).

This is a *visualization* layer only; the simulated physics does not use it.
In the time-bin model each photon is spread over the N time bins of its
pulse (``sequence/components/dps_hardware.py``, ``TimeBinPhoton``); a key
click measures the phase difference of two neighbouring bins, which behaves
like a qubit measured on the z axis. So ``backend/adapter.py`` places a key
click from a photon at z = +V (bit 0) or -V (bit 1), where V is the
interferometer visibility set by the user, and puts clicks without a key bit
(early/late slots, dark counts) at the origin. Points pull in toward the
centre as the visibility drops -- the geometric picture of lost coherence.

Rendered with plain QPainter (a simple perspective-projected wireframe
sphere + scatter points) so it has no extra GPU/OpenGL dependency and
stays smooth even for long traces.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget

Point3 = Tuple[float, float, float]


def _project(p: Point3, cx: float, cy: float, r: float, tilt: float = 0.35) -> QPointF:
    """Simple orthographic projection with a fixed tilt around the x-axis,
    so the sphere reads as 3D without a full 3D rendering stack."""
    x, y, z = p
    cos_t, sin_t = math.cos(tilt), math.sin(tilt)
    y2 = y * cos_t - z * sin_t
    z2 = y * sin_t + z * cos_t
    return QPointF(cx + x * r, cy - y2 * r), z2


class BlochSphereView(QWidget):
    """Accumulating scatter plot of pulse Bloch vectors on a wireframe sphere."""

    MAX_POINTS = 4000  # cap so a 20k-pulse run doesn't repaint forever

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(230)
        self._points: List[Tuple[Point3, QColor]] = []

    def clear_points(self) -> None:
        self._points = []
        self.update()

    def add_point(self, x: float, y: float, z: float, dark: bool, lost: bool) -> None:
        if lost:
            return  # nothing measured -- don't plot a phantom state
        if dark:
            color = QColor("#94a3b8")  # neutral grey: no real phase info
        elif z >= 0:
            color = QColor("#3b82f6")  # phase 0 / D0 -- matches legend
        else:
            color = QColor("#f39c12")  # phase pi / D1 -- matches legend
        if len(self._points) >= self.MAX_POINTS:
            self._points.pop(0)
        self._points.append(((x, y, z), color))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        r = max(10.0, min(w, h) / 2 - 24)

        # Sphere outline + equator + meridian for depth cues.
        pen = QPen(QColor("#334155"))
        pen.setWidthF(1.2)
        painter.setPen(pen)
        painter.drawEllipse(QRectF(cx - r, cy - r, 2 * r, 2 * r))

        steps = 48
        for ring in ("equator", "meridian"):
            pts = []
            for i in range(steps + 1):
                a = 2 * math.pi * i / steps
                if ring == "equator":
                    p3 = (math.cos(a), math.sin(a), 0.0)
                else:
                    p3 = (math.cos(a), 0.0, math.sin(a))
                pts.append(_project(p3, cx, cy, r)[0])
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])

        # Axis labels.
        painter.setPen(QPen(QColor("#64748b")))
        top, _ = _project((0, 0, 1), cx, cy, r)
        bot, _ = _project((0, 0, -1), cx, cy, r)
        painter.drawText(top + QPointF(-6, -8), "|0\u27e9")
        painter.drawText(bot + QPointF(-6, 16), "|1\u27e9")

        # Points, back-to-front by projected depth so it reads as 3D.
        projected = [(_project(p, cx, cy, r), color) for p, color in self._points]
        projected.sort(key=lambda t: t[0][1])
        for (pt, _depth), color in projected:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(pt, 2.6, 2.6)

        if not self._points:
            painter.setPen(QPen(QColor("#475569")))
            painter.drawText(
                QRectF(0, 0, w, h),
                Qt.AlignmentFlag.AlignCenter,
                "Bloch vectors will appear here once a run starts",
            )

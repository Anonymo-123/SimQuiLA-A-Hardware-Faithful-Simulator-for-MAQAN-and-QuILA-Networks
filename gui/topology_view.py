"""Animated Alice -> Fiber -> Bob topology canvas.

A research-grade *pseudo-3D* visualisation (perspective-shaded fibre tube,
glow/bloom via layered drop-shadows, and fading motion trails) built on
``QGraphicsView``/SVG-style 2D primitives rather than a full 3D engine --
this keeps rendering GPU-independent and reliably smooth at 60 FPS even
inside a plain desktop widget, while still giving the depth cues (tube
shading, highlight rail, drop shadows) that make it read as "3D-inspired"
rather than flat.

Rendering is done with a small, reused set of ``QGraphicsItem`` objects
(nodes, detector badges, fibre tube, Eve badge, and a pool of photon dots
+ trail dots) so that during a live run the canvas never rebuilds the
scene -- only item positions/colours/opacity change, keeping the
animation smooth even for large pulse counts.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPen,
    QRadialGradient,
)
from PyQt6.QtWidgets import (
    QGraphicsDropShadowEffect,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from gui.theme import Palette

PHASE0_COLOR = QColor("#3fa7ff")
PHASE1_COLOR = QColor("#ffb020")
KEYLESS_COLOR = QColor("#94a3b8")
CLICK_COLOR = QColor("#33e08a")
EVE_COLOR = QColor("#ff4d4d")
TRAIL_LENGTH = 6


def _glow(color: QColor, radius: float = 18.0) -> QGraphicsDropShadowEffect:
    """A soft coloured glow used throughout for a "bloom" look."""
    effect = QGraphicsDropShadowEffect()
    effect.setColor(color)
    effect.setOffset(0, 0)
    effect.setBlurRadius(radius)
    return effect


@dataclass
class _Photon:
    item: QGraphicsEllipseItem
    trail: List[QGraphicsEllipseItem]
    history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=TRAIL_LENGTH))
    progress: float = 1.1  # >1 means "at rest, hidden"
    speed: float = 0.06
    phase: int = 0
    intercepted: bool = False


class TopologyView(QGraphicsView):
    """Alice/Bob/fibre canvas with a small pool of animated, glowing photons."""

    photon_arrived = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setRenderHint(self.renderHints().Antialiasing)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(self.Shape.NoFrame)

        self._scene = QGraphicsScene(0, 0, 900, 380, self)
        self.setScene(self._scene)

        self._alice_x = 100.0
        self._bob_x = 770.0
        self._mid_y = 150.0

        self._attenuation_frac = 0.0  # 0..1, visual only (drives tube fade)

        self._build_static_items()
        self._photon_pool: List[_Photon] = [self._make_photon() for _ in range(24)]
        self._active_photons: List[_Photon] = []

        self._pulse_phase = 0.0
        self._zoom = 1.0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)  # ~60 FPS

        self._eve_visible = False
        self._alert_active = False

    # ------------------------------------------------------------------ #
    # Static scene construction
    # ------------------------------------------------------------------ #

    def _build_static_items(self) -> None:
        y = self._mid_y
        x0, x1 = self._alice_x + 48, self._bob_x - 48

        # Pseudo-3D fibre "tube": a shaded rounded rect with a top
        # highlight rail, giving a cylindrical/glass impression without a
        # real 3D renderer.
        self._tube = QGraphicsRectItem(QRectF(x0, y - 7, x1 - x0, 14))
        self._tube.setPen(QPen(Qt.PenStyle.NoPen))
        self._scene.addItem(self._tube)

        self._tube_highlight = QGraphicsRectItem(QRectF(x0, y - 5, x1 - x0, 3))
        self._tube_highlight.setPen(QPen(Qt.PenStyle.NoPen))
        self._tube_highlight.setBrush(QBrush(QColor(255, 255, 255, 40)))
        self._scene.addItem(self._tube_highlight)

        self._update_tube_gradient(QColor("#3a4250"))

        self._alice_node = self._make_node(self._alice_x, y, "Alice", "#3fa7ff")
        self._bob_node = self._make_node(self._bob_x, y, "Bob", "#33e08a")

        self._alice_label = self._make_caption(self._alice_x, y + 62, "IM + PM: 3-bin pulses")
        self._bob_label = self._make_caption(self._bob_x, y + 62, "DLI \u2192 1 SPD (time-multiplexed)")

        self._badge_text = {}
        self._d0_badge = self._make_badge(self._bob_x + 46, y - 48, "P0", key=0)
        self._d1_badge = self._make_badge(self._bob_x + 46, y + 22, "P1", key=1)

        self._eve_node = self._make_node(
            (self._alice_x + self._bob_x) / 2, y - 100, "Eve", "#ff4d4d"
        )
        self._eve_node.setOpacity(0.0)
        self._eve_label = self._make_caption(
            (self._alice_x + self._bob_x) / 2, y - 50, "Intercept-Resend"
        )
        self._eve_label.setOpacity(0.0)

        self._alert_text = QGraphicsSimpleTextItem("\u26a0 EAVESDROPPING DETECTED")
        self._alert_text.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self._alert_text.setBrush(QBrush(EVE_COLOR))
        tw = self._alert_text.boundingRect().width()
        self._alert_text.setPos((self._alice_x + self._bob_x) / 2 - tw / 2, 20)
        self._alert_text.setOpacity(0.0)
        self._scene.addItem(self._alert_text)

    def _update_tube_gradient(self, base_color: QColor) -> None:
        """Recolour the fibre tube to show cylindrical shading plus a
        distance-loss fade toward Bob (visual proxy for attenuation)."""
        x0, x1 = self._alice_x + 48, self._bob_x - 48
        grad = QLinearGradient(x0, 0, x1, 0)
        light = base_color.lighter(160)
        dark = base_color.darker(140)
        end_alpha = max(40, int(255 * (1.0 - 0.6 * self._attenuation_frac)))
        light.setAlpha(end_alpha)
        dark.setAlpha(end_alpha)
        grad.setColorAt(0.0, light)
        grad.setColorAt(0.5, base_color)
        grad.setColorAt(1.0, dark)
        self._tube.setBrush(QBrush(grad))

    def _make_node(self, x: float, y: float, label: str, color: str) -> QGraphicsEllipseItem:
        r = 42
        ellipse = QGraphicsEllipseItem(QRectF(-r, -r, 2 * r, 2 * r))
        ellipse.setPos(x, y)
        grad = QRadialGradient(0, -r * 0.3, r * 1.6)
        grad.setColorAt(0.0, QColor(color).lighter(160))
        grad.setColorAt(1.0, QColor(color))
        ellipse.setBrush(QBrush(grad))
        ellipse.setPen(QPen(QColor(color).darker(130), 2))
        ellipse.setGraphicsEffect(_glow(QColor(color), 24))
        self._scene.addItem(ellipse)

        text = QGraphicsSimpleTextItem(label, ellipse)
        font = QFont("Segoe UI", 11, QFont.Weight.DemiBold)
        text.setFont(font)
        text.setBrush(QBrush(QColor("#ffffff")))
        tw = text.boundingRect().width()
        th = text.boundingRect().height()
        text.setPos(-tw / 2, -th / 2)
        return ellipse

    def _make_caption(self, x: float, y: float, text: str) -> QGraphicsSimpleTextItem:
        item = QGraphicsSimpleTextItem(text)
        item.setBrush(QBrush(QColor("#8a93a3")))
        item.setFont(QFont("Segoe UI", 9))
        tw = item.boundingRect().width()
        item.setPos(x - tw / 2, y)
        self._scene.addItem(item)
        return item

    def _make_badge(self, x: float, y: float, label: str, key: int = -1) -> QGraphicsEllipseItem:
        r = 17
        ellipse = QGraphicsEllipseItem(QRectF(-r, -r, 2 * r, 2 * r))
        ellipse.setPos(x, y)
        ellipse.setBrush(QBrush(QColor("#1d232d")))
        ellipse.setPen(QPen(QColor("#3a4250"), 1.5))
        self._scene.addItem(ellipse)
        text = QGraphicsSimpleTextItem(label, ellipse)
        text.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        text.setBrush(QBrush(QColor("#e6e9ef")))
        tw = text.boundingRect().width()
        th = text.boundingRect().height()
        text.setPos(-tw / 2, -th / 2)
        if key >= 0:
            self._badge_text[key] = text
        return ellipse

    def _set_caption(self, item: QGraphicsSimpleTextItem, x: float, text: str) -> None:
        item.setText(text)
        item.setPos(x - item.boundingRect().width() / 2, item.pos().y())

    def _set_badge_label(self, key: int, label: str) -> None:
        text = self._badge_text.get(key)
        if text is None:
            return
        text.setText(label)
        text.setPos(-text.boundingRect().width() / 2, -text.boundingRect().height() / 2)

    def _make_photon(self) -> _Photon:
        r = 6
        item = QGraphicsEllipseItem(QRectF(-r, -r, 2 * r, 2 * r))
        item.setBrush(QBrush(PHASE0_COLOR))
        item.setPen(QPen(Qt.PenStyle.NoPen))
        item.setOpacity(0.0)
        item.setZValue(6)
        item.setGraphicsEffect(_glow(PHASE0_COLOR, 16))
        self._scene.addItem(item)

        trail = []
        for i in range(TRAIL_LENGTH):
            tr = r * (1.0 - i / TRAIL_LENGTH) * 0.8
            t_item = QGraphicsEllipseItem(QRectF(-tr, -tr, 2 * tr, 2 * tr))
            t_item.setPen(QPen(Qt.PenStyle.NoPen))
            t_item.setBrush(QBrush(PHASE0_COLOR))
            t_item.setOpacity(0.0)
            t_item.setZValue(4)
            self._scene.addItem(t_item)
            trail.append(t_item)

        return _Photon(item=item, trail=trail)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def set_eve_visible(self, visible: bool) -> None:
        self._eve_visible = visible
        self._eve_node.setOpacity(1.0 if visible else 0.0)
        self._eve_label.setOpacity(1.0 if visible else 0.0)
        self.set_alert(visible)

    def set_alert(self, active: bool) -> None:
        self._alert_active = active
        if not active:
            self._alert_text.setOpacity(0.0)

    def set_attenuation(self, fiber_loss_db_km: float, distance_km: float) -> None:
        """Visual-only proxy for total link loss; recolours the fibre tube
        so longer/lossier links look visibly dimmer toward Bob."""
        self.set_link_loss(fiber_loss_db_km * distance_km)

    def set_link_loss(self, total_db: float) -> None:
        """Visual-only: dim the fibre tube toward Bob for a lossier link."""
        self._attenuation_frac = max(0.0, min(1.0, total_db / 30.0))
        self._update_tube_gradient(QColor("#3a4250"))

    def set_receiver_layout(self, layout: str, continuous: bool = False) -> None:
        """Label Alice and Bob for the simulated hardware.

        "tdm": one detector; the two badges are the port-0 and port-1 time
        windows (P0 / P1). "dual": one detector per port (D0 / D1).
        ``continuous``: the Model B comparison (gap-free pulse train).
        """
        if continuous:
            self._set_caption(self._alice_label, self._alice_x, "Phase encoder (continuous train)")
            self._set_caption(self._bob_label, self._bob_x, "DLI + D0/D1 (Model B)")
            labels = ("D0", "D1")
        elif layout == "dual":
            self._set_caption(self._alice_label, self._alice_x, "IM + PM: 3-bin pulses")
            self._set_caption(self._bob_label, self._bob_x, "DLI \u2192 2 SPDs (one per port)")
            labels = ("D0", "D1")
        else:
            self._set_caption(self._alice_label, self._alice_x, "IM + PM: 3-bin pulses")
            self._set_caption(self._bob_label, self._bob_x, "DLI \u2192 1 SPD (time-multiplexed)")
            labels = ("P0", "P1")
        self._set_badge_label(0, labels[0])
        self._set_badge_label(1, labels[1])

    def flash_detector(self, detector_id: Optional[int]) -> None:
        badge = self._d0_badge if detector_id == 0 else self._d1_badge if detector_id == 1 else None
        if badge is None:
            return
        badge.setBrush(QBrush(CLICK_COLOR))
        badge.setGraphicsEffect(_glow(CLICK_COLOR, 20))
        QTimer.singleShot(140, lambda: (
            badge.setBrush(QBrush(QColor("#1d232d"))),
            badge.setGraphicsEffect(None),
        ))

    def launch_photon(self, phase: int, intercepted: bool, speed: float = 0.05,
                      keyless: bool = False) -> None:
        """Send one animated, glowing photon from Alice to Bob.

        Colour: Eve red, otherwise by key bit; gray (``keyless``) for a click
        that carries no key bit (early/late slot, guard band, outside windows).
        """
        photon = next((p for p in self._photon_pool if p.progress >= 1.0), None)
        if photon is None:
            photon = self._photon_pool[0]
        photon.progress = 0.0
        photon.speed = speed
        photon.phase = phase
        photon.intercepted = intercepted
        photon.history.clear()

        if intercepted:
            color = EVE_COLOR
        elif keyless:
            color = KEYLESS_COLOR
        else:
            color = PHASE1_COLOR if phase else PHASE0_COLOR
        photon.item.setBrush(QBrush(color))
        photon.item.setOpacity(0.95)
        photon.item.setGraphicsEffect(_glow(color, 20 if intercepted else 14))
        for t in photon.trail:
            t.setBrush(QBrush(color))
        if photon not in self._active_photons:
            self._active_photons.append(photon)
        if intercepted:
            self._flash_alert()

    def _flash_alert(self) -> None:
        if not self._eve_visible:
            return
        self._alert_text.setOpacity(1.0)
        QTimer.singleShot(220, lambda: self._alert_text.setOpacity(0.0) if self._alert_active else None)

    def reset(self) -> None:
        for p in self._photon_pool:
            p.progress = 1.1
            p.item.setOpacity(0.0)
            for t in p.trail:
                t.setOpacity(0.0)
            p.history.clear()
        self._active_photons.clear()
        self.set_eve_visible(False)
        self.set_attenuation(0.0, 0.0)

    def set_zoom(self, factor: float) -> None:
        self._zoom = factor
        self.resetTransform()
        self.scale(factor, factor)

    # ------------------------------------------------------------------ #
    # Animation tick (~60 FPS)
    # ------------------------------------------------------------------ #

    def _tick(self) -> None:
        x0 = self._alice_x + 48
        x1 = self._bob_x - 48
        y = self._mid_y
        self._pulse_phase += 0.12

        still_active = []
        for p in self._active_photons:
            p.progress += p.speed
            if p.progress >= 1.0:
                p.item.setOpacity(0.0)
                for t in p.trail:
                    t.setOpacity(0.0)
                self.photon_arrived.emit()
                continue

            x = x0 + (x1 - x0) * p.progress
            wobble = 5.0 if p.intercepted else 0.0
            yy = y + wobble * math.sin(p.progress * 20 + self._pulse_phase)

            # Attenuation visual: fade slightly as the photon travels
            # further into a lossy link.
            fade = 1.0 - 0.35 * self._attenuation_frac * p.progress
            p.item.setPos(x, yy)
            p.item.setOpacity(max(0.25, 0.95 * fade))

            p.history.appendleft((x, yy))
            for i, t_item in enumerate(p.trail):
                if i < len(p.history):
                    hx, hy = p.history[i]
                    t_item.setPos(hx, hy)
                    t_item.setOpacity(max(0.0, (1.0 - i / TRAIL_LENGTH) * 0.5 * fade))
                else:
                    t_item.setOpacity(0.0)

            still_active.append(p)
        self._active_photons = still_active

        # Subtle Eve "alert" pulse ring while active.
        if self._eve_visible:
            pulse = 1.0 + 0.06 * math.sin(self._pulse_phase * 2)
            self._eve_node.setScale(pulse)

    def apply_palette(self, palette: Palette) -> None:
        self.setStyleSheet(f"background-color: {palette.window_bg}; border: none;")
        self._scene.setBackgroundBrush(QBrush(QColor(palette.window_bg)))
        for cap in (self._alice_label, self._bob_label, self._eve_label):
            cap.setBrush(QBrush(QColor(palette.text_dim)))

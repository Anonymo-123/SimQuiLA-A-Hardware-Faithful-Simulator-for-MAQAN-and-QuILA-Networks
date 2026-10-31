"""Application-wide colour palettes and QSS stylesheets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    window_bg: str
    panel_bg: str
    panel_border: str
    text: str
    text_dim: str
    accent: str
    accent_hover: str
    accent_text: str
    good: str
    bad: str
    warn: str
    grid: str
    log_bg: str
    input_bg: str


DARK = Palette(
    window_bg="#12161c",
    panel_bg="#171c24",
    panel_border="#252c37",
    text="#e6e9ef",
    text_dim="#8a93a3",
    accent="#3b82f6",
    accent_hover="#2f6fd6",
    accent_text="#ffffff",
    good="#2ecc71",
    bad="#e74c3c",
    warn="#f39c12",
    grid="#232a35",
    log_bg="#0d1117",
    input_bg="#1d232d",
)

LIGHT = Palette(
    window_bg="#f2f4f7",
    panel_bg="#ffffff",
    panel_border="#d9dee5",
    text="#1b2430",
    text_dim="#5c6773",
    accent="#2563eb",
    accent_hover="#1d4fc4",
    accent_text="#ffffff",
    good="#1f9d55",
    bad="#d0342c",
    warn="#c67c05",
    grid="#e2e6ec",
    log_bg="#f7f8fa",
    input_bg="#eef1f5",
)


def build_stylesheet(p: Palette) -> str:
    return f"""
    QWidget {{
        background-color: {p.window_bg};
        color: {p.text};
        font-family: 'Segoe UI', 'Inter', sans-serif;
        font-size: 12px;
    }}
    QMainWindow, QStackedWidget {{
        background-color: {p.window_bg};
    }}
    QFrame#Panel, QGroupBox {{
        background-color: {p.panel_bg};
        border: 1px solid {p.panel_border};
        border-radius: 4px;
    }}
    QGroupBox {{
        margin-top: 14px;
        padding-top: 10px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 4px;
        color: {p.text_dim};
    }}
    QLabel#SectionTitle {{
        font-weight: 700;
        font-size: 12px;
        color: {p.text};
        letter-spacing: 0.5px;
    }}
    QLabel.dim {{
        color: {p.text_dim};
    }}
    QToolBar {{
        background-color: {p.panel_bg};
        border-bottom: 1px solid {p.panel_border};
        spacing: 6px;
        padding: 4px;
    }}
    QPushButton {{
        background-color: {p.input_bg};
        border: 1px solid {p.panel_border};
        border-radius: 3px;
        padding: 5px 12px;
        color: {p.text};
    }}
    QPushButton:hover {{
        background-color: {p.accent};
        color: {p.accent_text};
        border-color: {p.accent};
    }}
    QPushButton:disabled {{
        color: {p.text_dim};
        background-color: {p.input_bg};
    }}
    QPushButton#PrimaryButton {{
        background-color: {p.accent};
        color: {p.accent_text};
        font-weight: 600;
        border: none;
    }}
    QPushButton#PrimaryButton:hover {{
        background-color: {p.accent_hover};
    }}
    QPushButton#NavButton {{
        text-align: left;
        padding: 10px 14px;
        border: none;
        border-radius: 0px;
        border-left: 3px solid transparent;
    }}
    QPushButton#NavButton:checked {{
        background-color: {p.input_bg};
        border-left: 3px solid {p.accent};
        font-weight: 600;
    }}
    QSlider::groove:horizontal {{
        height: 4px;
        background: {p.grid};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {p.accent};
        width: 13px;
        margin: -5px 0;
        border-radius: 6px;
    }}
    QPlainTextEdit, QTextEdit {{
        background-color: {p.log_bg};
        border: 1px solid {p.panel_border};
        color: {p.text};
        font-family: 'Consolas', 'Cascadia Mono', monospace;
        font-size: 11px;
    }}
    QTableWidget {{
        background-color: {p.panel_bg};
        alternate-background-color: {p.input_bg};
        gridline-color: {p.grid};
        border: 1px solid {p.panel_border};
    }}
    QHeaderView::section {{
        background-color: {p.input_bg};
        color: {p.text_dim};
        border: none;
        padding: 4px;
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
    }}
    QScrollBar::handle:vertical {{
        background: {p.panel_border};
        border-radius: 4px;
        min-height: 20px;
    }}
    QCheckBox::indicator {{
        width: 14px;
        height: 14px;
        border: 1px solid {p.panel_border};
        border-radius: 2px;
        background: {p.input_bg};
    }}
    QCheckBox::indicator:checked {{
        background: {p.accent};
        border-color: {p.accent};
    }}
    QComboBox {{
        background-color: {p.input_bg};
        border: 1px solid {p.panel_border};
        border-radius: 3px;
        padding: 3px 6px;
    }}
    QStatusBar {{
        background-color: {p.panel_bg};
        border-top: 1px solid {p.panel_border};
        color: {p.text_dim};
    }}
    """

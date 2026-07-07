"""Dark theme: TradingView-style palette, Fusion QPalette, and app-wide QSS.

Single source of truth for GUI colors — chart widgets import ``COLORS``.
"""
from __future__ import annotations

import logging

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

log = logging.getLogger(__name__)

COLORS = {
    "bg": "#131722",       # window / chart background
    "base": "#1e222d",     # input fields, table base, panels
    "text": "#d1d4dc",     # primary text
    "up": "#26a69a",       # bullish candles / positive P&L
    "down": "#ef5350",     # bearish candles / negative P&L
    "accent": "#2962ff",   # highlights, equity curve, buttons
    "warn": "#ff9800",     # warnings, soft-brake state
    "err": "#f44336",      # errors, kill-switch state
}

# derived shades kept private (not part of the contract)
_BORDER = "#2a2e39"
_HOVER = "#2a2e39"
_DISABLED = "#5d606b"
_SELECTION_TEXT = "#ffffff"


def _build_palette() -> QPalette:
    pal = QPalette()
    bg = QColor(COLORS["bg"])
    base = QColor(COLORS["base"])
    text = QColor(COLORS["text"])
    accent = QColor(COLORS["accent"])
    disabled = QColor(_DISABLED)

    pal.setColor(QPalette.ColorRole.Window, bg)
    pal.setColor(QPalette.ColorRole.WindowText, text)
    pal.setColor(QPalette.ColorRole.Base, base)
    pal.setColor(QPalette.ColorRole.AlternateBase, bg)
    pal.setColor(QPalette.ColorRole.ToolTipBase, base)
    pal.setColor(QPalette.ColorRole.ToolTipText, text)
    pal.setColor(QPalette.ColorRole.Text, text)
    pal.setColor(QPalette.ColorRole.PlaceholderText, disabled)
    pal.setColor(QPalette.ColorRole.Button, base)
    pal.setColor(QPalette.ColorRole.ButtonText, text)
    pal.setColor(QPalette.ColorRole.BrightText, QColor(COLORS["err"]))
    pal.setColor(QPalette.ColorRole.Link, accent)
    pal.setColor(QPalette.ColorRole.LinkVisited, accent)
    pal.setColor(QPalette.ColorRole.Highlight, accent)
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(_SELECTION_TEXT))
    pal.setColor(QPalette.ColorRole.Light, QColor(_BORDER))
    pal.setColor(QPalette.ColorRole.Midlight, QColor(_BORDER))
    pal.setColor(QPalette.ColorRole.Mid, QColor(_BORDER))
    pal.setColor(QPalette.ColorRole.Dark, bg)
    pal.setColor(QPalette.ColorRole.Shadow, QColor("#000000"))

    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, disabled)
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight,
                 QColor(_BORDER))
    return pal


def _build_qss() -> str:
    c = dict(COLORS, border=_BORDER, hover=_HOVER, disabled=_DISABLED,
             sel_text=_SELECTION_TEXT)
    return """
QWidget {{ background-color: {bg}; color: {text}; font-size: 12px; }}
QMainWindow, QDialog {{ background-color: {bg}; }}
QToolTip {{ background-color: {base}; color: {text};
    border: 1px solid {border}; padding: 3px; }}

/* --- tabs --- */
QTabWidget::pane {{ border: 1px solid {border}; top: -1px; }}
QTabBar::tab {{ background: {bg}; color: {text}; padding: 6px 16px;
    border: 1px solid {border}; border-bottom: none;
    border-top-left-radius: 3px; border-top-right-radius: 3px; }}
QTabBar::tab:selected {{ background: {base}; border-bottom: 2px solid {accent}; }}
QTabBar::tab:hover:!selected {{ background: {hover}; }}
QTabBar::tab:disabled {{ color: {disabled}; }}

/* --- tables --- */
QTableView, QTreeView, QListView {{ background-color: {base};
    alternate-background-color: {bg}; gridline-color: {border};
    border: 1px solid {border}; selection-background-color: {accent};
    selection-color: {sel_text}; }}
QHeaderView::section {{ background-color: {bg}; color: {text};
    border: none; border-right: 1px solid {border};
    border-bottom: 1px solid {border}; padding: 4px 8px; }}
QTableCornerButton::section {{ background-color: {bg};
    border: 1px solid {border}; }}

/* --- group boxes --- */
QGroupBox {{ border: 1px solid {border}; border-radius: 4px;
    margin-top: 10px; padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left;
    left: 8px; padding: 0 4px; color: {text}; }}

/* --- buttons --- */
QPushButton {{ background-color: {base}; color: {text};
    border: 1px solid {border}; border-radius: 3px; padding: 5px 14px; }}
QPushButton:hover {{ background-color: {hover}; border-color: {accent}; }}
QPushButton:pressed {{ background-color: {accent}; color: {sel_text}; }}
QPushButton:checked {{ background-color: {accent}; color: {sel_text}; }}
QPushButton:disabled {{ color: {disabled}; border-color: {border}; }}
QPushButton:default {{ border-color: {accent}; }}

/* --- inputs --- */
QLineEdit, QPlainTextEdit, QTextEdit {{ background-color: {base};
    color: {text}; border: 1px solid {border}; border-radius: 3px;
    padding: 4px 6px; selection-background-color: {accent}; }}
QLineEdit:focus, QPlainTextEdit:focus {{ border-color: {accent}; }}
QLineEdit:disabled {{ color: {disabled}; }}

QComboBox {{ background-color: {base}; color: {text};
    border: 1px solid {border}; border-radius: 3px; padding: 4px 6px; }}
QComboBox:hover {{ border-color: {accent}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background-color: {base}; color: {text};
    border: 1px solid {border}; selection-background-color: {accent};
    selection-color: {sel_text}; }}

QSpinBox, QDoubleSpinBox, QDateEdit, QDateTimeEdit {{
    background-color: {base}; color: {text}; border: 1px solid {border};
    border-radius: 3px; padding: 3px 6px; }}
QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {accent}; }}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background-color: {bg}; border: 1px solid {border}; width: 14px; }}

QCheckBox, QRadioButton {{ spacing: 6px; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 14px; height: 14px;
    border: 1px solid {border}; background: {base}; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {accent}; border-color: {accent}; }}

/* --- progress bar --- */
QProgressBar {{ background-color: {base}; border: 1px solid {border};
    border-radius: 3px; text-align: center; color: {text}; height: 16px; }}
QProgressBar::chunk {{ background-color: {accent}; border-radius: 2px; }}

/* --- scrollbars --- */
QScrollBar:vertical {{ background: {bg}; width: 10px; margin: 0; }}
QScrollBar:horizontal {{ background: {bg}; height: 10px; margin: 0; }}
QScrollBar::handle {{ background: {border}; border-radius: 4px;
    min-height: 24px; min-width: 24px; }}
QScrollBar::handle:hover {{ background: {disabled}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

/* --- misc chrome --- */
QMenuBar {{ background-color: {bg}; }}
QMenuBar::item:selected {{ background: {hover}; }}
QMenu {{ background-color: {base}; border: 1px solid {border}; }}
QMenu::item:selected {{ background-color: {accent}; color: {sel_text}; }}
QStatusBar {{ background-color: {bg}; border-top: 1px solid {border}; }}
QSplitter::handle {{ background-color: {border}; }}
QDockWidget::title {{ background: {base}; padding: 4px;
    border: 1px solid {border}; }}
""".format(**c)


def apply_dark_theme(app: QApplication) -> None:
    """Apply Fusion style + dark QPalette + QSS to the whole application.

    Call once, right after QApplication construction. P&L red/green in
    tables must be set via ForegroundRole in models, not QSS.
    """
    app.setStyle("Fusion")
    app.setPalette(_build_palette())
    app.setStyleSheet(_build_qss())
    log.debug("dark theme applied")

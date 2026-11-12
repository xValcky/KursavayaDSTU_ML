"""Темы оформления: светлая и тёмная. Скруглённые кнопки, карточки, спокойные цвета."""

THEMES = {
    "light": {
        "ACCENT": "#5B7FDB", "ACCENT_HOVER": "#4A6CC4", "ACCENT_SOFT": "#E8EEFC", "ACCENT_PRESSED": "#D6E0FA",
        "ON_ACCENT": "#FFFFFF", "BG": "#F5F6FA", "CARD": "#FFFFFF", "BORDER": "#E3E6EE", "TEXT": "#2E3440",
        "MUTED": "#7A8194", "DISABLED_TEXT": "#B0B5C3", "DISABLED_BG": "#F0F1F5", "PRIMARY_DISABLED": "#AEBDE6",
        "CHECK_BORDER": "#C5CBD8", "GROOVE": "#E6E9F0", "HANDLE": "#FFFFFF", "SCROLL": "#CDD2DE",
        "ALT_ROW": "#F9FAFD", "CHIP_INFO_BG": "#EEF0F5", "GOOD_BG": "#E3F4EA", "GOOD": "#2E7D4F",
        "BAD_BG": "#FCE9E9", "BAD": "#A94442", "LIKE_BORDER": "#9FD6B0", "DISLIKE_BORDER": "#F0B8B8",
        "WARN_BG": "#FFF4E6", "WARN_BORDER": "#F2C894", "SEG_BG": "#E9ECF3",
    },
    "dark": {
        "ACCENT": "#7B9BF0", "ACCENT_HOVER": "#6A8AE6", "ACCENT_SOFT": "#28304A", "ACCENT_PRESSED": "#33406A",
        "ON_ACCENT": "#0F1320", "BG": "#14161B", "CARD": "#1E2128", "BORDER": "#2F343F", "TEXT": "#E6E8EE",
        "MUTED": "#9AA2B5", "DISABLED_TEXT": "#5C6372", "DISABLED_BG": "#23262D", "PRIMARY_DISABLED": "#3C4A70",
        "CHECK_BORDER": "#4A5060", "GROOVE": "#343944", "HANDLE": "#1E2128", "SCROLL": "#3A404C",
        "ALT_ROW": "#22252D", "CHIP_INFO_BG": "#2B2F38", "GOOD_BG": "#1D3A2A", "GOOD": "#7FD3A0",
        "BAD_BG": "#3D2224", "BAD": "#F0A0A0", "LIKE_BORDER": "#3E7A57", "DISLIKE_BORDER": "#8A4646",
        "WARN_BG": "#3A2E1E", "WARN_BORDER": "#7A5A2A", "SEG_BG": "#262A33",
    },
}


def build_qss(theme: str = "light") -> str:
    c = THEMES.get(theme, THEMES["light"])
    return """
* {{ font-family: "Segoe UI", "Arial"; font-size: 10pt; color: {TEXT}; }}
QMainWindow, QWidget#page, QDialog {{ background: {BG}; }}
QStatusBar {{ background: {BG}; color: {MUTED}; }}
QToolTip {{
    background: {CARD}; color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 6px 9px; font-size: 9.5pt;
}}
QMenu {{ background: {CARD}; color: {TEXT}; border: 1px solid {BORDER}; }}
QMenu::item:selected {{ background: {ACCENT_SOFT}; }}
QMessageBox, QInputDialog {{ background: {BG}; }}

QTabWidget::pane {{ border: none; background: {BG}; top: -1px; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{
    background: transparent; color: {MUTED};
    padding: 8px 18px; margin: 4px 4px 6px 0; border-radius: 14px;
}}
QTabBar::tab:hover {{ background: {ACCENT_SOFT}; color: {TEXT}; }}
QTabBar::tab:selected {{ background: {ACCENT}; color: {ON_ACCENT}; font-weight: 600; }}
QTabBar QToolButton {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 6px; }}

QGroupBox {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px;
    margin-top: 14px; padding: 18px 14px 12px 14px; font-weight: 600;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 16px; padding: 0 6px; color: {ACCENT}; background: transparent; }}

QPushButton {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px;
    padding: 8px 18px; font-weight: 500;
}}
QPushButton:hover {{ background: {ACCENT_SOFT}; border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {ACCENT_PRESSED}; }}
QPushButton:disabled {{ color: {DISABLED_TEXT}; background: {DISABLED_BG}; }}
QPushButton#primary {{
    background: {ACCENT}; color: {ON_ACCENT}; border: none; border-radius: 20px;
    padding: 12px 24px; font-size: 12pt; font-weight: 600;
}}
QPushButton#primary:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#primary:disabled {{ background: {PRIMARY_DISABLED}; }}
QPushButton#like {{ border-color: {LIKE_BORDER}; color: {GOOD}; }}
QPushButton#like:hover {{ background: {GOOD_BG}; }}
QPushButton#dislike {{ border-color: {DISLIKE_BORDER}; color: {BAD}; }}
QPushButton#dislike:hover {{ background: {BAD_BG}; }}
QPushButton#chip {{ border-radius: 14px; padding: 6px 14px; }}
QPushButton#chip:checked {{ background: {ACCENT}; color: {ON_ACCENT}; border-color: {ACCENT}; }}

QFrame#segment {{ background: {SEG_BG}; border: 1px solid {BORDER}; border-radius: 17px; }}
QPushButton#segBtn {{
    background: transparent; border: none; border-radius: 14px;
    padding: 6px 16px; color: {MUTED}; font-weight: 600;
}}
QPushButton#segBtn:hover {{ color: {TEXT}; }}
QPushButton#segBtn:checked {{ background: {ACCENT}; color: {ON_ACCENT}; }}
QPushButton#themeBtn {{ border-radius: 17px; padding: 6px 14px; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 5px 9px; selection-background-color: {ACCENT}; selection-color: {ON_ACCENT};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {CARD}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 8px;
    selection-background-color: {ACCENT_SOFT}; selection-color: {TEXT};
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 16px; border: none; }}

QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 18px; height: 18px; border-radius: 6px; border: 1px solid {CHECK_BORDER}; background: {CARD};
}}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

QSlider::groove:horizontal {{ height: 6px; background: {GROOVE}; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 3px; }}
QSlider::handle:horizontal {{
    background: {HANDLE}; border: 2px solid {ACCENT};
    width: 16px; height: 16px; margin: -6px 0; border-radius: 9px;
}}

QTableView {{
    background: {CARD}; alternate-background-color: {ALT_ROW};
    border: 1px solid {BORDER}; border-radius: 12px; gridline-color: transparent;
    selection-background-color: {ACCENT_SOFT}; selection-color: {TEXT};
}}
QTableView::item {{ padding: 4px 8px; border: none; }}
QHeaderView {{ background: {CARD}; }}
QHeaderView::section {{
    background: {CARD}; color: {MUTED}; border: none;
    border-bottom: 1px solid {BORDER}; padding: 6px 8px; font-weight: 600;
}}
QTableCornerButton::section {{ background: {CARD}; border: none; }}

QScrollBar:vertical, QScrollBar:horizontal {{ background: transparent; width: 10px; height: 10px; }}
QScrollBar::handle {{ background: {SCROLL}; border-radius: 5px; min-height: 30px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}

QSplitter::handle {{ background: transparent; }}
QLabel#hint {{ color: {MUTED}; }}
QLabel#value {{ color: {ACCENT}; font-weight: 600; min-width: 22px; }}
QFrame#banner {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px; }}
QFrame#card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px; }}
QFrame#card:hover {{ border-color: {ACCENT}; }}
QLabel#cardTitle {{ font-size: 12pt; font-weight: 600; }}
QLabel#cardPrice {{ font-size: 14pt; font-weight: 700; color: {ACCENT}; }}
QLabel#badge {{ background: {ACCENT_SOFT}; color: {ACCENT}; border-radius: 9px; padding: 2px 10px;
               font-size: 9pt; font-weight: 600; }}
QLabel#chipInfo {{ background: {CHIP_INFO_BG}; color: {MUTED}; border-radius: 8px; padding: 1px 8px; font-size: 8.5pt; }}
QLabel#chipGood {{ background: {GOOD_BG}; color: {GOOD}; border-radius: 8px; padding: 1px 8px; font-size: 8.5pt;
                  font-weight: 600; }}
QLabel#chipBad {{ background: {BAD_BG}; color: {BAD}; border-radius: 8px; padding: 1px 8px; font-size: 8.5pt;
                 font-weight: 600; }}
QLabel#blockTitle {{ font-size: 13pt; font-weight: 700; color: {TEXT}; padding-top: 6px; }}
QFrame#hintBar {{ background: {WARN_BG}; border: 1px solid {WARN_BORDER}; border-radius: 12px; }}
QFrame#presetInfo {{ background: {ACCENT_SOFT}; border-radius: 12px; }}
""".format(**c)


QSS = build_qss("light")  # для совместимости со старым кодом


def build_palette(theme: str = "light"):
    """Палитра Qt под тему: для системных частей (подсказки, диалоги, выпадающие списки), которые
    таблица стилей покрывает не полностью."""
    from PySide6.QtGui import QColor, QPalette
    c = THEMES.get(theme, THEMES["light"])
    pal = QPalette()
    for role, key in ((QPalette.Window, "BG"), (QPalette.WindowText, "TEXT"), (QPalette.Base, "CARD"),
                      (QPalette.AlternateBase, "ALT_ROW"), (QPalette.Text, "TEXT"), (QPalette.Button, "CARD"),
                      (QPalette.ButtonText, "TEXT"), (QPalette.ToolTipBase, "CARD"), (QPalette.ToolTipText, "TEXT"),
                      (QPalette.Highlight, "ACCENT"), (QPalette.HighlightedText, "ON_ACCENT"),
                      (QPalette.PlaceholderText, "MUTED"), (QPalette.BrightText, "BAD"), (QPalette.Link, "ACCENT")):
        pal.setColor(role, QColor(c[key]))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(c["DISABLED_TEXT"]))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(c["DISABLED_TEXT"]))
    return pal

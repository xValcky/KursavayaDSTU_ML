"""Мягкая светлая тема: скруглённые кнопки, карточки, спокойные цвета."""

ACCENT = "#5B7FDB"
ACCENT_HOVER = "#4A6CC4"
ACCENT_SOFT = "#E8EEFC"
BG = "#F5F6FA"
CARD = "#FFFFFF"
BORDER = "#E3E6EE"
TEXT = "#2E3440"
MUTED = "#7A8194"

QSS = f"""
* {{
    font-family: "Segoe UI", "Arial";
    font-size: 10pt;
    color: {TEXT};
}}
QMainWindow, QWidget#page {{ background: {BG}; }}
QStatusBar {{ background: {BG}; color: {MUTED}; }}

QTabWidget::pane {{ border: none; background: {BG}; top: -1px; }}
QTabBar::tab {{
    background: transparent; color: {MUTED};
    padding: 8px 18px; margin: 4px 4px 6px 0;
    border-radius: 14px;
}}
QTabBar::tab:hover {{ background: {ACCENT_SOFT}; color: {TEXT}; }}
QTabBar::tab:selected {{ background: {ACCENT}; color: white; font-weight: 600; }}

QGroupBox {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px;
    margin-top: 14px; padding: 18px 14px 12px 14px; font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 16px; padding: 0 6px;
    color: {ACCENT}; background: transparent;
}}

QPushButton {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px;
    padding: 8px 18px; font-weight: 500;
}}
QPushButton:hover {{ background: {ACCENT_SOFT}; border-color: {ACCENT}; }}
QPushButton:pressed {{ background: #D6E0FA; }}
QPushButton:disabled {{ color: #B0B5C3; background: #F0F1F5; }}
QPushButton#primary {{
    background: {ACCENT}; color: white; border: none; border-radius: 20px;
    padding: 12px 24px; font-size: 12pt; font-weight: 600;
}}
QPushButton#primary:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#primary:disabled {{ background: #AEBDE6; }}
QPushButton#like {{ border-color: #9FD6B0; color: #2E7D4F; }}
QPushButton#like:hover {{ background: #E7F6EC; }}
QPushButton#dislike {{ border-color: #F0B8B8; color: #A94442; }}
QPushButton#dislike:hover {{ background: #FCECEC; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 5px 9px; selection-background-color: {ACCENT};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px;
    selection-background-color: {ACCENT_SOFT}; selection-color: {TEXT};
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 16px; border: none; }}

QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 18px; height: 18px; border-radius: 6px;
    border: 1px solid #C5CBD8; background: {CARD};
}}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

QSlider::groove:horizontal {{ height: 6px; background: #E6E9F0; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 3px; }}
QSlider::handle:horizontal {{
    background: white; border: 2px solid {ACCENT};
    width: 16px; height: 16px; margin: -6px 0; border-radius: 9px;
}}

QTableView {{
    background: {CARD}; alternate-background-color: #F9FAFD;
    border: 1px solid {BORDER}; border-radius: 12px; gridline-color: transparent;
    selection-background-color: {ACCENT_SOFT}; selection-color: {TEXT};
}}
QTableView::item {{ padding: 4px 8px; border: none; }}
QHeaderView::section {{
    background: {CARD}; color: {MUTED}; border: none;
    border-bottom: 1px solid {BORDER}; padding: 6px 8px; font-weight: 600;
}}
QTableCornerButton::section {{ background: {CARD}; border: none; }}

QScrollBar:vertical, QScrollBar:horizontal {{ background: transparent; width: 10px; height: 10px; }}
QScrollBar::handle {{ background: #CDD2DE; border-radius: 5px; min-height: 30px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QSplitter::handle {{ background: transparent; }}
QLabel#hint {{ color: {MUTED}; }}
QLabel#value {{ color: {ACCENT}; font-weight: 600; min-width: 22px; }}
QFrame#banner {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px; }}
"""

QSS += f"""
QFrame#card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px; }}
QFrame#card:hover {{ border-color: {ACCENT}; }}
QLabel#cardTitle {{ font-size: 12pt; font-weight: 600; }}
QLabel#cardPrice {{ font-size: 14pt; font-weight: 700; color: {ACCENT}; }}
QLabel#badge {{ background: {ACCENT_SOFT}; color: {ACCENT}; border-radius: 9px; padding: 2px 10px;
                font-size: 9pt; font-weight: 600; }}
QPushButton#chip {{ border-radius: 14px; padding: 6px 14px; }}
QPushButton#chip:checked {{ background: {ACCENT}; color: white; border-color: {ACCENT}; }}
QCheckBox#switch {{ font-weight: 600; color: {ACCENT}; spacing: 10px; }}
QCheckBox#switch::indicator {{ width: 38px; height: 20px; border-radius: 10px;
                              background: #D5DAE5; border: none; }}
QCheckBox#switch::indicator:checked {{ background: {ACCENT}; }}
QScrollArea {{ border: none; background: transparent; }}
QDialog {{ background: {BG}; }}
"""

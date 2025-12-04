# gui/style.py
from __future__ import annotations

from PySide6.QtGui import QPalette, QColor, QFont
from PySide6.QtWidgets import QApplication


def apply_dark_theme(app: QApplication) -> None:
    """
    Apply a custom dark (black/red/yellow) theme to the whole application.

    Call this once in main.py after creating QApplication and before
    creating MainWindow.
    """
    # Use Fusion so we are independent from macOS/Windows native look
    app.setStyle("Fusion")

    # ---- Palette (base colors) ----
    palette = QPalette()

    bg_window = QColor("#151515")
    bg_panel = QColor("#1E1F26")
    text_primary = QColor("#F5F5F5")
    text_muted = QColor("#A0A0A0")
    accent = QColor("#DE3B40")
    accent_disabled = QColor("#444444")

    palette.setColor(QPalette.Window, bg_window)
    palette.setColor(QPalette.WindowText, text_primary)
    palette.setColor(QPalette.Base, QColor("#101010"))
    palette.setColor(QPalette.AlternateBase, bg_panel)
    palette.setColor(QPalette.ToolTipBase, bg_panel)
    palette.setColor(QPalette.ToolTipText, text_primary)
    palette.setColor(QPalette.Text, text_primary)
    palette.setColor(QPalette.Button, QColor("#232329"))
    palette.setColor(QPalette.ButtonText, text_primary)
    palette.setColor(QPalette.Highlight, accent)
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, accent_disabled)
    palette.setColor(QPalette.Disabled, QPalette.WindowText, accent_disabled)

    app.setPalette(palette)

    # ---- Global font ----
    # (Qt will gracefully fall back if "Manrope" is not installed.)
    app.setFont(QFont("Manrope", 11))

    # ---- QSS stylesheet ----
    qss = """
    /* Base widgets ------------------------------------------------------- */
    QMainWindow, QWidget {
        background-color: #151515;
        color: #F5F5F5;
    }

    QLabel {
        color: #F5F5F5;
    }

    QLabel[role="muted"] {
        color: #A0A0A0;
    }

    /* Group boxes -------------------------------------------------------- */
    QGroupBox {
        border: 1px solid #333333;
        border-radius: 8px;
        margin-top: 18px;
        background-color: #1E1F26;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 4px;
        color: #FFC857;
        font-weight: 600;
    }

    /* Buttons ------------------------------------------------------------ */
    QPushButton {
        background-color: #2B2B30;
        color: #F5F5F5;
        border-radius: 6px;
        padding: 6px 14px;
        border: 1px solid #333333;
        font-weight: 500;
    }
    QPushButton:hover {
        background-color: #3A3A40;
    }
    QPushButton:pressed {
        background-color: #1F1F23;
    }
    QPushButton:disabled {
        background-color: #202024;
        color: #666666;
        border-color: #2A2A2A;
    }

    /* Primary (red) buttons */
    QPushButton#primaryButton {
        background-color: #DE3B40;
        border-color: #DE3B40;
        color: #FFFFFF;
        font-weight: 600;
    }
    QPushButton#primaryButton:hover {
        background-color: #F24E54;
    }
    QPushButton#primaryButton:pressed {
        background-color: #C03035;
    }

    /* Subtle danger / outline buttons */
    QPushButton#dangerButton {
        background-color: transparent;
        border-color: #DE3B40;
        color: #DE3B40;
    }
    QPushButton#dangerButton:hover {
        background-color: rgba(222, 59, 64, 0.15);
    }

    /* Flat secondary buttons (tool-like) */
    QPushButton.flat {
        background-color: transparent;
        border: none;
        padding: 4px 8px;
    }
    QPushButton.flat:hover {
        background-color: #242428;
    }

    /* Tabs --------------------------------------------------------------- */
    QTabWidget::pane {
        border: 1px solid #333333;
        border-radius: 8px;
        background: #1E1F26;
    }
    QTabBar::tab {
        background: #1E1F26;
        color: #A0A0A0;
        padding: 6px 18px;
        margin-right: 2px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
    }
    QTabBar::tab:hover {
        background: #262732;
        color: #FFFFFF;
    }
    QTabBar::tab:selected {
        background: #DE3B40;
        color: #FFFFFF;
        font-weight: 600;
    }

    /* Combo boxes & spin boxes ------------------------------------------ */
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
        background-color: #232329;
        border-radius: 4px;
        border: 1px solid #333333;
        padding: 4px 6px;
        selection-background-color: #DE3B40;
        selection-color: #FFFFFF;
    }
    QComboBox::drop-down {
        border: 0;
        width: 20px;
    }
    QComboBox::down-arrow {
        image: none;
        border: none;
    }

    /* Check boxes -------------------------------------------------------- */
    QCheckBox {
        spacing: 6px;
        color: #F5F5F5;
    }
    QCheckBox::indicator {
        width: 16px;
        height: 16px;
        border-radius: 3px;
        border: 1px solid #666666;
        background: #151515;
    }
    QCheckBox::indicator:checked {
        background-color: #DE3B40;
        border-color: #DE3B40;
    }

    /* Toolbar & tool buttons --------------------------------------------- */
    QToolBar {
        background: #1E1F26;
        border-bottom: 1px solid #333333;
        spacing: 6px;
        padding: 4px;
    }
    QToolButton {
        background-color: #2B2B30;
        border-radius: 4px;
        padding: 4px 8px;
        border: 1px solid #333333;
    }
    QToolButton:hover {
        background-color: #3A3A40;
    }
    QToolButton:pressed {
        background-color: #1F1F23;
    }

    /* Status bar --------------------------------------------------------- */
    QStatusBar {
        background-color: #101010;
        color: #A0A0A0;
    }

    /* Sliders (for stretch/contrast etc.) -------------------------------- */
    QSlider::groove:horizontal {
        height: 4px;
        background: #333333;
        border-radius: 2px;
    }
    QSlider::handle:horizontal {
        background: #FFC857;
        width: 12px;
        border-radius: 6px;
        margin: -4px 0;
    }
    QSlider::sub-page:horizontal {
        background: #DE3B40;
        border-radius: 2px;
    }

    /* Scroll bars -------------------------------------------------------- */
    QScrollBar:vertical {
        background: #151515;
        width: 10px;
        margin: 0;
    }
    QScrollBar::handle:vertical {
        background: #333333;
        border-radius: 4px;
    }
    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical {
        height: 0;
    }
    QScrollBar:horizontal {
        background: #151515;
        height: 10px;
        margin: 0;
    }
    QScrollBar::handle:horizontal {
        background: #333333;
        border-radius: 4px;
    }
    QScrollBar::add-line:horizontal,
    QScrollBar::sub-line:horizontal {
        width: 0;
    }
    """

    app.setStyleSheet(qss)
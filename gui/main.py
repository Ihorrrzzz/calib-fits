# gui/main.py

import sys
from PySide6.QtWidgets import QApplication

from main_window import MainWindow
from style import apply_dark_theme


def main() -> None:
    app = QApplication(sys.argv)

    # Apply custom dark / red / yellow theme
    apply_dark_theme(app)

    win = MainWindow()
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
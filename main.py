import multiprocessing
import sys

from PySide6.QtWidgets import QApplication

from src.paths import ensure_runtime_files


def main():
    ensure_runtime_files()
    from src.ui.main_window import MainWindow
    from src.ui.style import QSS

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

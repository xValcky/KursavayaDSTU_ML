import multiprocessing
import sys

from PySide6.QtWidgets import QApplication

from src.paths import ensure_runtime_files


def main():
    ensure_runtime_files()
    from src.ui.main_window import MainWindow
    from src.ui.style import build_qss

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(build_qss("light"))  # окно само применит сохранённую тему
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

"""Общие фикстуры. Окно приложения в тестах полностью изолировано: копия БД, свои настройки
(ini-файл вместо реестра Windows) и заглушки модальных окон — тесты не трогают данные пользователя."""
import os
import shutil
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def window(tmp_path_factory):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtCore import QSettings, QThreadPool

    from src import db
    from src.paths import ensure_runtime_files
    ensure_runtime_files()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tmp = tmp_path_factory.mktemp("app")
    db.init_db()
    real_db = db.DB_PATH
    shutil.copy(real_db, tmp / "phones.db")
    db.DB_PATH = tmp / "phones.db"

    import src.ui.main_window as mw
    real_settings, real_env = mw.QSettings, mw.ENV_PATH
    mw.QSettings = lambda *a: QSettings(str(tmp / "settings.ini"), QSettings.IniFormat)
    mw.ENV_PATH = tmp / ".env"
    messages = []
    saved_boxes = {n: getattr(mw.QMessageBox, n) for n in ("information", "warning", "critical", "question")}
    for n in ("information", "warning", "critical"):
        setattr(mw.QMessageBox, n, staticmethod(lambda *a, **k: messages.append(str(a[2]) if len(a) > 2 else "")))
    mw.QMessageBox.question = staticmethod(lambda *a, **k: mw.QMessageBox.Yes)

    w = mw.MainWindow()
    w.messages = messages
    t0 = time.time()
    while w.pipe is None and time.time() - t0 < 300:
        app.processEvents()
        time.sleep(0.05)
    yield w
    QThreadPool.globalInstance().waitForDone(300_000)  # дождаться фоновых задач, чтобы не рвать сигналы
    app.processEvents()
    w.close()
    mw.QSettings, mw.ENV_PATH, db.DB_PATH = real_settings, real_env, real_db
    for n, f in saved_boxes.items():
        setattr(mw.QMessageBox, n, f)

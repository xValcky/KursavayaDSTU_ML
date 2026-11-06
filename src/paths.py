"""Пути к данным: одинаково работают из исходников и из собранного exe.

В exe (PyInstaller) встроенные файлы лежат во временной папке распаковки,
а база и модели должны сохраняться между запусками, поэтому при первом
запуске они копируются в рабочую папку рядом с exe (или в %LOCALAPPDATA%,
если рядом писать нельзя).
"""
import os
import shutil
import sys
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)
SOURCE_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", SOURCE_ROOT))


def _app_dir() -> Path:
    if not FROZEN:
        return SOURCE_ROOT
    candidate = Path(sys.executable).resolve().parent / "PhonePicker_data"
    try:
        candidate.mkdir(exist_ok=True)
        probe = candidate / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        return candidate
    except OSError:
        fallback = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "PhonePicker"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


# PHONEPICKER_HOME — явная рабочая папка (её использует build_exe.py, чтобы собрать чистую базу)
APP_DIR = Path(os.environ["PHONEPICKER_HOME"]).resolve() if os.environ.get("PHONEPICKER_HOME") else _app_dir()
DATA_DIR = APP_DIR / "data"
MODELS_DIR = APP_DIR / "models"
ENV_PATH = APP_DIR / ".env"


def ensure_runtime_files() -> None:
    """Первый запуск exe: скопировать встроенные БД, датасет и обученные модели."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not FROZEN:
        source_seed = SOURCE_ROOT / "data" / "phones_seed.csv"
        if DATA_DIR != source_seed.parent:
            shutil.copy2(source_seed, DATA_DIR / "phones_seed.csv")
        return
    for rel in ("data/phones_seed.csv", "data/phones.db", "models/pipeline.joblib",
                "models/experiments.joblib", ".env.example"):
        src, dst = BUNDLE_DIR / rel, APP_DIR / rel
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    # датасет всегда берём свежий из сборки (он только читается)
    shutil.copy2(BUNDLE_DIR / "data/phones_seed.csv", DATA_DIR / "phones_seed.csv")

"""Автосборка PhonePicker.exe одной командой:  python build_exe.py

Что делает:
  1. ставит зависимости (requirements.txt + PyInstaller);
  2. в отдельной папке build/bundle создаёт чистую базу из data/phones_seed.csv
     (без ваших тестовых анкет и отзывов), обучает модель и считает эксперименты;
  3. собирает dist/PhonePicker.exe через PhonePicker.spec;
  4. по желанию запускает exe для проверки.

Ключи:
  --skip-install   не запускать pip
  --no-experiments не считать сравнение моделей (быстрее на 1–2 минуты)
  --run            запустить готовый exe после сборки
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "build" / "bundle"
EXE = ROOT / "dist" / "PhonePicker.exe"

TRAIN_SCRIPT = """
from src.paths import ensure_runtime_files
ensure_runtime_files()
from src import db
from src.ml.classifier import PhonePipeline
from src.ml import experiments
db.init_db()
print(f"  база: {db.stats()['phones']} телефонов")
pipe = PhonePipeline().fit(db.get_phones())
pipe.save()
t = pipe.metrics_["test"]
print(f"  нейросеть: accuracy {t['accuracy']:.3f}, кластеров {pipe.k_}")
if {run_exp}:
    experiments.run_all(pipe, progress=lambda s: print("   ·", s))
"""


def step(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def run(cmd: list[str], env: dict | None = None) -> None:
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    if result.returncode != 0:
        sys.exit(f"\nОшибка: команда завершилась с кодом {result.returncode}: {' '.join(cmd)}")


def main() -> None:
    args = set(sys.argv[1:])
    started = time.time()
    if sys.version_info < (3, 10):
        sys.exit("Нужен Python 3.10 или новее")

    if "--skip-install" not in args:
        step("1/4 Установка зависимостей")
        run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt", "pyinstaller"])
    else:
        step("1/4 Установка зависимостей пропущена")

    step("2/4 Чистая база и обучение модели")
    shutil.rmtree(BUNDLE, ignore_errors=True)
    (BUNDLE / "data").mkdir(parents=True)
    (BUNDLE / "models").mkdir(parents=True)
    shutil.copy2(ROOT / ".env.example", BUNDLE / ".env.example")
    env = {**os.environ, "PHONEPICKER_HOME": str(BUNDLE), "PYTHONIOENCODING": "utf-8"}
    run([sys.executable, "-c", TRAIN_SCRIPT.replace("{run_exp}", str("--no-experiments" not in args))], env)

    step("3/4 Сборка exe (PyInstaller, 2–4 минуты)")
    for p in (ROOT / "build" / "PhonePicker", EXE):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            try:
                p.unlink()
            except PermissionError:
                sys.exit("Закройте запущенный PhonePicker.exe и повторите сборку")
    run([sys.executable, "-m", "PyInstaller", "PhonePicker.spec", "--noconfirm", "--log-level", "WARN"],
        {**os.environ, "PHONEPICKER_BUNDLE": str(BUNDLE)})

    step("4/4 Готово")
    size = EXE.stat().st_size / 2**20
    print(f"Файл: {EXE}  ({size:.0f} МБ)")
    print(f"Время сборки: {(time.time() - started) / 60:.1f} мин")
    if "--run" in args:
        subprocess.Popen([str(EXE)], cwd=EXE.parent)
        print("Exe запущен.")


if __name__ == "__main__":
    main()

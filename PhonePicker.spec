# Сборка: python build_exe.py  (готовит чистую базу и модели в build/bundle, затем вызывает этот spec)
# Результат: dist/PhonePicker.exe — один файл, внутри БД, датасет и обученные модели.
import os
from pathlib import Path

# откуда брать базу и модели: build/bundle от build_exe.py, иначе — рабочие файлы проекта
BUNDLE = Path(os.environ.get("PHONEPICKER_BUNDLE", "."))
BUNDLE_FILES = ["data/phones_seed.csv", "data/phones.db", "models/pipeline.joblib",
                "models/experiments.joblib", ".env.example"]

# Ненужные части Qt и библиотек — без них exe заметно меньше.
EXCLUDES = [
    "tkinter", "pygame", "IPython", "jupyter", "notebook", "pytest", "torch", "tensorflow",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel", "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtQuick3D",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.QtMultimedia", "PySide6.QtPdf",
    "PySide6.QtPdfWidgets", "PySide6.QtCharts", "PySide6.QtDataVisualization",
    "PySide6.QtBluetooth", "PySide6.QtPositioning", "PySide6.QtLocation", "PySide6.QtSensors",
    "PySide6.QtSql", "PySide6.QtTest", "PySide6.QtDesigner", "PySide6.QtNetwork",
]

a = Analysis(
    ["main.py"],
    pathex=[],
    datas=[(str(BUNDLE / rel), str(Path(rel).parent)) for rel in BUNDLE_FILES if (BUNDLE / rel).exists()],
    hiddenimports=["sklearn.neural_network", "sklearn.ensemble", "sklearn.cluster",
                   "scipy.cluster.hierarchy", "matplotlib.backends.backend_qtagg"],
    excludes=EXCLUDES,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="PhonePicker",
    console=False,
    upx=False,
    runtime_tmpdir=None,
)

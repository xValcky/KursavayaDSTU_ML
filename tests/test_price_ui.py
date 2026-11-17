"""Регрессии: вилка цены в окне, история цен (данные, график, совет «покупать / ждать»)."""
import os

import pandas as pd
import pytest

from src import db
from src.ml.price_forecast import advice

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ---------- вилка цены: ввод с клавиатуры (окно — фикстура window из conftest.py, изолировано) ----------
def _type(sb, text, commit=True):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    sb.setFocus()
    sb.lineEdit().selectAll()
    QTest.keyClicks(sb.lineEdit(), text)
    if commit:
        QTest.keyClick(sb.lineEdit(), Qt.Key_Return)


@pytest.mark.parametrize("rounds", [1, 3])
def test_typing_ceiling_keeps_floor(window, rounds):
    """Баг: «от 10 000», затем «до 50 000» — при наборе «до» промежуточное 5 000 сбрасывало «от» до 5 000."""
    for _ in range(rounds):  # «после каждого подбора» — повторный ввод тоже не должен ломать вилку
        _type(window.budget_min, "10000")
        _type(window.budget, "50000")
        a = window.answers()
        assert (a["budget_min"], a["budget_rub"]) == (10000, 50000)


def test_uncommitted_text_is_used(window):
    """Набрали «до» и сразу нажали «Подобрать» (без Enter) — берётся набранное значение."""
    _type(window.budget_min, "20000")
    _type(window.budget, "80000", commit=False)
    a = window.answers()
    assert (a["budget_min"], a["budget_rub"]) == (20000, 80000)


def test_ceiling_below_floor_pulls_floor_down(window):
    """Осознанно ввели «до» ниже «от» — вилка остаётся корректной (от ≤ до)."""
    _type(window.budget_min, "40000")
    _type(window.budget, "30000")
    a = window.answers()
    assert a["budget_min"] <= a["budget_rub"] == 30000


# ---------- история цен: данные ----------
def test_history_seed_integrity():
    hist = pd.read_csv(db.HISTORY_SEED_PATH)
    phones = pd.read_csv(db.SEED_PATH)
    assert set(phones["model"]) <= set(hist["model"]), "у каждой модели есть хотя бы стартовая точка"
    assert (hist["price_rub"] > 0).all()
    assert not hist.duplicated(["model", "date", "source"]).any()
    assert pd.to_datetime(hist["date"]).max() <= pd.Timestamp.today()


def test_history_loaded_into_db():
    db.init_db()
    ph = db.get_phones(include_archived=True)
    h = db.get_price_history()
    assert h["phone_id"].isin(ph["id"]).all()
    assert ph["id"].isin(h["phone_id"]).all()
    one = db.get_price_history(int(h["phone_id"].iloc[0]))
    assert one["date"].is_monotonic_increasing


# ---------- совет «покупать / ждать» ----------
def _hist(rows):
    return pd.DataFrame(rows, columns=["date", "price_rub", "source"])


def test_advice_uses_catalog_price():
    """Последняя строка истории — дорогая комплектация, а в каталоге цена ниже обычной → «хорошая цена»."""
    h = _hist([("2026-10-01", 50000, "a"), ("2026-10-10", 50000, "a"), ("2026-10-20", 60000, "b")])
    assert advice(h, None).startswith("Сейчас дороже")
    assert advice(h, None, current=45000).startswith("Хорошая цена")


def test_advice_same_day_sources_order_independent():
    """Несколько каналов в один день: результат не зависит от порядка строк."""
    rows = [("2026-10-01", 50000, "a"), ("2026-10-02", 50000, "a"),
            ("2026-10-03", 40000, "a"), ("2026-10-03", 60000, "b")]
    assert advice(_hist(rows), None) == advice(_hist(rows[:2] + rows[:1:-1]), None)


# ---------- график истории ----------
@pytest.mark.parametrize("hist", [
    _hist([]),
    _hist([("2026-08-01", 14000, "оценка: август 2026")]),
    _hist([("2026-10-01", 14000, "telegram:ishopper")]),
    _hist([(f"2026-10-{d:02d}", 10000 + d, f"api-{d}") for d in range(1, 25)]),  # источников больше, чем цветов
])
def test_price_history_chart_does_not_crash(hist):
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from src.ui import charts
    charts.set_theme("dark")
    c = charts.MplCanvas(4.5, 2.4)
    charts.plot_price_history(c, hist, "t", fair=12000, current=13000)
    c.fig.canvas.draw()

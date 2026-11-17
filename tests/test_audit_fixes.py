"""Регрессии по внешнему аудиту (BUG-xx — номера из отчёта). Каждый тест падал на коде до исправления."""
import pandas as pd
import pytest

from src import db
from src.providers.base import PriceProvider


# ---------- импорт CSV (BUG-08, 09, 10) ----------
@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init_db()
    return tmp_path


def _row(**kw):
    base = pd.read_csv(db.SEED_PATH).iloc[0].drop(labels=["price_source"]).to_dict()
    return {**base, **kw}


def test_csv_bad_prices_are_dropped(tmp_db):
    """BUG-10: цена 0 / −5000 попадала в базу и роняла модель справедливой цены (log → NaN)."""
    path = tmp_db / "bad.csv"
    pd.DataFrame([_row(model="Zero", price_rub=0), _row(model="Neg", price_rub=-5000),
                  _row(model="Good", price_rub=20000)]).to_csv(path, index=False)
    n, report = db.import_csv(str(path))
    ph = db.get_phones()
    assert n == 1 and set(report["bad_price"]) == {"Zero", "Neg"}
    assert not ph["model"].isin(["Zero", "Neg"]).any()
    from src.ml.price_model import PriceModel
    PriceModel().fit(ph)  # не падает


def test_csv_from_russian_excel(tmp_db):
    """BUG-08: CSV из Excel — cp1251 и разделитель «;»."""
    path = tmp_db / "cp.csv"
    pd.DataFrame([_row(model="Тестовый Кириллица", price_rub=15000)]).to_csv(path, index=False, sep=";",
                                                                             encoding="cp1251")
    assert db.import_csv(str(path))[0] == 1
    assert "Тестовый Кириллица" in set(db.get_phones()["model"])


def test_csv_missing_required_columns_has_clear_message(tmp_db):
    """BUG-09: вместо KeyError — понятное сообщение о недостающих колонках."""
    path = tmp_db / "no.csv"
    pd.DataFrame([{"brand": "A", "model": "B"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="обязательных колонок: os, price_rub"):
        db.import_csv(str(path))


def test_csv_minimal_columns_filled(tmp_db):
    """Только бренд, модель, ОС и цена — остальные характеристики заполняются типичными значениями."""
    path = tmp_db / "min.csv"
    pd.DataFrame([{"brand": "X", "model": "Minimal", "os": "Android", "price_rub": 15000}]).to_csv(path, index=False)
    assert db.import_csv(str(path))[0] == 1
    row = db.get_phones().query("model == 'Minimal'").iloc[0]
    assert row[["year", "ram_gb", "battery_mah", "active_h"]].notna().all()


# ---------- сопоставление цен API (BUG-15) ----------
class _Fake(PriceProvider):
    name = "fake"

    def __init__(self, offers):
        self.offers = offers

    def is_configured(self):
        return True

    def fetch_offers(self):
        return self.offers


@pytest.mark.parametrize("offers,expected", [
    ([("Samsung Galaxy S25 Ultra 12/256", 99990), ("Samsung Galaxy S25 8/128", 59990)], {1: 59990, 2: 99990}),
    ([("Samsung Galaxy S25 Ultra 12/256", 99990)], {2: 99990}),  # цена Ultra не достаётся базовой S25
])
def test_offer_goes_to_longest_model_name(offers, expected):
    phones = pd.DataFrame({"id": [1, 2], "model": ["Galaxy S25", "Galaxy S25 Ultra"]})
    assert _Fake(offers).match_prices(phones) == expected


# ---------- окно (фикстура window из conftest.py: копия БД и свои настройки) ----------
def _pick(window, lo=0, hi=50000, segments=False):
    window.segments_box.setChecked(segments)
    window.budget_min.setValue(lo)
    window.budget.setValue(hi)
    window.on_pick()


def test_corrupt_settings_do_not_crash(window):
    """BUG-11: битый JSON в настройках и частота вне списка не роняют окно."""
    import src.ui.main_window as mw
    window.settings.setValue("profiles", "{bad")
    assert window._profiles() == {}
    window.settings.setValue("favorites", "[1,")
    assert mw._json_setting(window.settings, "favorites", []) == []
    window.set_answers({"min_hz": 60})
    assert window.min_hz.currentIndex() == 0
    window.settings.setValue("profiles", "{}")


def test_duel_with_single_row_does_not_crash(window):
    """BUG-02: в выдаче одна модель → сообщение, а не IndexError."""
    _pick(window)
    full = window.last_res
    one = full["table"].head(1)
    window.last_res = {**full, "table": one}
    window.result_model.set_df(one.drop(columns=["_pros", "_badges", "_ok"]))
    window.result_view.selectRow(0)
    window.messages.clear()
    window.duel_with_best()
    assert window.messages and "одна модель" in window.messages[-1]
    window.last_res = full


def test_duel_leader_ignores_header_sort(window):
    """После сортировки по заголовку «лидер» дуэли — первый в выдаче, а не первая строка таблицы."""
    from PySide6.QtCore import Qt
    _pick(window)
    leader = int(window.last_res["table"].iloc[0]["id"])
    m = window.result_model
    window.result_view.sortByColumn(list(m.df.columns).index("Цена, ₽"), Qt.AscendingOrder)
    window.result_view.selectRow(2)
    window.duel_with_best()
    assert window.duel.a.currentData() == leader


def test_refresh_does_not_duplicate_survey(window):
    """BUG-04: пересчёт после справедливых цен / переключателей не пишет новую анкету и не уводит с вкладки."""
    _pick(window)
    n0 = db.stats()["survey_answers"]
    window.tabs.setCurrentIndex(0)
    window.on_aux_ready((window.price_model, window.forecast))
    window.diverse_box.toggle()
    window.diverse_box.toggle()
    assert db.stats()["survey_answers"] == n0 and window.tabs.currentIndex() == 0


def test_archived_phone_not_recommended_before_retrain(window):
    """BUG-06: модель в архиве сразу пропадает из выдачи, без переобучения."""
    _pick(window)
    best = int(window.last_res["table"].iloc[0]["id"])
    db.set_archived(best, True)
    try:
        window.reload_phones()
        window.on_pick()
        assert best not in set(window.last_res["table"]["id"])
    finally:
        db.set_archived(best, False)
        window.reload_phones()


def test_export_blocks_match_segments(window, tmp_path, monkeypatch):
    """BUG-19: при сегментах лист «Блоки» — те же модели, что на экране."""
    import src.ui.main_window as mw
    _pick(window, 0, 100000, segments=True)
    path = tmp_path / "sel.xlsx"
    monkeypatch.setattr(mw.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(path), "")))
    window.export_selection()
    blocks = pd.read_excel(path, sheet_name="Блоки")
    top = pd.read_excel(path, sheet_name="Топ")
    assert len(blocks) and set(blocks["телефон"]) <= set(top["Телефон"])


def test_favorites_tie_click_and_remove(window):
    """BUG-16 ничья, BUG-20 клик по карточке, BUG-21 убранная карточка исчезает."""
    import src.ui.main_window as mw
    from src.ui.widgets import PhoneCard
    ph = window.pipe.phones_
    pair = ph.groupby("ram_gb").filter(lambda g: len(g) >= 2).groupby("ram_gb").head(2).head(2)
    window.favorites = set(pair["model"])
    window.tabs.setCurrentIndex(mw.TAB_FAV)
    df = window.fav_compare.df
    assert df.loc[df["параметр"] == "ОЗУ, ГБ", "лучший"].iloc[0] == "одинаково"
    cards = [window.fav_cards.itemAt(i).widget() for i in range(window.fav_cards.count())]
    assert all(c.on_details is not None for c in cards if isinstance(c, PhoneCard))
    window.toggle_favorite(next(iter(window.favorites)))
    assert window.fav_cards.count() == 1
    window.favorites.clear()
    window.tabs.setCurrentIndex(0)


def test_export_report_when_nothing_ready(window, monkeypatch):
    """BUG-24: пустой экспорт — сообщение, а не IndexError из openpyxl."""
    monkeypatch.setattr(window, "exp", None)
    monkeypatch.setattr(window, "price_model", None)
    monkeypatch.setattr(window, "forecast", None)
    window.messages.clear()
    window.export_report()
    assert window.messages and "нечего" in window.messages[-1]


def test_save_keys_keeps_other_lines(window):
    """BUG-25: сохранение ключей не стирает комментарии и чужие переменные в .env."""
    import src.ui.main_window as mw
    mw.ENV_PATH.write_text("# мой комментарий\nOTHER=1\nOZON_API_KEY=old\n", encoding="utf-8")
    window.key_edits["OZON_API_KEY"].setText("new")
    window.save_keys()
    txt = mw.ENV_PATH.read_text(encoding="utf-8")
    assert "# мой комментарий" in txt and "OTHER=1" in txt
    assert "OZON_API_KEY=new" in txt and txt.count("OZON_API_KEY") == 1
    window.key_edits["OZON_API_KEY"].setText("")

"""Sanity-тесты подбора: ловят «поломку смысла» при правке весов и формул.

Запуск:  python -m pytest -q
"""
import numpy as np
import pytest

from src import db
from src.ml.ahp import ahp_weights
from src.ml.classifier import RUGGED_NAME, PhonePipeline
from src.ml.evaluation import ndcg_at_k
from src.ml.features import effective_refresh, is_rugged
from src.ml.labeling import IMPORTANCE_DEFAULTS, PRESETS
from src.ml.recommender import MAX_SAME_BRAND_TOP5, recommend, score_phones
from src.preprocess import MinMaxScalerCustom


@pytest.fixture(scope="module")
def pipe():
    db.init_db()
    return PhonePipeline.load() or PhonePipeline(n_surveys=3000).fit(db.get_phones())


def answers(preset="", **kw):
    a = dict(budget_rub=35000, **IMPORTANCE_DEFAULTS, gaming=0, photo=0, work=0, os_pref=0, need_nfc=1,
             esim_pref=0, ir_pref=0, screen_pref=0, min_hz=0, min_rom=128)
    a.update(PRESETS.get(preset, {}))
    a.update(kw)
    return a


# ---------- математика ----------
def test_minmax_constant_column_has_no_nan():
    x = np.array([[1.0, 5.0], [1.0, 7.0], [1.0, 9.0]])
    out = MinMaxScalerCustom().fit_transform(x)
    assert not np.isnan(out).any()


def test_effective_refresh_knee():
    assert effective_refresh(120) == 120
    assert effective_refresh(144) == pytest.approx(124.8)
    assert effective_refresh(60) == 60


def test_ahp_consistent_and_inconsistent():
    w, cr = ahp_weights(["a", "b", "c"], {("a", "b"): 3, ("a", "c"): 9, ("b", "c"): 3})
    assert sum(w.values()) == pytest.approx(1, abs=1e-3) and w["a"] > w["b"] > w["c"] and cr < 0.1
    _, cr_bad = ahp_weights(["a", "b", "c"], {("a", "b"): 9, ("b", "c"): 9, ("a", "c"): 1 / 9})
    assert cr_bad > 0.1


def test_ndcg_perfect_and_empty():
    rel = {"A": 3, "B": 2, "C": 1}
    assert ndcg_at_k(["A", "B", "C"], rel, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["X", "Y", "Z"], rel, 3) == 0.0


# ---------- кластеризация ----------
def test_rugged_phones_are_own_group(pipe):
    rug = is_rugged(pipe.phones_)
    if rug.any():
        groups = set(pipe.clusters_[rug])
        assert len(groups) == 1 and pipe.cluster_names_[groups.pop()] == RUGGED_NAME


def test_no_giant_cluster(pipe):
    sizes = np.bincount(pipe.clusters_)
    assert sizes.max() / sizes.sum() < 0.35  # раньше было 62% в одной группе


def test_soft_bonus_in_range(pipe):
    s = score_phones(pipe, answers())
    assert (s["bonus"] >= 0).all() and (s["bonus"] <= 0.4 + 1e-9).all()


# ---------- смысл выдачи ----------
def test_taxi_top5_long_battery(pipe):
    t = recommend(pipe, answers("Такси и курьер", budget_rub=30000), top_n=5)["table"]
    median = pipe.phones_.loc[pipe.phones_["price_rub"] <= 33000, "active_h"].median()
    assert (t["Активно, ч"] >= median).all()


def test_photo_top5_good_camera(pipe):
    t = recommend(pipe, answers("Фото и видео", budget_rub=40000), top_n=5)["table"]
    assert (t["Камера, МП"] >= 50).all()


@pytest.mark.parametrize("preset", list(PRESETS))
def test_top10_within_budget(pipe, preset):
    t = recommend(pipe, answers(preset, budget_rub=40000))["table"]
    assert (t["Цена, ₽"] <= 40000 * 1.2).all()


@pytest.mark.parametrize("preset", list(PRESETS))
def test_diversity(pipe, preset):
    t = recommend(pipe, answers(preset, budget_rub=40000))["table"]
    brands = t["Телефон"].head(5).str.split().str[0]
    assert brands.value_counts().max() <= MAX_SAME_BRAND_TOP5
    assert t["Кластер"].nunique() >= 3


def test_hint_when_requirements_too_strict(pipe):
    r = recommend(pipe, answers(budget_rub=12000, esim_pref=2, ir_pref=2))
    assert r["hint"] is not None and r["hint"]["text"]

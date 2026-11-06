"""Эксперименты для курсовой: сравнение моделей, абляция «с кластеризацией / без», веса признаков.

Запуск из консоли:  python -m src.ml.experiments [--grid]
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from src import db
from src.ml import clustering
from src.ml.classifier import MODEL_VERSION, PhonePipeline
from src.ml.ensemble import make_mlp, model_zoo
from src.ml.features import PhoneFeatureSpace
from src.paths import FROZEN, MODELS_DIR

EXP_PATH = MODELS_DIR / "experiments.joblib"


def clustering_experiment(pipe: PhonePipeline) -> pd.DataFrame:
    """Силуэт KMeans с весами признаков и без них (при одинаковом k)."""
    raw = PhoneFeatureSpace(use_weights=False).fit(pipe.phones_)
    rows = []
    for name, x in (("без весов (только StandardScaler)", raw.transform(pipe.phones_)),
                    ("с весами (эксперт + MI + штраф корреляций)", pipe.x_)):
        lab = clustering.make_kmeans(pipe.k_).fit_predict(x)
        sizes = np.bincount(lab)
        # насколько кластеры просто повторяют ценовые метки (чем ближе к 1 — тем сильнее «перетягивает» цена)
        purity = sum(np.bincount(pipe.labels_[lab == c]).max() for c in range(pipe.k_)) / len(lab)
        rows.append({"пространство": name, "silhouette": clustering.silhouette_score(x, lab),
                     "размеры кластеров": "/".join(map(str, sizes)),
                     "совпадение с метками цены": purity})
    return pd.DataFrame(rows)


def ablation(pipe: PhonePipeline, n_splits: int = 5, progress=print) -> pd.DataFrame:
    surveys_x_full, y = pipe.X_, pipe.y_
    n_base = surveys_x_full.shape[1] - 2 * pipe.k_
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    rows = []
    for name, model in model_zoo(pipe.mlp_params).items():
        for variant, X in (("без кластеризации", surveys_x_full[:, :n_base]),
                           ("с кластеризацией", surveys_x_full)):
            progress(f"{name} — {variant}")
            res = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=1)
            rows.append({"модель": name, "вариант": variant,
                         "accuracy": res["test_accuracy"].mean(),
                         "accuracy_std": res["test_accuracy"].std(),
                         "f1_macro": res["test_f1_macro"].mean()})
    return pd.DataFrame(rows)


def grid_search_mlp(pipe: PhonePipeline) -> pd.DataFrame:
    params = {"clf__hidden_layer_sizes": [(32,), (64, 32), (128, 64)],
              "clf__alpha": [1e-4, 1e-3, 1e-2]}
    gs = GridSearchCV(make_mlp(), params, cv=3, scoring="f1_macro", n_jobs=1 if FROZEN else -1)
    gs.fit(pipe.X_, pipe.y_)
    res = pd.DataFrame(gs.cv_results_)[["params", "mean_test_score", "std_test_score"]]
    return res.sort_values("mean_test_score", ascending=False)


def run_all(pipe: PhonePipeline, grid: bool = False, progress=print) -> dict:
    progress("Сравнение алгоритмов кластеризации")
    out = {
        "algorithms": clustering.compare_algorithms(pipe.x_, pipe.k_),
        "weights": clustering_experiment(pipe),
        "ablation": ablation(pipe, progress=progress),
    }
    if grid:
        progress("GridSearch для MLP")
        out["grid"] = grid_search_mlp(pipe)
    EXP_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({**out, "version": MODEL_VERSION}, EXP_PATH)
    return out


def load_results() -> dict | None:
    try:
        res = joblib.load(EXP_PATH)
    except Exception:
        return None
    return res if res.get("version") == MODEL_VERSION else None


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    db.init_db()
    pipe = PhonePipeline.load() or PhonePipeline().fit(db.get_phones(), db.get_liked_feedback())
    res = run_all(pipe, grid="--grid" in sys.argv)
    for key, table in res.items():
        if key == "version":
            continue
        print(f"\n=== {key} ===")
        print(table.round(4).to_string(index=False))
    pivot = res["ablation"].pivot(index="модель", columns="вариант", values="accuracy")
    print("\n=== accuracy: с кластеризацией vs без ===")
    print(pivot.assign(прирост=pivot["с кластеризацией"] - pivot["без кластеризации"]).round(4))

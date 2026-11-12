"""Проверка методов на эталонном наборе UCI Wine (178 вин, 13 признаков, 3 сорта).

Для Wine опубликованы результаты (leave-one-out): RDA 100%, QDA 99,4%, LDA 98,9%, 1NN 96,1%.
Если наш конвейер (масштабирование → модели → ансамбли, KMeans) даёт сопоставимые цифры,
значит методы реализованы правильно — сильный аргумент на защите.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from src.ml.ensemble import model_zoo
from src.paths import BUNDLE_DIR, SOURCE_ROOT

WINE_COLUMNS = ["Class label", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
                "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
                "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
PUBLISHED = {"RDA": 1.0, "QDA": 0.994, "LDA": 0.989, "1NN": 0.961}


def wine_path() -> Path | None:
    for base in (SOURCE_ROOT, BUNDLE_DIR):
        p = base / "data" / "benchmarks" / "wine.data"
        if p.exists():
            return p
    return None


def wine_benchmark(progress=lambda s: None) -> pd.DataFrame | None:
    path = wine_path()
    if path is None:
        return None
    df = pd.read_csv(path, header=None, names=WINE_COLUMNS)
    X, y = df.iloc[:, 1:].to_numpy(float), df["Class label"].to_numpy() - 1
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    rows = []
    for name, model in model_zoo().items():
        progress(f"Wine: {name}")
        s = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        rows.append({"метод": name, "accuracy (10-fold)": s.mean(), "±": s.std(), "источник": "наш конвейер"})
    km = KMeans(n_clusters=3, n_init=20, random_state=0).fit_predict(StandardScaler().fit_transform(X))
    rows.append({"метод": "KMeans k=3 (без учителя): ARI / NMI с сортами",
                 "accuracy (10-fold)": adjusted_rand_score(y, km), "±": normalized_mutual_info_score(y, km),
                 "источник": "наш конвейер"})
    rows += [{"метод": k, "accuracy (10-fold)": v, "±": np.nan, "источник": "публикация (leave-one-out)"}
             for k, v in PUBLISHED.items()]
    return pd.DataFrame(rows)

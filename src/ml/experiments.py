"""Эксперименты для курсовой: сравнение моделей, абляция «с кластеризацией / без», веса признаков,
стабильность групп по шпилям, кривые ROC, границы решений, эталон Wine; выгрузка в Excel.

Запуск из консоли:  python -m src.ml.experiments [--grid]
"""
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.preprocessing import StandardScaler, label_binarize

from src import db
from src.ml import clustering
from src.ml.benchmark import wine_benchmark
from src.ml.classifier import MODEL_VERSION, PhonePipeline
from src.ml.ensemble import make_mlp, model_zoo
from src.ml.features import PhoneFeatureSpace
from src.paths import APP_DIR, FROZEN, MODELS_DIR

EXP_PATH = MODELS_DIR / "experiments.joblib"
REPORT_PATH = APP_DIR / "reports" / "results_Comparison.xlsx"
ROC_MODELS = ["Логистическая регрессия", "Дерево решений", "KNN", "Нейросеть MLP", "Majority Vote (ансамбль)"]


def clustering_experiment(pipe: PhonePipeline) -> pd.DataFrame:
    """Как делить телефоны: весь рынок без весов / с весами против раздельно по шпилям (наш вариант)."""
    raw = PhoneFeatureSpace(use_weights=False).fit(pipe.phones_)
    k = 5
    rows = []
    for name, x in (("весь рынок, без весов, k=5", raw.transform(pipe.phones_)),
                    ("весь рынок, с весами, k=5", pipe.x_)):
        lab = clustering.make_kmeans(k).fit_predict(x)
        sizes = np.bincount(lab)
        rows.append({"вариант": name, "групп": k, "silhouette": clustering.silhouette_score(x, lab),
                     "крупнейшая группа, %": sizes.max() / len(lab) * 100,
                     "размеры": "/".join(map(str, sorted(sizes, reverse=True)))})
    sizes = np.bincount(pipe.clusters_)
    rows.append({"вариант": "по шпилям + броневики правилом (наш)", "групп": pipe.k_,
                 "silhouette": clustering.silhouette_score(pipe.x_, pipe.clusters_),
                 "крупнейшая группа, %": sizes.max() / len(pipe.clusters_) * 100,
                 "размеры": "/".join(map(str, sorted(sizes, reverse=True)))})
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


def roc_data(pipe: PhonePipeline, progress=print) -> dict:
    """Кривые ROC (усреднённые по классам 0/1/2, micro-average) на кросс-валидации."""
    zoo = model_zoo(pipe.mlp_params)
    yb = label_binarize(pipe.y_, classes=[0, 1, 2])
    out = {}
    for name in ROC_MODELS:
        progress(f"ROC: {name}")
        proba = cross_val_predict(zoo[name], pipe.X_, pipe.y_, cv=3, method="predict_proba")
        fpr, tpr, _ = roc_curve(yb.ravel(), proba.ravel())
        out[name] = {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}
    return out


def decision_regions(pipe: PhonePipeline, n: int = 1500, grid: int = 160) -> dict:
    """Границы решений моделей на 2D-проекции анкет (PCA), как в ens.py."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pipe.X_), size=min(n, len(pipe.X_)), replace=False)
    X2 = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(pipe.X_))[idx]
    y = pipe.y_[idx]
    xs = np.linspace(X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5, grid)
    ys = np.linspace(X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5, grid)
    xx, yy = np.meshgrid(xs, ys)
    zoo = model_zoo(pipe.mlp_params)
    regions = {}
    for name in ["Логистическая регрессия", "Дерево решений", "KNN", "Majority Vote (ансамбль)"]:
        model = zoo[name].fit(X2, y)
        regions[name] = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return {"X": X2, "y": y, "xx": xx, "yy": yy, "regions": regions}


def tier_report(pipe: PhonePipeline) -> pd.DataFrame:
    return pd.DataFrame(pipe.tier_stats_)


def grid_search_mlp(pipe: PhonePipeline) -> pd.DataFrame:
    params = {"clf__hidden_layer_sizes": [(32,), (64, 32), (128, 64)],
              "clf__alpha": [1e-4, 1e-3, 1e-2]}
    gs = GridSearchCV(make_mlp(), params, cv=3, scoring="f1_macro", n_jobs=1 if FROZEN else -1)
    gs.fit(pipe.X_, pipe.y_)
    res = pd.DataFrame(gs.cv_results_)[["params", "mean_test_score", "std_test_score"]]
    res["params"] = res["params"].astype(str)
    return res.sort_values("mean_test_score", ascending=False)


def run_all(pipe: PhonePipeline, grid: bool = False, progress=print) -> dict:
    progress("Сравнение алгоритмов кластеризации")
    budget = np.flatnonzero(pipe.labels_ == 0)
    out = {
        "algorithms": clustering.compare_algorithms(pipe.x_[budget], 4),
        "weights": clustering_experiment(pipe),
        "tiers": tier_report(pipe),
        "ablation": ablation(pipe, progress=progress),
        "roc": roc_data(pipe, progress=progress),
        "regions": decision_regions(pipe),
    }
    wine = wine_benchmark(progress)
    if wine is not None:
        out["wine"] = wine
    if grid:
        progress("GridSearch для MLP")
        out["grid"] = grid_search_mlp(pipe)
    EXP_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({**out, "version": MODEL_VERSION}, EXP_PATH)
    return out


def export_excel(exp: dict | None, price_model=None, forecast=None, path=REPORT_PATH) -> str:
    """Все таблицы экспериментов в один файл results_Comparison.xlsx (по листу на эксперимент)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sheets = {}
    if exp:
        names = {"ablation": "Классификация", "algorithms": "Алгоритмы кластеризации",
                 "weights": "Варианты кластеризации", "tiers": "Группы по шпилям", "wine": "Эталон Wine",
                 "grid": "GridSearch MLP"}
        sheets.update({title: exp[key] for key, title in names.items() if key in exp})
        if "roc" in exp:
            sheets["ROC AUC"] = pd.DataFrame([{"модель": k, "ROC AUC (micro)": v["auc"]} for k, v in exp["roc"].items()])
    if price_model is not None:
        sheets["Регрессия цены"] = price_model.results_
        sheets["Справедливые цены"] = price_model.oof_
    if forecast is not None:
        sheets["Прогноз цен"] = forecast.results_
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        for title, df in sheets.items():
            df.to_excel(xl, sheet_name=title[:31], index=False)
    return str(path)


def load_results() -> dict | None:
    try:
        res = joblib.load(EXP_PATH)
    except Exception:
        return None
    return res if res.get("version") == MODEL_VERSION else None


if __name__ == "__main__":
    from src.ml.price_forecast import PriceForecast
    from src.ml.price_model import PriceModel
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    db.init_db()
    pipe = PhonePipeline.load() or PhonePipeline().fit(db.get_phones(), db.get_liked_feedback())
    res = run_all(pipe, grid="--grid" in sys.argv)
    pm = PriceModel().fit(pipe.phones_)
    fc = PriceForecast().fit(db.get_price_history(), dict(zip(pipe.phones_["id"].astype(int), pipe.labels_)))
    for key, table in res.items():
        if isinstance(table, pd.DataFrame):
            print(f"\n=== {key} ===")
            print(table.round(4).to_string(index=False))
    print("\n=== ROC AUC ===")
    for k, v in res["roc"].items():
        print(f"  {k}: {v['auc']:.3f}")
    print("\n=== Регрессия цены ===")
    print(pm.results_.round(3).to_string(index=False))
    print("\nОтчёт:", export_excel(res, pm, fc))

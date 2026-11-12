"""«Справедливая цена» — регрессия цены телефона по его характеристикам (лаба 3).

Методы: парная и множественная линейная регрессия, нелинейная регрессия через scipy.curve_fit,
KNN, SVR, дерево решений, случайный лес, бэггинг, градиентный бустинг, однослойный и
многослойный персептроны. Сравнение по MAE / MSE / коэффициенту корреляции на кросс-валидации,
с масштабированием и без.

Справедливая цена каждого телефона — прогноз лучшей модели, которая этот телефон при обучении
НЕ видела (cross_val_predict), иначе модель «подгоняется» под его же цену.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.ml.features import effective_refresh
from src.paths import MODELS_DIR

PRICE_MODEL_PATH = MODELS_DIR / "price_model.joblib"
PRICE_MODEL_VERSION = 2  # меняется при изменении набора моделей/признаков
REG_FEATURES = ["antutu_k", "ram_gb", "rom_gb", "camera_mp", "battery_mah", "screen_in", "refresh_hz",
                "weight_g", "charge_w", "year", "has_5g", "nfc", "esim", "ir_port", "is_ios"]
LOG = ["antutu_k", "ram_gb", "rom_gb", "camera_mp", "battery_mah", "charge_w", "weight_g"]
MIN_BRAND = 5          # бренды с ≥ 5 моделями получают свой признак (one-hot)
CV = KFold(n_splits=5, shuffle=True, random_state=0)


def design_matrix(df: pd.DataFrame, brands: list[str]) -> pd.DataFrame:
    d = pd.DataFrame(index=df.index)
    for f in REG_FEATURES:
        if f == "is_ios":
            d[f] = (df["os"] == "iOS").astype(float)
        elif f == "refresh_hz":
            d[f] = effective_refresh(df[f])
        elif f in LOG:
            d[f] = np.log1p(df[f].astype(float))
        else:
            d[f] = df[f].astype(float)
    for b in brands:
        d[f"brand_{b}"] = (df["brand"] == b).astype(float)
    return d


class PowerLawRegressor(BaseEstimator, RegressorMixin):
    """Нелинейная регрессия через curve_fit:  цена = a · AnTuTu^b · ROM^c · RAM^d · МП^e · exp(g·(год−2024)).

    Работает на сырых признаках (передаются колонки antutu_k, rom_gb, ram_gb, camera_mp, year)."""

    COLS = ["antutu_k", "rom_gb", "ram_gb", "camera_mp", "year"]

    @staticmethod
    def f(X, a, b, c, d, e, g):
        antutu, rom, ram, mp, year = X
        return a * antutu ** b * rom ** c * ram ** d * mp ** e * np.exp(g * (year - 2024))

    def fit(self, X, y):
        X = np.asarray(X, float).T
        self.params_, _ = curve_fit(self.f, X, np.exp(y), p0=[300, 0.4, 0.1, 0.1, 0.05, 0.1], maxfev=20000)
        return self

    def predict(self, X):
        return np.log(np.clip(self.f(np.asarray(X, float).T, *self.params_), 1000, None))


def _scaled(est):
    return Pipeline([("sc", StandardScaler()), ("reg", est)])


def model_zoo(pair_feature: str) -> dict:
    """Все методы лабы 3. Параметрические — с подбором гиперпараметров (GridSearchCV)."""
    gs = lambda est, grid: GridSearchCV(est, grid, cv=3, scoring="neg_mean_absolute_error")
    return {
        f"Парная регрессия ({pair_feature})": ("pair", LinearRegression()),
        "Множественная регрессия": ("all", LinearRegression()),
        "Нелинейная (curve_fit)": ("raw", PowerLawRegressor()),
        "KNN": ("all", gs(_scaled(KNeighborsRegressor()), {"reg__n_neighbors": [3, 5, 7, 10]})),
        "KNN без масштабирования": ("all", gs(KNeighborsRegressor(), {"n_neighbors": [3, 5, 7, 10]})),
        "SVR": ("all", gs(_scaled(SVR()), {"reg__C": [1, 10, 100], "reg__epsilon": [0.05, 0.1]})),
        "SVR без масштабирования": ("all", gs(SVR(), {"C": [1, 10, 100]})),
        "Дерево решений": ("all", gs(DecisionTreeRegressor(random_state=0), {"max_depth": [3, 5, 8, None]})),
        "Случайный лес": ("all", RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0)),
        "Бэггинг (деревья)": ("all", BaggingRegressor(DecisionTreeRegressor(), n_estimators=200, random_state=0)),
        "Градиентный бустинг": ("all", gs(GradientBoostingRegressor(random_state=0),
                                         {"n_estimators": [200, 400], "max_depth": [2, 3]})),
        # однослойный персептрон с одним нейроном ≈ линейная модель с функцией активации
        "Персептрон (1 нейрон)": ("all", _scaled(MLPRegressor(hidden_layer_sizes=(1,), activation="identity",
                                                              max_iter=5000, alpha=0.1, random_state=0))),
        # на 287 примерах сеть быстро переобучается: нужна сильная L2-регуляризация (alpha=3)
        "Многослойный персептрон": ("all", _scaled(MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=5000,
                                                                alpha=3.0, random_state=0))),
    }


class PriceModel:
    def fit(self, phones: pd.DataFrame, progress=lambda s: None):
        df = phones.reset_index(drop=True)
        counts = df["brand"].value_counts()
        self.brands_ = sorted(counts[counts >= MIN_BRAND].index)
        X = design_matrix(df, self.brands_)
        y = np.log(df["price_rub"].to_numpy(float))
        corr = X[REG_FEATURES].corrwith(pd.Series(y)).abs().sort_values(ascending=False)
        self.corr_ = pd.concat([X[REG_FEATURES], pd.Series(y, name="ln(цена)")], axis=1).corr()
        self.pair_feature_ = corr.index[0]
        inputs = {"pair": X[[self.pair_feature_]], "all": X, "raw": df[PowerLawRegressor.COLS].astype(float)}

        rows, preds = [], {}
        for name, (kind, est) in model_zoo(self.pair_feature_).items():
            progress(f"регрессия: {name}")
            oof = cross_val_predict(est, inputs[kind], y, cv=CV)
            price, pred = np.exp(y), np.exp(oof)
            preds[name] = pred
            rows.append({"метод": name, "MAE, ₽": mean_absolute_error(price, pred),
                         "MSE, ₽²": mean_squared_error(price, pred),
                         "корреляция": float(np.corrcoef(price, pred)[0, 1]),
                         "MAE, %": float(np.mean(np.abs(pred / price - 1)) * 100)})
        self.results_ = pd.DataFrame(rows).sort_values("MAE, ₽").reset_index(drop=True)
        self.best_name_ = self.results_.iloc[0]["метод"]
        kind, est = model_zoo(self.pair_feature_)[self.best_name_]
        self.best_kind_ = kind
        self.model_ = clone(est).fit(inputs[kind], y)
        self.fair_ = dict(zip(df["id"].astype(int), preds[self.best_name_]))
        self.oof_ = pd.DataFrame({"id": df["id"], "price": np.exp(y), "fair": preds[self.best_name_],
                                  "linear": preds["Множественная регрессия"],
                                  "pair": preds[f"Парная регрессия ({self.pair_feature_})"],
                                  "nonlinear": preds["Нелинейная (curve_fit)"], self.pair_feature_: X[self.pair_feature_]})
        self.tree_ = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y)
        self.tree_features_ = list(X.columns)
        self.version_ = PRICE_MODEL_VERSION
        return self

    def save(self, path: Path = PRICE_MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path = PRICE_MODEL_PATH, phones: pd.DataFrame | None = None) -> "PriceModel | None":
        try:
            m = joblib.load(path)
        except Exception:
            return None
        if getattr(m, "version_", 0) != PRICE_MODEL_VERSION:
            return None
        if phones is not None and set(m.fair_) != set(phones["id"].astype(int)):
            return None  # база изменилась — пересчитать
        return m

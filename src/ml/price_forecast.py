"""«Покупать сейчас или ждать?» — прогноз цен как временного ряда (лаба 3, часть про 1 и 7 шагов).

По одной модели телефона в постах обычно 10–30 цен с неравными промежутками — для прогноза мало.
Поэтому прогнозируется ИНДЕКС цен шпиля: для каждого телефона цена делится на его медиану,
затем по каждому дню берётся медиана таких относительных цен; пропуски дней интерполируются.

Признаки (как в лабе): значения ряда за 14 прошлых дней (лаги), скользящее среднее за 7 дней,
первая разность. Прогноз на 1 шаг и на 7 шагов (выход — вектор y(t+1)…y(t+7)).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.clustering import TIER_NAMES

LAGS, HORIZON = 14, 7


def build_index(history: pd.DataFrame, tiers: dict[int, int]) -> pd.DataFrame:
    """Дневной индекс цен по шпилям и по всему рынку (1.0 — «обычная» цена модели)."""
    h = history.copy()
    h["date"] = pd.to_datetime(h["date"])
    h["tier"] = h["phone_id"].map(tiers)
    h = h.dropna(subset=["tier"])
    h["rel"] = h["price_rub"] / h.groupby("phone_id")["price_rub"].transform("median")
    days = pd.date_range(h["date"].min(), h["date"].max(), freq="D")
    out = {"Весь рынок": h.groupby("date")["rel"].median()}
    for t, name in TIER_NAMES.items():
        s = h[h["tier"] == t].groupby("date")["rel"].median()
        if len(s) >= 10:
            out[name] = s
    return pd.DataFrame({k: v.reindex(days).interpolate(limit_direction="both") for k, v in out.items()})


def supervised(series: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    X, Y = [], []
    for t in range(LAGS, len(series) - horizon + 1):
        window = series[t - LAGS:t]
        X.append(np.r_[window, window[-7:].mean(), window[-1] - window[-2]])
        Y.append(series[t:t + horizon])
    return np.array(X), np.array(Y)


def _models(horizon: int) -> dict:
    multi = lambda est: est if horizon == 1 else MultiOutputRegressor(est)
    return {
        "Наивный (последнее значение)": None,
        "Скользящее среднее 7 дней": None,
        "Линейная регрессия": LinearRegression(),
        "Случайный лес": multi(RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=0)),
        "Градиентный бустинг": multi(GradientBoostingRegressor(n_estimators=200, max_depth=2, random_state=0)),
        "Многослойный персептрон": Pipeline([("sc", StandardScaler()),
                                             ("mlp", MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1.0,
                                                                  max_iter=5000, random_state=0))]),
    }


def _predict(name, model, X):
    if name.startswith("Наивный"):
        return np.repeat(X[:, LAGS - 1:LAGS], 1, axis=1)
    if name.startswith("Скользящее"):
        return X[:, LAGS:LAGS + 1]
    p = model.predict(X)
    return p.reshape(len(X), -1)


class PriceForecast:
    def fit(self, history: pd.DataFrame, tiers: dict[int, int]):
        self.index_ = build_index(history, tiers)
        rows, self.forecast_ = [], {}
        for col in self.index_.columns:
            y = self.index_[col].to_numpy()
            if len(y) < LAGS + HORIZON + 10:
                continue
            for horizon in (1, HORIZON):
                X, Y = supervised(y, horizon)
                split = int(len(X) * 0.75)  # время не перемешиваем: обучение на прошлом, тест на будущем
                for name, model in _models(horizon).items():
                    if model is not None:
                        model.fit(X[:split], Y[:split] if horizon > 1 else Y[:split, 0])
                    pred = _predict(name, model, X[split:])
                    if pred.shape[1] == 1 and horizon > 1:
                        pred = np.repeat(pred, horizon, axis=1)
                    err = np.abs(pred - Y[split:])
                    rows.append({"ряд": col, "горизонт": f"{horizon} дн.", "метод": name,
                                 "MAE, п.п.": float(err.mean() * 100), "MAE последнего дня, п.п.": float(err[:, -1].mean() * 100)})
            # прогноз на 7 дней вперёд лучшей моделью для горизонта 7
            res7 = [r for r in rows if r["ряд"] == col and r["горизонт"] == f"{HORIZON} дн."]
            best = min(res7, key=lambda r: r["MAE, п.п."])["метод"]
            X, Y = supervised(y, HORIZON)
            model = _models(HORIZON)[best]
            if model is not None:
                model.fit(X, Y)
            last = np.r_[y[-LAGS:], y[-7:].mean(), y[-1] - y[-2]][None, :]
            fc = _predict(best, model, last)[0]
            if len(fc) == 1:
                fc = np.repeat(fc, HORIZON)
            self.forecast_[col] = {"method": best, "values": fc,
                                   "dates": pd.date_range(self.index_.index[-1] + pd.Timedelta(days=1), periods=HORIZON)}
        self.results_ = pd.DataFrame(rows)
        return self

    def change(self, series: str) -> float | None:
        """Ожидаемое изменение индекса за 7 дней, доли (−0.03 = цены сегмента снизятся на 3%)."""
        fc = self.forecast_.get(series)
        if not fc:
            return None
        return float(fc["values"][-1] / self.index_[series].iloc[-1] - 1)


def advice(phone_history: pd.DataFrame, tier_change: float | None) -> str | None:
    """Подсказка «покупать / ждать» для конкретного телефона."""
    if len(phone_history) < 3:
        if tier_change is None:
            return None
        return (f"Цены сегмента за неделю, по прогнозу, {'снизятся' if tier_change < 0 else 'вырастут'} на "
                f"{abs(tier_change):.0%}." if abs(tier_change) >= 0.01 else "Цены сегмента на неделю — без изменений.")
    prices = phone_history["price_rub"].to_numpy()
    last, med = prices[-1], np.median(prices)
    if last <= med * 0.97:
        return f"Хорошая цена сейчас: на {1 - last / med:.0%} ниже обычной для этой модели."
    if tier_change is not None and tier_change <= -0.02:
        return f"Можно подождать: цены сегмента за неделю, по прогнозу, снизятся на {abs(tier_change):.0%}."
    if last >= med * 1.05:
        return f"Сейчас дороже обычного на {last / med - 1:.0%} — если не срочно, стоит подождать."
    return "Цена обычная для этой модели."

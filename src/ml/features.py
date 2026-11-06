"""Признаки телефонов: нормализация, веса признаков и PCA.

Веса нужны, чтобы при кластеризации признаки «не перетягивали друг друга»:
без них цена, RAM и AnTuTu (сильно коррелируют) задают почти всё расстояние,
и кластеры получаются просто «дешёвые / дорогие».
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif

from src.preprocess import StandardScalerCustom

PHONE_FEATURES = ["price_rub", "ram_gb", "rom_gb", "battery_mah", "screen_in", "refresh_hz",
                  "weight_g", "camera_mp", "antutu_k", "charge_w", "has_5g", "nfc", "esim",
                  "is_ios", "year", "ir_port", "active_h", "standby_d"]

FEATURE_NAMES_RU = {
    "price_rub": "цена", "ram_gb": "ОЗУ", "rom_gb": "память", "battery_mah": "батарея",
    "screen_in": "экран", "refresh_hz": "частота экрана", "weight_g": "вес",
    "camera_mp": "камера", "antutu_k": "производительность", "charge_w": "быстрая зарядка",
    "has_5g": "5G", "nfc": "NFC", "esim": "eSIM", "is_ios": "iOS", "year": "новизна", "ir_port": "ИК-порт",
    "active_h": "автономность (активно)", "standby_d": "автономность (ожидание)",
}

# Экспертные веса: что реально отличает телефоны по «характеру».
EXPERT_WEIGHTS = {
    "price_rub": 1.0, "ram_gb": 0.6, "rom_gb": 0.5, "battery_mah": 0.5, "screen_in": 0.9,
    "refresh_hz": 0.6, "weight_g": 0.9, "camera_mp": 0.9, "antutu_k": 1.0, "charge_w": 0.6,
    "has_5g": 0.3, "nfc": 0.3, "esim": 0.3, "is_ios": 0.7, "year": 0.4, "ir_port": 0.3,
    "active_h": 0.9, "standby_d": 0.6,
}

# Признаки, к которым применяется логарифм (сильно скошенные распределения).
LOG_FEATURES = ["price_rub", "ram_gb", "rom_gb", "battery_mah", "antutu_k", "charge_w", "weight_g"]


def phone_matrix(df: pd.DataFrame) -> np.ndarray:
    data = df.copy()
    data["is_ios"] = (data["os"] == "iOS").astype(float)
    for c in LOG_FEATURES:
        data[c] = np.log1p(data[c].astype(float))
    return data[PHONE_FEATURES].to_numpy(dtype=np.float64)


class PhoneFeatureSpace:
    """StandardScaler (свой, из курса) + веса признаков + PCA для визуализации."""

    def __init__(self, use_weights: bool = True):
        self.use_weights = use_weights
        self.scaler = StandardScalerCustom()
        self.weights_ = np.ones(len(PHONE_FEATURES))
        self.mi_ = None
        self.pca2_ = None

    def fit(self, df: pd.DataFrame, labels: np.ndarray | None = None):
        z = self.scaler.fit_transform(phone_matrix(df))
        if self.use_weights:
            expert = np.array([EXPERT_WEIGHTS[f] for f in PHONE_FEATURES])
            if labels is not None:
                mi = mutual_info_classif(z, labels, random_state=0)
                self.mi_ = mi
                # Информативность признака относительно меток гасится:
                # признаки, которые сильно объясняют метку класса (цена/мощность),
                # уже учтены в метке — в кластеризации им достаточно меньшего веса.
                damp = 1.0 / (1.0 + 2.0 * mi / (mi.max() + 1e-9))
                w = expert * damp
            else:
                w = expert
            with np.errstate(invalid="ignore", divide="ignore"):  # постоянный признак (например, нет iOS) даёт NaN
                corr = np.nan_to_num(np.abs(np.corrcoef(z, rowvar=False)))
            np.fill_diagonal(corr, 0)
            # штраф за избыточность: сильно коррелирующие признаки делят вес между собой
            excess = np.where(corr >= 0.6, corr - 0.6, 0.0).sum(axis=1)
            w = w / (1.0 + 2.0 * excess)
            self.weights_ = w / w.mean()
        self.pca2_ = PCA(n_components=2, random_state=0).fit(self.transform(df))
        return self

    def standardize(self, df: pd.DataFrame) -> np.ndarray:
        return self.scaler._transform(phone_matrix(df))

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.standardize(df) * np.sqrt(self.weights_)

    def to_2d(self, x: np.ndarray) -> np.ndarray:
        return self.pca2_.transform(x)

    def weights_table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "признак": [FEATURE_NAMES_RU[f] for f in PHONE_FEATURES],
            "вес": self.weights_.round(3),
            "MI с меткой": (self.mi_.round(3) if self.mi_ is not None else np.nan),
        }).sort_values("вес", ascending=False)

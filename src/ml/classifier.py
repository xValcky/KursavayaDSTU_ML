"""Полный ML-конвейер: метки → веса признаков → кластеризация → нейросеть-классификатор.

Схема:
  телефоны → метки [0,1,2] → взвешенное пространство признаков → KMeans
  анкета → (признаки анкеты + признаки кластеров) → MLP → метка класса [0,1,2]
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from src.ml import clustering
from src.ml.ensemble import make_mlp
from src.ml.features import PhoneFeatureSpace
from src.ml.labeling import generate_dataset, phone_labels, survey_vector, utility
from src.paths import MODELS_DIR
from src.preprocess import split_train_val_test

MODEL_PATH = MODELS_DIR / "pipeline.joblib"
# меняется при изменении признаков/анкеты: старые сохранённые модели не загружаются
MODEL_VERSION = 3


class PhonePipeline:
    def __init__(self, n_surveys: int = 6000, mlp_params: dict | None = None, seed: int = 0,
                 k: int | None = None):
        self.n_surveys = n_surveys
        self.mlp_params = mlp_params or {}
        self.k = k  # число кластеров; None — подобрать по силуэту
        self.seed = seed

    # ---------- признаки анкеты ----------
    def cluster_features(self, a: dict) -> np.ndarray:
        """Для каждого кластера: лучшая полезность телефона из кластера и доля доступных по бюджету."""
        u = utility(a, self.z_, self.phones_)
        affordable = self.phones_["price_rub"].to_numpy() <= a["budget_rub"]
        best = [u[self.clusters_ == c].max() for c in range(self.k_)]
        share = [affordable[self.clusters_ == c].mean() for c in range(self.k_)]
        return np.array(best + share)

    def features(self, surveys: list[dict], with_clusters: bool = True) -> np.ndarray:
        base = np.array([survey_vector(a) for a in surveys])
        if not with_clusters:
            return base
        return np.hstack([base, np.array([self.cluster_features(a) for a in surveys])])

    # ---------- обучение ----------
    def fit(self, phones: pd.DataFrame, feedback: list[tuple[dict, int]] | None = None):
        self.phones_ = phones.reset_index(drop=True)
        self.labels_ = phone_labels(self.phones_)

        self.space_ = PhoneFeatureSpace(use_weights=True).fit(self.phones_, self.labels_)
        self.z_ = self.space_.standardize(self.phones_)
        self.x_ = self.space_.transform(self.phones_)

        self.k_ = self.k or clustering.choose_k(self.x_)
        self.kmeans_ = clustering.make_kmeans(self.k_).fit(self.x_)
        self.clusters_ = self.kmeans_.labels_
        self.cluster_names_ = clustering.cluster_names(self.z_, self.clusters_)

        surveys, y = generate_dataset(self.n_surveys, self.z_, self.phones_, self.labels_, self.seed)
        for answers, phone_id in feedback or []:
            match = np.flatnonzero(self.phones_["id"].to_numpy() == phone_id)
            if len(match):
                # понравившийся телефон — сильный сигнал, добавляем с повтором
                surveys += [answers] * 5
                y = np.append(y, [self.labels_[match[0]]] * 5)
        self.n_feedback_ = len(feedback or [])

        X = self.features(surveys)
        perm = np.random.default_rng(self.seed).permutation(len(X))
        X, y = X[perm], y[perm]
        idx_tr, idx_val, idx_te = split_train_val_test(np.arange(len(X)), (70, 15, 15), percent=True)

        self.model_ = make_mlp(**self.mlp_params).fit(X[idx_tr], y[idx_tr])
        mlp = self.model_.named_steps["clf"]
        self.loss_curve_ = list(mlp.loss_curve_)
        self.val_curve_ = list(mlp.validation_scores_ or [])

        self.metrics_ = {}
        for part, idx in (("val", idx_val), ("test", idx_te)):
            pred = self.model_.predict(X[idx])
            proba = self.model_.predict_proba(X[idx])
            self.metrics_[part] = {
                "accuracy": accuracy_score(y[idx], pred),
                "f1_macro": f1_score(y[idx], pred, average="macro"),
                "roc_auc_ovr": roc_auc_score(y[idx], proba, multi_class="ovr"),
            }
        self.confusion_ = confusion_matrix(y[idx_te], self.model_.predict(X[idx_te]))
        self.class_balance_ = np.bincount(y, minlength=3)
        self.X_, self.y_ = X, y
        return self

    def predict_proba(self, a: dict) -> np.ndarray:
        return self.model_.predict_proba(self.features([a]))[0]

    # ---------- сохранение ----------
    def save(self, path: Path = MODEL_PATH):
        self.version_ = MODEL_VERSION
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path = MODEL_PATH) -> "PhonePipeline | None":
        try:
            pipe = joblib.load(path)
        except Exception:
            return None
        return pipe if getattr(pipe, "version_", None) == MODEL_VERSION else None

"""Полный ML-конвейер: метки → веса признаков → кластеризация по шпилям → нейросеть-классификатор.

Схема:
  телефоны → метки [0,1,2] (ценовые «шпили») → взвешенное пространство признаков
           → «броневики» отдельно по правилу → KMeans внутри каждого шпиля (k по силуэту)
  анкета → (признаки анкеты + признаки кластеров) → MLP → метка класса [0,1,2]
"""
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, silhouette_score

from src.ml import clustering
from src.ml.ensemble import make_mlp
from src.ml.features import PhoneFeatureSpace, is_rugged
from src.ml.labeling import generate_dataset, phone_labels, survey_vector, utility
from src.paths import MODELS_DIR
from src.preprocess import split_train_val_test

MODEL_PATH = MODELS_DIR / "pipeline.joblib"
REGISTRY_PATH = MODELS_DIR / "registry.json"
# меняется при изменении признаков/анкеты: старые сохранённые модели не загружаются
MODEL_VERSION = 5
RUGGED_NAME = "Защищённые «броневики»"


class PhonePipeline:
    def __init__(self, n_surveys: int = 6000, mlp_params: dict | None = None, seed: int = 0,
                 k: int | None = None):
        self.n_surveys = n_surveys
        self.mlp_params = mlp_params or {}
        self.k = k  # число групп в каждом шпиле; None — подобрать по силуэту
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

    # ---------- кластеризация по шпилям ----------
    def _cluster_by_tiers(self):
        """Отдельный KMeans в каждом ценовом шпиле; «броневики» — своя группа по правилу.

        На всей базе KMeans склеивал 62% моделей в одну группу; внутри шпиля группы ровнее.
        """
        n = len(self.phones_)
        self.rugged_ = is_rugged(self.phones_)
        self.clusters_ = np.full(n, -1)
        self.cluster_names_, self.cluster_tier_, self.tier_stats_ = {}, {}, []
        gid = 0
        for tier, tier_name in clustering.TIER_NAMES.items():
            idx = np.flatnonzero((self.labels_ == tier) & ~self.rugged_)
            if len(idx) == 0:
                continue
            x = self.x_[idx]
            k = min(self.k, len(idx) // 2) if self.k else clustering.choose_k(x)
            k = max(1, k)
            lab = clustering.make_kmeans(k).fit_predict(x) if k > 1 else np.zeros(len(idx), int)
            # имена — по отличию от «среднего телефона своего шпиля», иначе в бюджете все «бюджетные»
            z_local = self.z_[idx] - self.z_[idx].mean(axis=0)
            names = clustering.cluster_names(z_local, lab)
            for c in range(k):
                self.clusters_[idx[lab == c]] = gid + c
                self.cluster_names_[gid + c] = f"{tier_name} · {names[c]}"
                self.cluster_tier_[gid + c] = tier
            self.tier_stats_.append({
                "шпиль": tier_name, "моделей": len(idx), "k": k,
                "silhouette": round(float(silhouette_score(x, lab)), 3) if k > 1 else np.nan,
                "ARI по 10 сидам": round(clustering.stability(x, k), 3) if k > 1 else np.nan,
                "мин. группа": int(np.bincount(lab).min()), "макс. группа": int(np.bincount(lab).max()),
            })
            gid += k
        if self.rugged_.any():
            self.clusters_[self.rugged_] = gid
            self.cluster_names_[gid] = RUGGED_NAME
            self.cluster_tier_[gid] = -1
            self.tier_stats_.append({"шпиль": "Броневики (правило: вес ≥ 300 г)", "моделей": int(self.rugged_.sum()),
                                     "k": 1, "silhouette": np.nan, "ARI по 10 сидам": np.nan,
                                     "мин. группа": int(self.rugged_.sum()), "макс. группа": int(self.rugged_.sum())})
            gid += 1
        self.k_ = gid
        self.centers_ = np.array([self.x_[self.clusters_ == c].mean(axis=0) for c in range(self.k_)])

    # ---------- обучение ----------
    def fit(self, phones: pd.DataFrame, feedback: list[tuple[dict, int]] | None = None,
            dislikes: list[tuple[dict, int]] | None = None):
        self.phones_ = phones.reset_index(drop=True)
        self.labels_ = phone_labels(self.phones_)

        self.space_ = PhoneFeatureSpace(use_weights=True).fit(self.phones_, self.labels_)
        self.z_ = self.space_.standardize(self.phones_)
        self.x_ = self.space_.transform(self.phones_)
        self._cluster_by_tiers()

        surveys, y = generate_dataset(self.n_surveys, self.z_, self.phones_, self.labels_, self.seed)
        ids = self.phones_["id"].to_numpy()
        for answers, phone_id in feedback or []:
            match = np.flatnonzero(ids == phone_id)
            if len(match):
                # понравившийся телефон — сильный сигнал, добавляем с повтором
                surveys += [answers] * 5
                y = np.append(y, [self.labels_[match[0]]] * 5)
        for answers, phone_id in dislikes or []:
            # «не подходит»: правильный ответ — класс лучшего телефона БЕЗ отвергнутого
            u = utility(answers, self.z_, self.phones_)
            u[ids == phone_id] = -np.inf
            surveys += [answers] * 3
            y = np.append(y, [self.labels_[int(np.argmax(u))]] * 3)
        self.n_feedback_ = len(feedback or []) + len(dislikes or [])

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
        self.trained_at_ = datetime.now().isoformat(timespec="seconds")
        return self

    def predict_proba(self, a: dict) -> np.ndarray:
        return self.model_.predict_proba(self.features([a]))[0]

    # ---------- сохранение ----------
    def save(self, path: Path = MODEL_PATH):
        self.version_ = MODEL_VERSION
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        self._register(path)

    def _register(self, path: Path):
        """Реестр моделей: что, когда и с какими метриками обучено."""
        try:
            reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8")) if REGISTRY_PATH.exists() else []
        except Exception:
            reg = []
        t = self.metrics_["test"]
        reg.append({"file": path.name, "version": MODEL_VERSION, "trained_at": getattr(self, "trained_at_", ""),
                    "saved_at": datetime.now().isoformat(timespec="seconds"), "phones": len(self.phones_),
                    "clusters": self.k_, "surveys": self.n_surveys, "feedback": self.n_feedback_,
                    "accuracy": round(t["accuracy"], 4), "f1_macro": round(t["f1_macro"], 4),
                    "roc_auc": round(t["roc_auc_ovr"], 4)})
        REGISTRY_PATH.write_text(json.dumps(reg[-50:], ensure_ascii=False, indent=1), encoding="utf-8")

    @staticmethod
    def load(path: Path = MODEL_PATH) -> "PhonePipeline | None":
        try:
            pipe = joblib.load(path)
        except Exception:
            return None
        return pipe if getattr(pipe, "version_", None) == MODEL_VERSION else None

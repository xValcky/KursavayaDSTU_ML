"""Итоговый подбор: класс от нейросети + кластер + полезность телефона для анкеты.

Плюс поиск похожих телефонов (ближайшие соседи в пространстве признаков) и
объяснение «почему этот телефон» (вклад каждого признака в оценку).
"""
import numpy as np
import pandas as pd

from src.ml.classifier import PhonePipeline
from src.ml.features import FEATURE_NAMES_RU, PHONE_FEATURES
from src.ml.labeling import CLASS_NAMES, preference_weights, utility

_IDX = {f: i for i, f in enumerate(PHONE_FEATURES)}

# Группы признаков для «похожих по …»
SIMILAR_BY = {
    "all": ("всем параметрам", None),
    "camera": ("камере", ["camera_mp", "rom_gb"]),
    "battery": ("автономности", ["battery_mah", "active_h", "standby_d", "charge_w"]),
    "performance": ("производительности", ["antutu_k", "ram_gb", "refresh_hz"]),
    "screen": ("экрану и размеру", ["screen_in", "refresh_hz", "weight_g"]),
    "price": ("цене", ["price_rub"]),
    "cheaper": ("всему, но дешевле", None),
}


def full_name(brand: str, model: str) -> str:
    """«Honor» + «Honor 400» → «Honor 400», «Xiaomi» + «Redmi 15» → «Xiaomi Redmi 15»."""
    return model if model.lower().startswith(brand.lower()) else f"{brand} {model}"


def plain_pros(p: pd.Series, top_features: list[str]) -> list[str]:
    """Плюсы телефона простым языком — для карточек в простом режиме."""
    phrases = {
        "camera_mp": f"камера {p['camera_mp']:.0f} МП",
        "active_h": f"≈{p['active_h']:.0f} ч активной работы",
        "standby_d": f"≈{p['standby_d']:.0f} дней в ожидании",
        "battery_mah": f"батарея {p['battery_mah']:.0f} мА·ч",
        "antutu_k": f"мощный чип (AnTuTu {p['antutu_k']:.0f} тыс.)",
        "refresh_hz": f"плавный экран {p['refresh_hz']:.0f} Гц",
        "charge_w": f"зарядка {p['charge_w']:.0f} Вт",
        "rom_gb": f"память {p['rom_gb']:.0f} ГБ",
        "ram_gb": f"ОЗУ {p['ram_gb']:.0f} ГБ",
        "screen_in": f"экран {p['screen_in']:g}″",
        "weight_g": f"лёгкий, {p['weight_g']:.0f} г",
        "year": f"модель {p['year']:.0f} года",
        "esim": "есть eSIM", "ir_port": "есть ИК-порт (пульт)", "nfc": "есть NFC", "has_5g": "есть 5G",
    }
    return [phrases[f] for f in top_features if f in phrases][:3] or ["сбалансированный вариант"]


def score_phones(pipe: PhonePipeline, a: dict) -> tuple[np.ndarray, np.ndarray, int]:
    proba = pipe.predict_proba(a)
    cf = pipe.cluster_features(a)
    best_cluster = int(np.argmax(cf[:pipe.k_]))
    u = utility(a, pipe.z_, pipe.phones_)
    u_norm = (u - u.mean()) / (u.std() + 1e-9)
    # согласие с нейросетью: телефон предсказанного класса получает бонус по вероятности
    score = u_norm + 1.5 * proba[pipe.labels_] + 0.4 * (pipe.clusters_ == best_cluster)
    return score, proba, best_cluster


def recommend(pipe: PhonePipeline, a: dict, top_n: int = 10) -> dict:
    score, proba, best_cluster = score_phones(pipe, a)
    pred_class = int(np.argmax(proba))
    phones = pipe.phones_
    order = np.argsort(-score)[:top_n]
    contrib = pipe.z_ * preference_weights(a)
    rows = []
    for i in order:
        p = phones.iloc[i]
        top = [PHONE_FEATURES[j] for j in np.argsort(-contrib[i])[:4] if contrib[i, j] > 0]
        reasons = ", ".join(FEATURE_NAMES_RU[f] for f in top[:3])
        if p["price_rub"] > a["budget_rub"]:
            reasons += f"; дороже бюджета на {p['price_rub'] - a['budget_rub']:,.0f} ₽".replace(",", " ")
        rows.append({
            "id": int(p["id"]), "Телефон": full_name(p["brand"], p["model"]),
            "Цена, ₽": int(p["price_rub"]), "Класс": CLASS_NAMES[int(pipe.labels_[i])],
            "Кластер": pipe.cluster_names_[int(pipe.clusters_[i])],
            "ОЗУ/память": f"{p['ram_gb']:.0f}/{p['rom_gb']:.0f} ГБ",
            "Экран": f"{p['screen_in']:g}″ · {p['refresh_hz']:.0f} Гц",
            "Активно, ч": float(p["active_h"]), "Ожидание, дн": float(p["standby_d"]),
            "Камера, МП": int(p["camera_mp"]),
            "AnTuTu, тыс.": int(p["antutu_k"]), "eSIM": int(p["esim"]), "ИК": int(p["ir_port"]),
            "Оценка": round(float(score[i]), 2),
            "Почему": reasons or "сбалансированный вариант",
            "_pros": plain_pros(p, top),
        })
    return {
        "proba": proba, "pred_class": pred_class, "class_name": CLASS_NAMES[pred_class],
        "best_cluster": best_cluster, "cluster_name": pipe.cluster_names_[best_cluster],
        "table": pd.DataFrame(rows),
    }


def explain(pipe: PhonePipeline, a: dict, phone_id: int) -> pd.Series:
    """Вклад каждого признака в оценку телефона для этой анкеты («Почему он?»)."""
    i = int(np.flatnonzero(pipe.phones_["id"].to_numpy() == phone_id)[0])
    contrib = pipe.z_[i] * preference_weights(a)
    s = pd.Series(contrib, index=[FEATURE_NAMES_RU[f] for f in PHONE_FEATURES])
    return s[s.abs() > 0.01].sort_values()


def similar_phones(pipe: PhonePipeline, phone_id: int, by: str = "all", n: int = 6) -> pd.DataFrame:
    """Ближайшие соседи телефона: по всем признакам (взвешенное пространство кластеризации)
    или только по группе признаков; для групп — в ценовом коридоре ±30%."""
    phones = pipe.phones_
    i = int(np.flatnonzero(phones["id"].to_numpy() == phone_id)[0])
    price = phones["price_rub"].to_numpy()
    cols = SIMILAR_BY[by][1]
    overall = np.linalg.norm(pipe.x_ - pipe.x_[i], axis=1)
    if cols is None:
        base = overall
    else:
        z = pipe.z_[:, [_IDX[c] for c in cols]]
        base = np.linalg.norm(z - z[i], axis=1)
    # штрафы влияют только на порядок, а «сходство» в процентах считается по base
    rank = base + 0.05 * overall - (0.3 * (pipe.clusters_ == pipe.clusters_[i]) if cols is None else 0)
    if cols is not None and by != "price":
        rank = rank + 10.0 * (np.abs(price / price[i] - 1) > 0.3)
    if by == "cheaper":
        rank = rank + 100.0 * (price >= price[i] * 0.97)
    rank[i] = np.inf
    order = np.argsort(rank)[:n]
    scale = np.median(base) + 1e-9
    dist = base
    rows = []
    for j in order:
        p = phones.iloc[j]
        rows.append({"id": int(p["id"]), "Телефон": full_name(p["brand"], p["model"]),
                     "Цена, ₽": int(p["price_rub"]),
                     "Сходство": f"{np.exp(-dist[j] / scale):.0%}",
                     "Кластер": pipe.cluster_names_[int(pipe.clusters_[j])],
                     "Разница в цене": f"{p['price_rub'] - price[i]:+,.0f} ₽".replace(",", " ")})
    return pd.DataFrame(rows)


def phone_row(pipe: PhonePipeline, phone_id: int) -> pd.Series:
    i = int(np.flatnonzero(pipe.phones_["id"].to_numpy() == phone_id)[0])
    p = pipe.phones_.iloc[i].copy()
    p["class_name"] = CLASS_NAMES[int(pipe.labels_[i])]
    p["cluster_name"] = pipe.cluster_names_[int(pipe.clusters_[i])]
    p["full_name"] = full_name(p["brand"], p["model"])
    return p

"""Итоговый подбор: класс от нейросети + мягкий бонус кластера + полезность телефона для анкеты.

Плюс: разнообразие выдачи, блоки («лучший / дешевле / премиум / другая группа»), подсказки при
слабой выдаче, поиск похожих и объяснение «почему этот телефон».
"""
import numpy as np
import pandas as pd

from src.ml.classifier import PhonePipeline
from src.ml.features import FEATURE_NAMES_RU, PHONE_FEATURES, is_rugged
from src.ml.labeling import (CLASS_NAMES, REQUIREMENT_TITLES, feasible_mask, penalty_breakdown,
                             preference_weights, utility)

_IDX = {f: i for i, f in enumerate(PHONE_FEATURES)}
CLUSTER_BONUS = 0.4       # максимум мягкого бонуса кластера
CLASS_BONUS = 1.5         # вес вероятности класса от нейросети
MAX_SAME_BRAND_TOP5 = 2
MIN_GROUPS_TOP10 = 3
DEAL_THRESHOLD, OVERPRICE_THRESHOLD = 0.15, 0.20

# Группы признаков для «похожих по …»
SIMILAR_BY = {
    "all": ("всем параметрам", None),
    "camera": ("камере", ["camera_mp", "rom_gb"]),
    "battery": ("автономности", ["battery_mah", "active_h", "standby_d", "charge_w"]),
    "performance": ("производительности", ["antutu_k", "ram_gb", "refresh_hz"]),
    "screen": ("экрану и размеру", ["screen_in", "refresh_hz", "weight_g"]),
    "price": ("цене", ["price_rub"]),
    "cheaper": ("всему, но дешевле", None),
    "undervalued": ("недооценённые (цена ниже справедливой)", None),
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


def price_badges(p: pd.Series, fair: float | None = None) -> list[str]:
    """Значки для карточки: откуда цена, «выгодно / переплата», «броневик»."""
    badges = []
    src = str(p.get("price_source", ""))
    if src.startswith("Telegram"):
        badges.append("цена из канала, без пошлины")
    elif src.startswith("маркетплейсы"):
        badges.append("цена — оценка")
    if fair:
        diff = p["price_rub"] / fair - 1
        # пороги шире средней ошибки модели (≈16%), чтобы значок был не случайным
        if diff <= -DEAL_THRESHOLD:
            badges.append(f"выгодно {diff:+.0%}")
        elif diff >= OVERPRICE_THRESHOLD:
            badges.append(f"переплата {diff:+.0%}")
    if p["weight_g"] >= 300:
        badges.append("защищённый")
    return badges


# ---------- оценка ----------
def soft_cluster_bonus(pipe: PhonePipeline, best_u: np.ndarray, T: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Мягкий бонус вместо «+0.4 всё или ничего».

    p_ij — насколько телефон i похож на центр кластера j (softmax по расстояниям);
    a_j  — насколько кластер j подходит анкете (softmax по лучшей полезности внутри кластера).
    bonus_i = 0.4 · Σ_j p_ij a_j / max(a): телефон «в центре» лучшего кластера получает почти 0.4,
    телефон на границе двух кластеров — промежуточное значение, без обрыва.
    """
    d2 = ((pipe.x_[:, None, :] - pipe.centers_[None, :, :]) ** 2).sum(-1)
    tau = np.median(d2.min(axis=1)) + 1e-9
    p = np.exp(-(d2 - d2.min(axis=1, keepdims=True)) / tau)
    p /= p.sum(axis=1, keepdims=True)
    a = np.exp((best_u - best_u.max()) / T)
    a /= a.sum()
    return CLUSTER_BONUS * (p @ a) / a.max(), a


def score_phones(pipe: PhonePipeline, a: dict) -> dict:
    proba = pipe.predict_proba(a)
    u = utility(a, pipe.z_, pipe.phones_)
    feas = feasible_mask(a, pipe.phones_)
    # нормируем по подходящим телефонам: иначе огромные штрафы «дороже бюджета» у остальных
    # раздувают разброс, и разница камер/батарей между подходящими почти не видна
    ref = u[feas] if feas.sum() >= 5 else u
    u_norm = (u - ref.mean()) / (ref.std() + 1e-9)
    best_u = np.array([u_norm[pipe.clusters_ == c].max() for c in range(pipe.k_)])
    bonus, affinity = soft_cluster_bonus(pipe, best_u)
    score = u_norm + CLASS_BONUS * proba[pipe.labels_] + bonus
    return {"score": score, "proba": proba, "u_norm": u_norm, "bonus": bonus, "feasible": feas,
            "best_cluster": int(np.argmax(affinity)), "affinity": affinity}


def diversify(order: np.ndarray, brands: np.ndarray, clusters: np.ndarray, n: int) -> list[int]:
    """Разнообразие: не больше 2 моделей одного бренда в топ-5 и минимум 3 группы в топ-10."""
    chosen, rest = [], list(order)
    while rest and len(chosen) < n:
        pick = None
        for i in rest:
            if len(chosen) < 5 and sum(brands[j] == brands[i] for j in chosen) >= MAX_SAME_BRAND_TOP5:
                continue
            groups = {clusters[j] for j in chosen}
            if 7 <= len(chosen) < 10 and len(groups) < MIN_GROUPS_TOP10 and clusters[i] in groups:
                continue
            pick = i
            break
        pick = rest[0] if pick is None else pick
        chosen.append(pick)
        rest.remove(pick)
    return chosen


def weak_result_hint(a: dict, phones: pd.DataFrame, feas: np.ndarray) -> dict | None:
    """Если подходящих телефонов мало — какое требование отсекло больше всего моделей в бюджете."""
    if feas.sum() >= 8:
        return None
    in_budget = phones["price_rub"].to_numpy() <= a["budget_rub"] * 1.1
    pen = penalty_breakdown(a, phones).drop(columns="budget")
    cut = {k: int(((pen[k] > 0) & in_budget).sum()) for k in pen.columns}
    key = max(cut, key=cut.get)
    if cut[key] == 0:
        cheapest = phones.loc[feasible_mask(a, phones, slack=99), "price_rub"]
        return {"key": "budget", "count": int(feas.sum()),
                "text": f"Подходящих телефонов всего {int(feas.sum())}: мал бюджет. "
                        f"Самый дешёвый подходящий — {cheapest.min():,.0f} ₽.".replace(",", " ")
                if len(cheapest) else "Под эти требования телефонов нет."}
    return {"key": key, "count": int(feas.sum()),
            "text": f"Подходящих телефонов всего {int(feas.sum())}. Больше всего моделей в бюджете отсекло "
                    f"требование «{REQUIREMENT_TITLES[key]}» ({cut[key]} шт.)."}


def _row(pipe, i, score, contrib, a, fair):
    p = pipe.phones_.iloc[i]
    top = [PHONE_FEATURES[j] for j in np.argsort(-contrib[i])[:4] if contrib[i, j] > 0]
    reasons = ", ".join(FEATURE_NAMES_RU[f] for f in top[:3])
    if p["price_rub"] > a["budget_rub"]:
        reasons += f"; дороже бюджета на {p['price_rub'] - a['budget_rub']:,.0f} ₽".replace(",", " ")
    fair_price = fair.get(int(p["id"])) if fair is not None else None
    return {
        "id": int(p["id"]), "Телефон": full_name(p["brand"], p["model"]),
        "Цена, ₽": int(p["price_rub"]),
        "Справедливая, ₽": int(round(fair_price, -2)) if fair_price else np.nan,
        "Класс": CLASS_NAMES[int(pipe.labels_[i])],
        "Кластер": pipe.cluster_names_[int(pipe.clusters_[i])],
        "ОЗУ/память": f"{p['ram_gb']:.0f}/{p['rom_gb']:.0f} ГБ",
        "Экран": f"{p['screen_in']:g}″ · {p['refresh_hz']:.0f} Гц",
        "Активно, ч": float(p["active_h"]), "Ожидание, дн": float(p["standby_d"]),
        "Камера, МП": int(p["camera_mp"]),
        "AnTuTu, тыс.": int(p["antutu_k"]), "eSIM": int(p["esim"]), "ИК": int(p["ir_port"]),
        "Оценка": round(float(score[i]), 2),
        "Почему": reasons or "сбалансированный вариант",
        "_pros": plain_pros(p, top),
        "_badges": price_badges(p, fair_price),
    }


def recommend(pipe: PhonePipeline, a: dict, top_n: int = 10, fair: dict | None = None,
              diverse: bool = True) -> dict:
    s = score_phones(pipe, a)
    score, proba = s["score"], s["proba"]
    pred_class = int(np.argmax(proba))
    phones = pipe.phones_
    order = np.argsort(-score)
    brands, clusters = phones["brand"].to_numpy(), pipe.clusters_
    picked = diversify(order, brands, clusters, top_n) if diverse else list(order[:top_n])
    contrib = pipe.z_ * preference_weights(a)
    table = pd.DataFrame([_row(pipe, i, score, contrib, a, fair) for i in picked])
    return {
        "proba": proba, "pred_class": pred_class, "class_name": CLASS_NAMES[pred_class],
        "best_cluster": s["best_cluster"], "cluster_name": pipe.cluster_names_[s["best_cluster"]],
        "table": table, "blocks": result_blocks(pipe, a, score, picked, contrib, fair),
        "hint": weak_result_hint(a, phones, s["feasible"]), "feasible": int(s["feasible"].sum()),
    }


def result_blocks(pipe, a, score, picked, contrib, fair) -> list[dict]:
    """Выдача блоками: лучший выбор, дешевле с тем же характером, премиум-альтернатива, другая группа."""
    phones = pipe.phones_
    price = phones["price_rub"].to_numpy()
    best = picked[0]
    used = {best}
    blocks = [{"title": "Лучший выбор", "rows": [_row(pipe, best, score, contrib, a, fair)]}]

    def take(mask, n, title, note):
        cand = [i for i in np.argsort(-score) if mask[i] and i not in used][:n]
        if cand:
            used.update(cand)
            blocks.append({"title": title, "note": note,
                           "rows": [_row(pipe, i, score, contrib, a, fair) for i in cand]})

    same = pipe.clusters_ == pipe.clusters_[best]
    take(same & (price <= price[best] * 0.85), 2, "Дешевле с тем же характером",
         "та же группа, цена ниже минимум на 15%")
    take((price > price[best]) & (price <= a["budget_rub"] * 1.25) & (pipe.labels_ >= pipe.labels_[best]), 2,
         "Премиум-альтернатива", "дороже, но до +25% к бюджету")
    top3_groups = {pipe.clusters_[i] for i in picked[:3]}
    take(~np.isin(pipe.clusters_, list(top3_groups)) & (price <= a["budget_rub"] * 1.1), 2,
         "Из другой группы", "другой характер железа")
    return blocks


def explain(pipe: PhonePipeline, a: dict, phone_id: int) -> pd.Series:
    """Вклад каждого признака в оценку телефона для этой анкеты («Почему он?»)."""
    i = int(np.flatnonzero(pipe.phones_["id"].to_numpy() == phone_id)[0])
    contrib = pipe.z_[i] * preference_weights(a)
    s = pd.Series(contrib, index=[FEATURE_NAMES_RU[f] for f in PHONE_FEATURES])
    rugged = a.get("imp_rugged", 0) / 10 * 1.2 * float(is_rugged(pipe.phones_.iloc[[i]])[0])
    if rugged:
        s["защищённый корпус"] = rugged
    return s[s.abs() > 0.01].sort_values()


def similar_phones(pipe: PhonePipeline, phone_id: int, by: str = "all", n: int = 6,
                   fair: dict | None = None) -> pd.DataFrame:
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
    if by == "undervalued" and fair:
        ratio = np.array([price[j] / fair.get(int(pid), price[j]) for j, pid in enumerate(phones["id"])])
        rank = rank + 3.0 * (ratio - 1) + 10.0 * (np.abs(price / price[i] - 1) > 0.35)
    rank[i] = np.inf
    order = np.argsort(rank)[:n]
    scale = np.median(base) + 1e-9
    rows = []
    for j in order:
        p = phones.iloc[j]
        f = fair.get(int(p["id"])) if fair else None
        rows.append({"id": int(p["id"]), "Телефон": full_name(p["brand"], p["model"]),
                     "Цена, ₽": int(p["price_rub"]),
                     "Сходство": f"{np.exp(-base[j] / scale):.0%}",
                     "Кластер": pipe.cluster_names_[int(pipe.clusters_[j])],
                     "Разница в цене": f"{p['price_rub'] - price[i]:+,.0f} ₽".replace(",", " "),
                     "К справедливой": f"{p['price_rub'] / f - 1:+.0%}" if f else ""})
    return pd.DataFrame(rows)


def phone_row(pipe: PhonePipeline, phone_id: int, fair: dict | None = None) -> pd.Series:
    i = int(np.flatnonzero(pipe.phones_["id"].to_numpy() == phone_id)[0])
    p = pipe.phones_.iloc[i].copy()
    p["class_name"] = CLASS_NAMES[int(pipe.labels_[i])]
    p["cluster_name"] = pipe.cluster_names_[int(pipe.clusters_[i])]
    p["full_name"] = full_name(p["brand"], p["model"])
    p["badges"] = price_badges(p, fair.get(int(p["id"])) if fair else None)
    p["fair_price"] = fair.get(int(p["id"])) if fair else None
    return p

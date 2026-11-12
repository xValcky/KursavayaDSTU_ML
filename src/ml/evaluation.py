"""Оценка качества ранжирования на эталонных запросах: NDCG@k.

Эталон — docs/reference_queries.json: «анкета → какие модели эксперт считает хорошими ответами»
(relevance 3 — идеально, 2 — хорошо, 1 — допустимо). Размечает ответственный за функционал.
Запуск:  python -m src.ml.evaluation
"""
import json

import numpy as np

from src.paths import SOURCE_ROOT

REFERENCE_PATH = SOURCE_ROOT / "docs" / "reference_queries.json"


def ndcg_at_k(ranked: list[str], relevance: dict[str, float], k: int = 5) -> float:
    """NDCG@k: 1 — эксперт поставил бы те же модели в том же порядке, 0 — ни одной нужной в топе."""
    gains = [relevance.get(m, 0.0) for m in ranked[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return float(dcg / idcg) if idcg else float("nan")


def evaluate(pipe, queries: list[dict], k: int = 5, **recommend_kw) -> list[dict]:
    from src.ml.labeling import IMPORTANCE_DEFAULTS, PRESETS
    from src.ml.recommender import recommend
    base = dict(budget_rub=35000, **IMPORTANCE_DEFAULTS, gaming=0, photo=0, work=0, os_pref=0, need_nfc=1,
                esim_pref=0, ir_pref=0, screen_pref=0, min_hz=0, min_rom=128)
    out = []
    for q in queries:
        if not q.get("relevant"):
            continue
        a = {**base, **PRESETS.get(q.get("preset", ""), {}), **q.get("answers", {})}
        models = recommend(pipe, a, top_n=k, **recommend_kw)["table"]["Телефон"].tolist()
        out.append({"запрос": q["name"], f"NDCG@{k}": ndcg_at_k(models, q["relevant"], k), "топ": models})
    return out


if __name__ == "__main__":
    from src import db
    from src.ml.classifier import PhonePipeline
    db.init_db()
    pipe = PhonePipeline.load() or PhonePipeline().fit(db.get_phones())
    queries = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))["queries"]
    res = evaluate(pipe, queries)
    if not res:
        print("В docs/reference_queries.json пока нет размеченных запросов (поле relevant пустое).")
    for r in res:
        print(f"{r['запрос']}: NDCG@5 = {r['NDCG@5']:.3f}  ← {r['топ']}")
    if res:
        print("Среднее NDCG@5:", round(np.nanmean([r["NDCG@5"] for r in res]), 3))

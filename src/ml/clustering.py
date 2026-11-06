"""Кластеризация телефонов: k-means++, метод локтя, силуэт, иерархическая, DBSCAN."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score

from src.ml.features import PHONE_FEATURES

# (признак, знак отклонения центроида) -> как назвать кластер
CLUSTER_WORDS = {
    ("price_rub", -1): "бюджетные", ("price_rub", 1): "премиум",
    ("battery_mah", 1): "автономные", ("camera_mp", 1): "камерофоны",
    ("antutu_k", 1): "мощные", ("refresh_hz", 1): "игровые",
    ("weight_g", -1): "компактные", ("weight_g", 1): "тяжёлые/защищённые",
    ("screen_in", 1): "большой экран", ("screen_in", -1): "небольшой экран",
    ("is_ios", 1): "iPhone", ("charge_w", 1): "быстрая зарядка",
    ("esim", 1): "с eSIM", ("has_5g", -1): "без 5G", ("year", -1): "прошлых лет",
    ("antutu_k", -1): "базовые", ("nfc", -1): "без NFC", ("ir_port", 1): "с ИК-портом", ("active_h", 1): "долго работают", ("standby_d", 1): "долго в ожидании",
}


def make_kmeans(k: int) -> KMeans:
    return KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, tol=1e-4, random_state=0)


def elbow(x: np.ndarray, k_max: int = 10) -> list[float]:
    return [make_kmeans(k).fit(x).inertia_ for k in range(1, k_max + 1)]


def silhouette_by_k(x: np.ndarray, k_range=range(2, 11)) -> dict[int, float]:
    return {k: silhouette_score(x, make_kmeans(k).fit_predict(x)) for k in k_range}


def choose_k(x: np.ndarray, k_min: int = 4, k_max: int = 7) -> int:
    """k с лучшим силуэтом среди интерпретируемых значений (не меньше 4 групп)."""
    scores = silhouette_by_k(x, range(k_min, k_max + 1))
    return max(scores, key=scores.get)


def compare_algorithms(x: np.ndarray, k: int) -> pd.DataFrame:
    rows = []
    for name, model in [
        ("KMeans (k-means++)", make_kmeans(k)),
        ("Agglomerative (complete)", AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="complete")),
        ("Agglomerative (ward)", AgglomerativeClustering(n_clusters=k, linkage="ward")),
        ("DBSCAN", DBSCAN(eps=_dbscan_eps(x), min_samples=4)),
    ]:
        lab = model.fit_predict(x)
        mask = lab >= 0
        n_cl = len(set(lab[mask]))
        ok = n_cl >= 2
        rows.append({
            "алгоритм": name, "кластеров": n_cl, "шум": int((~mask).sum()),
            "silhouette": silhouette_score(x[mask], lab[mask]) if ok else np.nan,
            "Davies–Bouldin": davies_bouldin_score(x[mask], lab[mask]) if ok else np.nan,
        })
    return pd.DataFrame(rows)


def _dbscan_eps(x: np.ndarray) -> float:
    """eps как медиана расстояния до 4-го соседа (эвристика «колена» k-distance)."""
    d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    return float(np.median(np.sort(d, axis=1)[:, 4]))


def silhouette_values(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return silhouette_samples(x, labels, metric="euclidean")


def hierarchical_linkage(x: np.ndarray) -> np.ndarray:
    return linkage(x, method="complete", metric="euclidean")


def cluster_names(z_std: np.ndarray, labels: np.ndarray) -> dict[int, str]:
    """Имя кластера по двум самым выраженным отклонениям центроида (в σ)."""
    names = {}
    for c in np.unique(labels):
        centroid = z_std[labels == c].mean(axis=0)
        words = []
        for i in np.argsort(-np.abs(centroid)):
            key = (PHONE_FEATURES[i], int(np.sign(centroid[i])))
            if key in CLUSTER_WORDS and CLUSTER_WORDS[key] not in words:
                words.append(CLUSTER_WORDS[key])
            if len(words) == 2:
                break
        title = ", ".join(words)
        if title and not title.startswith("iPhone"):
            title = title[:1].upper() + title[1:]
        names[int(c)] = title or f"Кластер {c}"
    return names


def cluster_profile_table(phones: pd.DataFrame, labels: np.ndarray, names: dict) -> pd.DataFrame:
    df = phones.assign(cluster=labels)
    prof = df.groupby("cluster").agg(
        телефонов=("model", "count"), цена_медиана=("price_rub", "median"),
        батарея=("battery_mah", "mean"), камера=("camera_mp", "mean"),
        antutu_k=("antutu_k", "mean"), вес=("weight_g", "mean"),
        доля_iOS=("os", lambda s: (s == "iOS").mean()))
    prof.insert(0, "название", [names[c] for c in prof.index])
    return prof.round(1)

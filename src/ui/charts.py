"""Графики matplotlib, встроенные в Qt, в спокойном едином стиле."""
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from scipy.cluster.hierarchy import dendrogram

from src.ml.features import effective_refresh
from src.ml.labeling import CLASS_NAMES
from src.ml.recommender import full_name

PALETTE = ["#5B7FDB", "#F2A65A", "#5DB58A", "#E07A7A", "#9C82D4", "#4FB3BF", "#D9B44A", "#A0806B"]
CLASS_COLORS = ["#8FBF9F", "#7FA6E0", "#C99BD9"]
TEXT, MUTED, GRID = "#2E3440", "#7A8194", "#ECEEF3"
SOFT_BLUES = LinearSegmentedColormap.from_list("soft", ["#FFFFFF", "#DCE5FA", "#5B7FDB"])
SOFT_DIVERGING = LinearSegmentedColormap.from_list("div", ["#E07A7A", "#FFFFFF", "#5B7FDB"])

mpl.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": "#D5D9E2", "axes.labelcolor": MUTED, "axes.titlecolor": TEXT,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlepad": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

# Признаки для профилей кластеров: (столбец, подпись, инвертировать?)
PROFILE_FEATURES = [("price_rub", "Цена", False), ("antutu_k", "Мощность", False),
                    ("camera_mp", "Камера", False), ("active_h", "Работа активно", False),
                    ("standby_d", "Ожидание", False),
                    ("screen_in", "Экран", False), ("weight_g", "Лёгкость", True),
                    ("charge_w", "Зарядка", False), ("refresh_hz", "Плавность", False)]


def cluster_color(c: int) -> str:
    return PALETTE[c % len(PALETTE)]


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, width=6, height=4):
        self.fig = Figure(figsize=(width, height), layout="constrained")
        super().__init__(self.fig)

    def clear(self, nrows=1, ncols=1, **kw):
        self.fig.clear()
        return self.fig.subplots(nrows, ncols, **kw)

    def done(self):
        self.draw_idle()


def _feature_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Признаки в шкале 0..1: перцентиль в базе; для частоты экрана — шкала эффективной частоты
    (выше 120 Гц с коэффициентом 0.2), чтобы 144+ Гц не выглядели намного лучше 120 Гц."""
    data = {}
    for col, title, invert in PROFILE_FEATURES:
        if col == "refresh_hz":
            hz = pd.Series(effective_refresh(df[col]), index=df.index)
            r = (hz - hz.min()) / (hz.max() - hz.min() + 1e-9)
        else:
            r = df[col].rank(pct=True)
        data[title] = 1 - r if invert else r
    return pd.DataFrame(data)


def _percentile_profile(pipe) -> pd.DataFrame:
    """Средний перцентиль признака по кластеру (0..1): удобно сравнивать разные единицы."""
    return _feature_scores(pipe.phones_).groupby(pipe.clusters_).mean()


def _radar_axes(canvas, n):
    canvas.fig.clear()
    return canvas.fig.add_subplot(111, projection="polar"), np.linspace(0, 2 * np.pi, n, endpoint=False)


def _style_radar(ax, angles, labels):
    ax.set_xticks(angles, labels, color=TEXT)
    ax.set_yticks([0.25, 0.5, 0.75], ["", "", ""])
    ax.set_ylim(0, 1)
    ax.spines["polar"].set_color(GRID)
    ax.grid(color=GRID)


# ---------- Результат ----------
def plot_proba(canvas: MplCanvas, proba):
    ax = canvas.clear()
    names = [CLASS_NAMES[i] for i in range(3)]
    best = int(np.argmax(proba))
    colors = [CLASS_COLORS[i] if i == best else "#E4E7EE" for i in range(3)]
    bars = ax.barh(names, proba, color=colors, height=0.55)
    for b, p in zip(bars, proba):
        ax.text(p + 0.02, b.get_y() + b.get_height() / 2, f"{p:.0%}", va="center", color=TEXT)
    ax.set_xlim(0, 1.12)
    ax.invert_yaxis()
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.set_title("Ответ нейросети: класс телефона")
    canvas.done()


def plot_reco_radar(canvas: MplCanvas, pipe, phone_ids, top=3, title="Топ-3: сравнение характеристик (перцентиль в базе)"):
    """Сравнение трёх лучших вариантов по перцентилям характеристик."""
    df = pipe.phones_
    labels = [t for _, t, _ in PROFILE_FEATURES]
    ax, angles = _radar_axes(canvas, len(labels))
    ranks = _feature_scores(df)
    for k, pid in enumerate(phone_ids[:top]):
        i = int(np.flatnonzero(df["id"].to_numpy() == pid)[0])
        v = ranks.iloc[i].to_numpy()
        ax.plot(np.append(angles, angles[0]), np.append(v, v[0]), color=PALETTE[k], lw=2,
                label=full_name(df.iloc[i]["brand"], df.iloc[i]["model"]))
        ax.fill(angles, v, color=PALETTE[k], alpha=0.12)
    _style_radar(ax, angles, labels)
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.15, 0.5), fontsize=9)
    canvas.done()


def plot_reco_scores(canvas: MplCanvas, table: pd.DataFrame):
    ax = canvas.clear()
    t = table.iloc[::-1]
    colors = [CLASS_COLORS[list(CLASS_NAMES.values()).index(c)] for c in t["Класс"]]
    ax.barh(t["Телефон"], t["Оценка"], color=colors, height=0.6)
    for y, (s, p) in enumerate(zip(t["Оценка"], t["Цена, ₽"])):
        ax.text(s + 0.03, y, f"{p:,} ₽".replace(",", " "), va="center", fontsize=8, color=MUTED)
    ax.set_xlim(0, t["Оценка"].max() * 1.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Итоговая оценка соответствия")
    ax.set_title("Насколько каждый вариант подходит вам")
    canvas.done()


# ---------- Кластеры ----------
def _confidence_ellipse(ax, pts, color):
    if len(pts) < 3:
        return
    cov = np.cov(pts, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*vecs[:, -1][::-1]))
    w, h = 2 * 1.8 * np.sqrt(np.maximum(vals[::-1], 1e-6))
    ax.add_patch(Ellipse(pts.mean(axis=0), w, h, angle=angle, facecolor=color, alpha=0.08,
                         edgecolor=color, lw=1.2, ls="--"))


def plot_clusters(canvas: MplCanvas, pipe, highlight_ids=(), compact=False):
    ax = canvas.clear()
    xy = pipe.space_.to_2d(pipe.x_)
    centers = pipe.space_.to_2d(pipe.kmeans_.cluster_centers_)
    markers = ["o", "s", "^"]
    for c in range(pipe.k_):
        col = cluster_color(c)
        in_c = pipe.clusters_ == c
        _confidence_ellipse(ax, xy[in_c], col)
        for lab in range(3):
            m = in_c & (pipe.labels_ == lab)
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1], s=34, c=col, marker=markers[lab], alpha=0.85,
                           edgecolor="white", linewidth=0.6)
        ax.scatter(*centers[c], s=180, marker="X", c=col, edgecolor=TEXT, linewidth=1, zorder=5)
        if not compact:
            ax.annotate(f"{pipe.cluster_names_[c]} ({in_c.sum()})", centers[c], xytext=(8, 8),
                        textcoords="offset points", fontsize=8, color=TEXT,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, alpha=0.9))
    if len(highlight_ids):
        m = np.isin(pipe.phones_["id"].to_numpy(), list(highlight_ids))
        ax.scatter(xy[m, 0], xy[m, 1], s=150, facecolors="none", edgecolors="#E07A7A", linewidth=2,
                   label="Ваши рекомендации", zorder=6)
        ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"Карта телефонов: {pipe.k_} кластеров KMeans (PCA)   ○ бюджет  □ средний  △ флагман")
    ax.set_xlabel("Главная компонента 1")
    ax.set_ylabel("Главная компонента 2")
    canvas.done()


def plot_cluster_radar(canvas: MplCanvas, pipe):
    prof = _percentile_profile(pipe)
    labels = list(prof.columns)
    ax, angles = _radar_axes(canvas, len(labels))
    for c, row in prof.iterrows():
        v = row.to_numpy()
        ax.plot(np.append(angles, angles[0]), np.append(v, v[0]), color=cluster_color(c), lw=2,
                label=pipe.cluster_names_[c])
        ax.fill(angles, v, color=cluster_color(c), alpha=0.08)
    _style_radar(ax, angles, labels)
    ax.set_title("Профили кластеров")
    ax.legend(loc="center left", bbox_to_anchor=(1.08, 0.5), fontsize=8)
    canvas.done()


def plot_price_box(canvas: MplCanvas, pipe):
    ax = canvas.clear()
    groups = [pipe.phones_["price_rub"][pipe.clusters_ == c] / 1000 for c in range(pipe.k_)]
    bp = ax.boxplot(groups, vert=False, patch_artist=True, widths=0.55,
                    medianprops=dict(color=TEXT), whiskerprops=dict(color=MUTED),
                    capprops=dict(color=MUTED), flierprops=dict(marker="o", markersize=3, alpha=0.5))
    for c, patch in enumerate(bp["boxes"]):
        patch.set(facecolor=cluster_color(c), alpha=0.35, edgecolor=cluster_color(c))
    ax.set_yticks(range(1, pipe.k_ + 1), [pipe.cluster_names_[c] for c in range(pipe.k_)])
    ax.set_xlabel("Цена, тыс. ₽")
    ax.set_title("Разброс цен внутри кластеров")
    canvas.done()


def plot_price_perf(canvas: MplCanvas, pipe):
    ax = canvas.clear()
    df = pipe.phones_
    sizes = 20 + 60 * (df["battery_mah"].clip(upper=8000) - 2000) / 6000
    for c in range(pipe.k_):
        m = pipe.clusters_ == c
        ax.scatter(df["price_rub"][m] / 1000, df["antutu_k"][m], s=sizes[m], c=cluster_color(c), alpha=0.7,
                   edgecolor="white", linewidth=0.6, label=pipe.cluster_names_[c])
    # линия «цена/производительность» — тренд по всей базе (лог-регрессия)
    x = np.log(df["price_rub"] / 1000)
    a, b = np.polyfit(x, df["antutu_k"], 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(np.exp(xs), a * xs + b, color=MUTED, ls="--", lw=1, label="средний тренд")
    best = (df["antutu_k"] - (a * x + b)).nlargest(4).index
    for n, i in enumerate(best):
        ax.annotate(full_name(df.loc[i, "brand"], df.loc[i, "model"]),
                    (df.loc[i, "price_rub"] / 1000, df.loc[i, "antutu_k"]),
                    xytext=(-14 if n % 2 == 0 else 14, 12 + 6 * n), textcoords="offset points",
                    ha="right" if n % 2 == 0 else "left", fontsize=7.5,
                    color=TEXT, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xscale("log")
    ticks = [7, 10, 15, 20, 30, 50, 70, 100, 150]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Цена, тыс. ₽ (лог. шкала)")
    ax.set_ylabel("AnTuTu, тыс.")
    ax.set_title("Цена vs производительность (размер точки — батарея; подписаны самые выгодные)")
    ax.legend(fontsize=8, loc="upper left")
    canvas.done()


def plot_corr(canvas: MplCanvas, pipe):
    from src.ml.features import FEATURE_NAMES_RU, PHONE_FEATURES
    ax = canvas.clear()
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.nan_to_num(np.corrcoef(pipe.z_, rowvar=False))
    im = ax.imshow(corr, cmap=SOFT_DIVERGING, vmin=-1, vmax=1)
    names = [FEATURE_NAMES_RU[f] for f in PHONE_FEATURES]
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            if abs(corr[i, j]) >= 0.5 and i != j:
                ax.text(j, i, f"{corr[i, j]:.1f}", ha="center", va="center", fontsize=6.5, color=TEXT)
    canvas.fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Корреляции признаков")
    canvas.done()


def plot_classes(canvas: MplCanvas, pipe):
    ax1, ax2 = canvas.clear(1, 2)
    names = [CLASS_NAMES[i] for i in range(3)]
    for ax, counts, title in ((ax1, np.bincount(pipe.labels_, minlength=3), "Телефоны по классам"),
                              (ax2, pipe.class_balance_, "Обучающие анкеты по классам")):
        bars = ax.bar(names, counts, color=CLASS_COLORS, width=0.55)
        for b, n in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2, n, str(n), ha="center", va="bottom", color=TEXT)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    canvas.done()


def plot_class_cluster_matrix(canvas: MplCanvas, pipe):
    ax = canvas.clear()
    m = pd.crosstab(pd.Series(pipe.clusters_).map(pipe.cluster_names_),
                    pd.Series(pipe.labels_).map(CLASS_NAMES))
    m = m.reindex(columns=[CLASS_NAMES[i] for i in range(3)], fill_value=0)
    ax.imshow(m.to_numpy(), cmap=SOFT_BLUES, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, m.iat[i, j], ha="center", va="center", color=TEXT)
    ax.set_xticks(range(3), m.columns)
    ax.set_yticks(range(len(m)), m.index)
    ax.grid(False)
    ax.set_title("Кластеры × классы: кластеры не равны ценовым классам")
    canvas.done()


# ---------- выбор k ----------
def plot_elbow_silhouette(canvas: MplCanvas, inertias, sil_by_k, k_chosen):
    ax1, ax2 = canvas.clear(1, 2)
    ax1.plot(range(1, len(inertias) + 1), inertias, marker="o", color=PALETTE[0])
    ax1.axvline(k_chosen, color=PALETTE[3], ls="--", lw=1)
    ax1.set_xlabel("Число кластеров")
    ax1.set_ylabel("Distortion (inertia)")
    ax1.set_title("Метод локтя")
    ks = list(sil_by_k)
    ax2.plot(ks, [sil_by_k[k] for k in ks], marker="o", color=PALETTE[1])
    ax2.axvline(k_chosen, color=PALETTE[3], ls="--", lw=1, label=f"выбрано k={k_chosen}")
    ax2.set_xlabel("Число кластеров")
    ax2.set_ylabel("Средний силуэт")
    ax2.set_title("Силуэт по k")
    ax2.legend()
    canvas.done()


def plot_silhouette(canvas: MplCanvas, sil_vals, labels, names):
    """Силуэтный график в стиле clustering.py курса."""
    ax = canvas.clear()
    y_lower, yticks = 0, []
    clusters = np.unique(labels)
    for c in clusters:
        vals = np.sort(sil_vals[labels == c])
        y_upper = y_lower + len(vals)
        ax.fill_betweenx(range(y_lower, y_upper), 0, vals, color=cluster_color(c), alpha=0.7)
        yticks.append((y_lower + y_upper) / 2)
        y_lower = y_upper + 3
    ax.axvline(np.mean(sil_vals), color=PALETTE[3], ls="--", lw=1, label=f"среднее {np.mean(sil_vals):.2f}")
    ax.set_yticks(yticks, [names[c] for c in clusters], fontsize=8)
    ax.set_xlabel("Коэффициент силуэта")
    ax.set_title("Силуэтный график кластеров")
    ax.legend()
    canvas.done()


def plot_dendrogram(canvas: MplCanvas, link, labels):
    ax = canvas.clear()
    from scipy.cluster import hierarchy
    hierarchy.set_link_color_palette(PALETTE)
    dendrogram(link, labels=labels, ax=ax, leaf_font_size=5, leaf_rotation=90,
               above_threshold_color="#C5CBD8")
    ax.grid(False)
    ax.set_ylabel("Евклидово расстояние")
    ax.set_title("Иерархическая кластеризация (complete linkage)")
    canvas.done()


# ---------- нейросеть и модели ----------
def plot_training(canvas: MplCanvas, pipe):
    ax1, ax2 = canvas.clear(1, 2)
    ax1.plot(pipe.loss_curve_, color=PALETTE[0], lw=2, label="loss (train)")
    ax1.set_xlabel("Эпоха")
    ax1.set_ylabel("Log-loss")
    if pipe.val_curve_:
        ax1b = ax1.twinx()
        ax1b.plot(pipe.val_curve_, color=PALETTE[1], lw=2, label="accuracy (val)")
        ax1b.set_ylabel("Accuracy (val)")
        ax1b.grid(False)
        ax1b.spines["right"].set_visible(True)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax1b.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="center right")
    ax1.set_title("Обучение нейросети MLP")
    cmx = pipe.confusion_
    norm = cmx / cmx.sum(axis=1, keepdims=True)
    ax2.imshow(norm, cmap=SOFT_BLUES, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f"{cmx[i, j]}\n{norm[i, j]:.0%}", va="center", ha="center", color=TEXT)
    names = [CLASS_NAMES[i] for i in range(3)]
    ax2.set_xticks(range(3), names)
    ax2.set_yticks(range(3), names)
    ax2.grid(False)
    ax2.set_xlabel("Предсказанный класс")
    ax2.set_ylabel("Истинный класс")
    ax2.set_title("Матрица ошибок (тест)")
    canvas.done()


def plot_ablation(canvas: MplCanvas, df):
    ax = canvas.clear()
    pivot = df.pivot(index="модель", columns="вариант", values="accuracy").sort_values("с кластеризацией")
    y = np.arange(len(pivot))
    ax.barh(y - 0.2, pivot["без кластеризации"], height=0.38, label="без кластеризации", color="#C5CBD8")
    ax.barh(y + 0.2, pivot["с кластеризацией"], height=0.38, label="с кластеризацией", color=PALETTE[0])
    for i, (a, b) in enumerate(zip(pivot["без кластеризации"], pivot["с кластеризацией"])):
        d = b - a
        ax.text(max(a, b) + 0.003, i, f"{d:+.1%}", va="center", fontsize=8,
                color=PALETTE[2] if d > 0 else PALETTE[3])
    ax.set_yticks(y, pivot.index)
    ax.set_xlim(max(0.0, pivot.min().min() - 0.05), min(1.0, pivot.max().max() + 0.04))
    ax.set_xlabel("Accuracy (5-fold CV)")
    ax.set_title("Сравнение моделей: с кластеризацией vs без")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    canvas.done()


def plot_weights(canvas: MplCanvas, table):
    ax = canvas.clear()
    t = table.sort_values("вес")
    colors = [PALETTE[0] if w >= 1 else "#C5CBD8" for w in t["вес"]]
    ax.barh(t["признак"], t["вес"], color=colors, height=0.6)
    ax.axvline(1.0, color=MUTED, ls="--", lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_title("Веса признаков (1 — средний)")
    canvas.done()


def plot_contrib(canvas: MplCanvas, contrib: pd.Series, title: str):
    """«Почему он?» — вклад признаков в оценку телефона под ваши ответы."""
    ax = canvas.clear()
    colors = [PALETTE[2] if v > 0 else PALETTE[3] for v in contrib]
    ax.barh(contrib.index, contrib.values, color=colors, height=0.6)
    ax.axvline(0, color=MUTED, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("вклад в оценку (зелёное — плюс, красное — минус)")
    ax.set_title(title)
    canvas.done()

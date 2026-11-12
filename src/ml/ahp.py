"""Метод анализа иерархий (AHP): веса критериев из попарных сравнений, а не «на глаз».

Эксперт заполняет только верхний треугольник матрицы: насколько критерий i важнее j по шкале Саати
(1 — одинаково, 3 — немного важнее, 5 — важнее, 7 — сильно, 9 — абсолютно; 1/3 … 1/9 — наоборот).
Веса — главный собственный вектор матрицы. Индекс согласованности CR < 0,1 означает, что суждения
не противоречат друг другу (нет «A важнее B, B важнее C, но C важнее A»).
"""
import numpy as np

# Случайный индекс согласованности Саати для n = 1…10
RANDOM_INDEX = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]


def ahp_weights(criteria: list[str], judgments: dict[tuple[str, str], float]) -> tuple[dict, float]:
    """judgments[(a, b)] = во сколько раз a важнее b. Возвращает (веса, CR)."""
    n = len(criteria)
    idx = {c: i for i, c in enumerate(criteria)}
    m = np.ones((n, n))
    for (a, b), v in judgments.items():
        m[idx[a], idx[b]] = v
        m[idx[b], idx[a]] = 1.0 / v
    vals, vecs = np.linalg.eig(m)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    w /= w.sum()
    ci = (vals[k].real - n) / (n - 1) if n > 2 else 0.0
    cr = ci / RANDOM_INDEX[n - 1] if n > 2 and RANDOM_INDEX[n - 1] else 0.0
    return dict(zip(criteria, w.round(4))), float(cr)


def weights_to_sliders(weights: dict[str, float], scale: int = 10) -> dict[str, int]:
    """Веса AHP → значения ползунков 0…10 (самый важный критерий = 10)."""
    top = max(weights.values())
    return {k: int(round(v / top * scale)) for k, v in weights.items()}

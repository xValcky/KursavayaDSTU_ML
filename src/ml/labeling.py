"""Метки классов телефонов [0,1,2], полезность телефона для анкеты и генератор анкет.

0 — бюджетный, 1 — средний, 2 — флагман.
"""
import numpy as np
import pandas as pd

from src.ml.features import PHONE_FEATURES, effective_refresh, is_rugged

CLASS_NAMES = {0: "Бюджетный", 1: "Средний", 2: "Флагман"}

# важности 0..10 (ползунки опроса)
IMPORTANCE_FIELDS = ["imp_camera", "imp_active", "imp_standby", "imp_perf", "imp_compact",
                     "imp_charge", "imp_storage", "imp_new", "imp_value", "imp_rugged"]
IMPORTANCE_TITLES = {
    "imp_camera": "Камера", "imp_active": "Без розетки при активном использовании",
    "imp_standby": "Долго держит заряд в ожидании", "imp_perf": "Производительность",
    "imp_compact": "Компактность и лёгкость", "imp_charge": "Быстрая зарядка",
    "imp_storage": "Много памяти", "imp_new": "Новинка, свежая модель", "imp_value": "Выгодная цена",
    "imp_rugged": "Прочный защищённый корпус",
}
# подсказки к ползункам (показываются при наведении)
IMPORTANCE_HINTS = {
    "imp_camera": "Мегапиксели основной камеры. При сценарии «Фото и видео» вес камеры выше.",
    "imp_active": "Сколько часов телефон проработает с включённым экраном (оценка по батарее, экрану и чипу).",
    "imp_standby": "Сколько дней продержится в режиме ожидания — важно для «звонилки» и работы.",
    "imp_perf": "Мощность чипа (AnTuTu), объём ОЗУ и плавность экрана. Выше 120 Гц почти не учитывается.",
    "imp_compact": "Чем выше, тем сильнее штраф за вес и большую диагональ.",
    "imp_charge": "Мощность зарядки в ваттах: быстрее заряжается.",
    "imp_storage": "Объём постоянной и оперативной памяти.",
    "imp_new": "Год выпуска: свежие модели дольше получают обновления.",
    "imp_value": "Сильнее тянет к более дешёвым моделям внутри бюджета.",
    "imp_rugged": "Тяжёлые защищённые «броневики»: падения, вода, огромная батарея. Пригодится курьеру и на стройке.",
}
IMPORTANCE_DEFAULTS = {"imp_camera": 5, "imp_active": 6, "imp_standby": 4, "imp_perf": 5, "imp_compact": 3,
                       "imp_charge": 4, "imp_storage": 4, "imp_new": 3, "imp_value": 5, "imp_rugged": 0}
# в простом режиме видны только эти ползунки
SIMPLE_IMPORTANCE = ["imp_camera", "imp_active", "imp_perf", "imp_compact"]

SURVEY_FIELDS = ["budget_rub", *IMPORTANCE_FIELDS,
                 "gaming", "photo", "work", "os_pref", "need_nfc", "esim_pref", "ir_pref",
                 "screen_pref", "min_hz", "min_rom"]

# Готовые профили для простого режима: значения ползунков и сценариев
PRESETS = {
    "Игры": {"gaming": 1, "imp_perf": 10, "imp_active": 8, "imp_charge": 7, "imp_camera": 3, "imp_compact": 1,
             "screen_pref": 3, "min_hz": 120},
    "Фото и видео": {"photo": 1, "imp_camera": 10, "imp_storage": 8, "imp_perf": 6, "imp_active": 6},
    "Кино и соцсети": {"imp_active": 8, "imp_standby": 5, "imp_camera": 5, "imp_perf": 4, "screen_pref": 3,
                       "imp_compact": 2},
    "Работа": {"work": 1, "imp_standby": 8, "imp_active": 7, "imp_storage": 6, "imp_perf": 6, "esim_pref": 1},
    "Для родителей": {"imp_standby": 9, "imp_active": 7, "imp_value": 9, "imp_perf": 2, "imp_camera": 3,
                      "imp_new": 1, "screen_pref": 3},
    "Такси и курьер": {"work": 1, "imp_active": 10, "imp_standby": 6, "imp_perf": 6, "imp_charge": 6,
                       "imp_rugged": 6, "imp_camera": 1, "imp_compact": 0, "imp_value": 6},
}
PRESET_HINTS = {
    "Игры": "Мощный чип, плавный экран от 120 Гц, большая диагональ, автономность и быстрая зарядка.",
    "Фото и видео": "Камера на первом месте, много памяти под видео, хорошая автономность.",
    "Кино и соцсети": "Большой экран и долгая работа без розетки.",
    "Работа": "Долго держит заряд, много памяти, eSIM — плюс.",
    "Для родителей": "Простой и выгодный: долго в ожидании, большой экран, без переплаты за мощность.",
    "Такси и курьер": "Максимум часов работы с экраном (навигатор весь день), шустрый интерфейс, "
                      "быстрая зарядка; защищённый корпус — плюс.",
}
OS_PREF = {0: "Неважно", 1: "Android", 2: "iOS"}
# 0 — не нужен, 1 — может быть (плюс, если есть), 2 — обязательно
OPTION_PREF = {0: "Не нужен", 1: "Может быть", 2: "Обязательно"}
# размер экрана: (подпись, диапазон диагонали в дюймах)
SCREEN_PREF = {0: ("Неважно", (0.0, 99.0)), 1: ("Компактный, до 6,3″", (0.0, 6.3)),
               2: ("Обычный, 6,3–6,7″", (6.3, 6.75)), 3: ("Большой, от 6,7″", (6.7, 99.0)),
               4: ("Очень большой, от 6,9″", (6.88, 99.0))}
HZ_PREF = [0, 90, 120, 144]

_IDX = {f: i for i, f in enumerate(PHONE_FEATURES)}


def phone_labels(df: pd.DataFrame) -> np.ndarray:
    """Метка класса по взвешенному скорингу перцентилей характеристик."""
    rank = lambda c: df[c].rank(pct=True).to_numpy()
    # для частоты экрана не ранг, а шкала эффективной частоты: 144+ Гц почти не лучше 120 Гц
    hz = effective_refresh(df["refresh_hz"])
    hz_score = (hz - hz.min()) / (hz.max() - hz.min() + 1e-9)
    score = (0.35 * rank("price_rub") + 0.25 * rank("antutu_k") + 0.10 * rank("camera_mp")
             + 0.10 * rank("ram_gb") + 0.10 * hz_score + 0.05 * rank("charge_w")
             + 0.05 * rank("year"))
    lo, hi = np.quantile(score, [0.40, 0.72])
    return np.digitize(score, [lo, hi]).astype(int)


def preference_weights(a: dict) -> np.ndarray:
    """Перевод ответов опроса в веса по стандартизованным признакам телефона."""
    cam, perf, comp = a["imp_camera"] / 10, a["imp_perf"] / 10, a["imp_compact"] / 10
    act, stb = a["imp_active"] / 10, a["imp_standby"] / 10
    chg, sto, new = a["imp_charge"] / 10, a["imp_storage"] / 10, a["imp_new"] / 10
    big = {0: 0.0, 1: -0.8, 2: 0.0, 3: 0.6, 4: 0.9}[a["screen_pref"]]
    w = np.zeros(len(PHONE_FEATURES))
    w[_IDX["camera_mp"]] = cam + 0.5 * a["photo"]
    w[_IDX["antutu_k"]] = perf + 0.8 * a["gaming"] + 0.2
    w[_IDX["refresh_hz"]] = 0.4 * perf + 0.4 * a["gaming"] + (0.4 if a["min_hz"] >= 120 else 0.0)
    w[_IDX["ram_gb"]] = 0.4 * perf + 0.3 * a["work"] + 0.3 * sto
    w[_IDX["rom_gb"]] = 0.3 * a["photo"] + 0.8 * sto + 0.1
    w[_IDX["battery_mah"]] = 0.2 * (act + stb)
    w[_IDX["active_h"]] = act
    w[_IDX["standby_d"]] = stb
    w[_IDX["charge_w"]] = chg + 0.2 * act
    w[_IDX["screen_in"]] = big - 0.5 * comp
    w[_IDX["weight_g"]] = -0.9 * comp - 0.2
    w[_IDX["year"]] = 0.1 + 0.8 * new
    w[_IDX["has_5g"]] = 0.2
    w[_IDX["nfc"]] = 0.2
    w[_IDX["esim"]] = 0.1 * a["work"] + (0.4 if a["esim_pref"] == 1 else 0.0)
    w[_IDX["ir_port"]] = 0.4 if a["ir_pref"] == 1 else 0.0
    return w


# Названия жёстких требований (для подсказки «что отсекло больше всего»)
REQUIREMENT_TITLES = {"budget": "бюджет", "os": "система", "nfc": "NFC", "esim": "eSIM — обязательно",
                      "ir": "ИК-порт — обязательно", "screen": "размер экрана", "hz": "частота экрана",
                      "rom": "объём памяти"}


def penalty_breakdown(a: dict, phones: pd.DataFrame) -> pd.DataFrame:
    """Штрафы по каждому требованию отдельно (строка — телефон, колонка — требование)."""
    price = phones["price_rub"].to_numpy()
    ratio = price / a["budget_rub"]
    n = len(phones)
    zero = np.zeros(n)
    os_bad = zero
    if a["os_pref"] == 1:
        os_bad = (phones["os"] != "Android").to_numpy()
    elif a["os_pref"] == 2:
        os_bad = (phones["os"] != "iOS").to_numpy()
    lo, hi = SCREEN_PREF[a["screen_pref"]][1]
    screen = phones["screen_in"].to_numpy()
    miss = np.clip(lo - screen, 0, None) + np.clip(screen - hi, 0, None)
    return pd.DataFrame({
        "budget": 30.0 * np.clip(ratio - 1.0, 0, None),           # дороже бюджета — резко хуже
        "os": 6.0 * os_bad,
        "nfc": 6.0 * (phones["nfc"].to_numpy() == 0) if a["need_nfc"] else zero,
        "esim": 6.0 * (phones["esim"].to_numpy() == 0) if a["esim_pref"] == 2 else zero,
        "ir": 6.0 * (phones["ir_port"].to_numpy() == 0) if a["ir_pref"] == 2 else zero,
        "screen": 15.0 * miss + 3.0 * (miss > 0),                  # не тот размер — почти как неподходящая ОС
        "hz": 5.0 * (phones["refresh_hz"].to_numpy() < a["min_hz"]),
        "rom": 3.0 * (phones["rom_gb"].to_numpy() < a["min_rom"]),
    })


def feasible_mask(a: dict, phones: pd.DataFrame, slack: float = 1.1) -> np.ndarray:
    """Телефоны, проходящие все жёсткие требования и не дороже бюджета более чем на 10%."""
    pen = penalty_breakdown(a, phones).drop(columns="budget")
    return (pen.sum(axis=1).to_numpy() == 0) & (phones["price_rub"].to_numpy() <= a["budget_rub"] * slack)


def utility(a: dict, z: np.ndarray, phones: pd.DataFrame) -> np.ndarray:
    """Полезность каждого телефона для анкеты: качество по интересам минус штрафы."""
    u = z @ preference_weights(a)
    ratio = phones["price_rub"].to_numpy() / a["budget_rub"]
    u -= (0.3 + 2.0 * a["imp_value"] / 10) * ratio  # чем важнее выгода, тем сильнее тянет к дешёвым
    u += a.get("imp_rugged", 0) / 10 * 1.2 * is_rugged(phones)  # «броневик» — плюс, а не мусор
    return u - penalty_breakdown(a, phones).sum(axis=1).to_numpy()


def survey_vector(a: dict) -> np.ndarray:
    v = np.array([a[f] for f in SURVEY_FIELDS], dtype=np.float64)
    v[0] = np.log(a["budget_rub"])
    v[-1] = np.log2(a["min_rom"])
    return v


def random_survey(rng: np.random.Generator) -> dict:
    gaming, photo, work = (rng.random(3) < [0.3, 0.35, 0.3]).astype(int)
    imp = lambda base: int(np.clip(round(rng.normal(base, 2.5)), 0, 10))
    pick = lambda options, p: int(rng.choice(options, p=p))
    screen_p = [0.35, 0.12, 0.2, 0.23, 0.1] if not gaming else [0.25, 0.03, 0.12, 0.35, 0.25]
    return {
        "budget_rub": float(np.clip(np.exp(rng.normal(np.log(35000), 0.8)), 7000, 200000)),
        "imp_camera": imp(4 + 4 * photo), "imp_active": imp(6 + gaming), "imp_standby": imp(4 + 2 * work),
        "imp_perf": imp(4 + 4 * gaming), "imp_compact": imp(3), "imp_charge": imp(4 + 2 * gaming),
        "imp_storage": imp(4 + 3 * photo), "imp_new": imp(3), "imp_value": imp(5),
        "imp_rugged": int(np.clip(round(rng.normal(1 + 3 * work, 2.5)), 0, 10)),
        "gaming": int(gaming), "photo": int(photo), "work": int(work),
        "os_pref": pick([0, 1, 2], [0.5, 0.3, 0.2]),
        "need_nfc": int(rng.random() < 0.8),
        "esim_pref": pick([0, 1, 2], [0.6, 0.25, 0.15] if not work else [0.4, 0.35, 0.25]),
        "ir_pref": pick([0, 1, 2], [0.65, 0.22, 0.13]),
        "screen_pref": pick([0, 1, 2, 3, 4], screen_p),
        "min_hz": pick(HZ_PREF, [0.45, 0.1, 0.35, 0.1] if not gaming else [0.2, 0.05, 0.45, 0.3]),
        "min_rom": pick([64, 128, 256, 512], [0.15, 0.4, 0.35, 0.1]),
    }


def target_label(a: dict, z: np.ndarray, phones: pd.DataFrame, labels: np.ndarray,
                 rng: np.random.Generator) -> int:
    """«Истинная» метка анкеты — класс лучшего для человека телефона.

    Человек не всегда точно знает свои приоритеты, поэтому оценки в анкете
    зашумляются: на самом деле он выбирает по «скрытым» важностям.
    """
    hidden = dict(a)
    for k in IMPORTANCE_FIELDS:
        hidden[k] = float(np.clip(a[k] + rng.normal(0, 1.5), 0, 10))
    hidden["budget_rub"] = a["budget_rub"] * float(np.exp(rng.normal(0, 0.08)))
    u = utility(hidden, z, phones) + rng.normal(0, 0.25, len(phones))
    return int(labels[np.argmax(u)])


def generate_dataset(n: int, z: np.ndarray, phones: pd.DataFrame, labels: np.ndarray,
                     seed: int = 0) -> tuple[list[dict], np.ndarray]:
    rng = np.random.default_rng(seed)
    surveys = [random_survey(rng) for _ in range(n)]
    y = np.array([target_label(a, z, phones, labels, rng) for a in surveys])
    return surveys, y

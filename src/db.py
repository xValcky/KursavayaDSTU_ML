"""SQLite-база: телефоны, анкеты опроса, отзывы пользователя и история цен."""
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data_loader import check_data_quality
from src.paths import DATA_DIR
from src.preprocess import drop_invalid, handle_missing

DB_PATH = DATA_DIR / "phones.db"
SEED_PATH = DATA_DIR / "phones_seed.csv"

NUMERIC_COLS = ["year", "price_rub", "ram_gb", "rom_gb", "battery_mah", "screen_in",
                "refresh_hz", "weight_g", "camera_mp", "antutu_k", "charge_w",
                "has_5g", "nfc", "esim", "ir_port"]
CATEGORY_VALUES = {"os": ["Android", "iOS", "HarmonyOS"]}

SCHEMA = """
CREATE TABLE IF NOT EXISTS phones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    brand TEXT, model TEXT UNIQUE, os TEXT, year INTEGER,
    price_rub REAL, ram_gb REAL, rom_gb REAL, battery_mah REAL, screen_in REAL,
    refresh_hz REAL, weight_g REAL, camera_mp REAL, antutu_k REAL, charge_w REAL,
    has_5g INTEGER, nfc INTEGER, esim INTEGER, ir_port INTEGER DEFAULT 0,
    active_h REAL, standby_d REAL,
    price_source TEXT DEFAULT 'seed', price_updated TEXT
);
CREATE TABLE IF NOT EXISTS survey_answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT, answers TEXT, predicted_class INTEGER
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    survey_id INTEGER, phone_id INTEGER, liked INTEGER, created TEXT
);
CREATE TABLE IF NOT EXISTS price_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phone_id INTEGER, price_rub REAL, source TEXT, created TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def connect():
    """Соединение с транзакцией: commit при успехе, rollback при ошибке, затем закрытие."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def estimate_autonomy(df: pd.DataFrame) -> pd.DataFrame:
    """Оценочная модель автономности (не замер!).

    active_h  — часов активного использования (экран включён): ёмкость / средний ток,
                ток растёт с диагональю, частотой экрана и мощностью чипа;
    standby_d — дней в режиме ожидания: ёмкость / фоновый ток (5G и мощный чип тратят больше).
    Новые поколения чипов экономичнее — поправка до −12% за 3 года.
    """
    df = df.copy()
    eff = 1.0 - 0.04 * (df["year"] - 2023).clip(0, 3)
    foldable = (df["screen_in"] > 7.5).astype(float)
    active_ma = (380 + 55 * (df["screen_in"] - 6.0) + 0.35 * (df["refresh_hz"] - 60)
                 + 0.04 * df["antutu_k"] + 120 * foldable) * eff
    idle_ma = (9 + 3 * df["has_5g"] + 0.002 * df["antutu_k"]) * eff
    df["active_h"] = (df["battery_mah"] / active_ma).round(1)
    df["standby_d"] = (df["battery_mah"] / idle_ma / 24).round(1)
    return df


def clean_phone_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Проверка и очистка таблицы телефонов функциями из курса."""
    if "ir_port" not in df.columns:
        df = df.assign(ir_port=0)
    report = check_data_quality(df, NUMERIC_COLS, CATEGORY_VALUES)
    df = drop_invalid(df, numeric_cols=NUMERIC_COLS, category_values=CATEGORY_VALUES)
    df = handle_missing(df, "dropnan", columns=["brand", "model", "price_rub"])
    df = handle_missing(df, "mean", columns=[c for c in NUMERIC_COLS if c in df.columns])
    return estimate_autonomy(df), report


def upsert_phones(df: pd.DataFrame, source: str = "seed") -> int:
    df, _ = clean_phone_frame(df)
    cols = ["brand", "model", "os"] + NUMERIC_COLS + ["active_h", "standby_d"]
    rows = [tuple(r[c] for c in cols) + (r.get("price_source", source) or source, _now())
            for _, r in df.iterrows()]
    placeholders = ",".join("?" * (len(cols) + 2))
    update = ",".join(f"{c}=excluded.{c}" for c in cols[1:] + ["price_source", "price_updated"])
    with connect() as conn:
        conn.executemany(
            f"INSERT INTO phones ({','.join(cols)},price_source,price_updated) VALUES ({placeholders}) "
            f"ON CONFLICT(model) DO UPDATE SET {update}", rows)
    return len(rows)


def _seed_hash() -> str:
    return hashlib.md5(SEED_PATH.read_bytes()).hexdigest()


def init_db() -> None:
    """Создать схему; при изменении встроенного датасета синхронизировать с ним базу."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(phones)")}
        for col, sql_type in (("ir_port", "INTEGER DEFAULT 0"), ("active_h", "REAL"), ("standby_d", "REAL")):
            if col not in columns:  # база от старой версии приложения
                conn.execute(f"ALTER TABLE phones ADD COLUMN {col} {sql_type}")
                conn.execute("DELETE FROM meta WHERE key='seed_hash'")  # пересчитать из датасета
        row = conn.execute("SELECT value FROM meta WHERE key='seed_hash'").fetchone()
    if row and row["value"] == _seed_hash():
        return
    seed = pd.read_csv(SEED_PATH)
    with connect() as conn:
        # модели, которых больше нет в датасете, удаляем (кроме импортированных пользователем)
        marks = ",".join("?" * len(seed))
        conn.execute(f"DELETE FROM phones WHERE model NOT IN ({marks}) AND price_source NOT LIKE 'csv:%'",
                     list(seed["model"]))
    upsert_phones(seed, "seed")
    with connect() as conn:
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('seed_hash', ?)", (_seed_hash(),))


def import_csv(path: str) -> tuple[int, dict]:
    df = pd.read_csv(path).drop(columns=["price_source"], errors="ignore")
    _, report = clean_phone_frame(df)
    return upsert_phones(df, f"csv:{Path(path).name}"), report


def get_phones() -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query("SELECT * FROM phones ORDER BY price_rub", conn)


def update_price(phone_id: int, price: float, source: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE phones SET price_rub=?, price_source=?, price_updated=? WHERE id=?",
                     (price, source, _now(), phone_id))
        conn.execute("INSERT INTO price_history (phone_id, price_rub, source, created) VALUES (?,?,?,?)",
                     (phone_id, price, source, _now()))


def save_survey(answers: dict, predicted_class: int) -> int:
    with connect() as conn:
        cur = conn.execute("INSERT INTO survey_answers (created, answers, predicted_class) VALUES (?,?,?)",
                           (_now(), json.dumps(answers, ensure_ascii=False), predicted_class))
        return cur.lastrowid


def save_feedback(survey_id: int, phone_id: int, liked: bool) -> None:
    with connect() as conn:
        conn.execute("INSERT INTO feedback (survey_id, phone_id, liked, created) VALUES (?,?,?,?)",
                     (survey_id, phone_id, int(liked), _now()))


def get_liked_feedback() -> list[tuple[dict, int]]:
    """Пары (анкета, id понравившегося телефона) — дополнительные данные для дообучения.

    Анкеты старых версий (без новых полей) пропускаются.
    """
    from src.ml.labeling import SURVEY_FIELDS
    with connect() as conn:
        rows = conn.execute(
            "SELECT s.answers, f.phone_id FROM feedback f JOIN survey_answers s ON s.id = f.survey_id "
            "WHERE f.liked = 1").fetchall()
    pairs = [(json.loads(r["answers"]), r["phone_id"]) for r in rows]
    return [(a, pid) for a, pid in pairs if all(k in a for k in SURVEY_FIELDS)]


def stats() -> dict:
    with connect() as conn:
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("phones", "survey_answers", "feedback", "price_history")}

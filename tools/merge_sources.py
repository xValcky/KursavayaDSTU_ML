"""Слияние всех источников в единую базу data/phones_seed.csv.

Источники (data/sources/):
  base_aug2026.csv     — первая база, август 2026 (все модели остаются);
  ishopper_specs.csv   — модели из t.me/ishopper с характеристиками из постов (заменяют старые значения);
  extra_specs.csv      — модели, найденные только в t.me/polzaSKIDKI (характеристики из постов/обзоров);
  ishopper.json, polzaskidki.json — цены из постов (tools/scrape_telegram.py).

Правило цены: медиана цен из обоих каналов за последние 60 дней, иначе последняя
известная цена из каналов, иначе августовская оценка.

Запуск:  python tools/merge_sources.py [--unmatched]
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "sources"
OUT = ROOT / "data" / "phones_seed.csv"
HISTORY_OUT = ROOT / "data" / "price_history_seed.csv"
FRESH_FROM = "2026-09-06"
EXCLUDED_BRANDS = {"Huawei", "Sony", "Meizu"}
EXCLUDED_MODELS = {"realme 16 Pro (Harry Potter Edition)"}
# позиции постов, которые нельзя сопоставлять ни с какой моделью (лимитки и т.п.)
EXCLUDED_RAW = re.compile(r"harry\s*potter", re.I)
COLUMNS = ["brand", "model", "os", "year", "price_rub", "ram_gb", "rom_gb", "battery_mah", "screen_in",
           "refresh_hz", "weight_g", "camera_mp", "antutu_k", "charge_w", "has_5g", "nfc", "esim", "ir_port",
           "price_source"]
# Названия в постах, которые отличаются от названия модели в базе
ALIASES = {"galaxys26ultra": ["samsungs26ultra"], "phone(3)": ["nothingphone3"], "motog100s": ["motorolag100s"],
           "legiony70(2026)": ["legiony70"],
           "nubiaz70sultra": ["nubiaz70sultra"], "vivoiqoo15t": ["iqoo15t"]}


def key(text: str) -> str:
    t = text.lower().replace("plus", "+").replace("ё", "е")
    t = re.sub(r"\b(5g|global|глобальн\w*)\b", "", t)
    return re.sub(r"[^a-z0-9а-я+()]", "", t).replace("()", "")


def load_specs() -> pd.DataFrame:
    base = pd.read_csv(SRC / "base_aug2026.csv")
    newer = [pd.read_csv(SRC / f) for f in ("ishopper_specs.csv", "extra_specs.csv") if (SRC / f).exists()]
    df = base.copy()
    df["_k"] = df["model"].map(key)
    stats = {"base": len(df), "updated_specs": 0, "added": 0}
    for new in newer:
        for _, row in new.iterrows():
            k = key(row["model"])
            hit = df.index[df["_k"] == k]
            values = {c: row[c] for c in COLUMNS if c in row and pd.notna(row[c])}
            if len(hit):
                for c, v in values.items():
                    df.loc[hit[0], c] = v
                stats["updated_specs"] += 1
            else:
                df = pd.concat([df, pd.DataFrame([{**values, "_k": k}])], ignore_index=True)
                stats["added"] += 1
    df = df[~df["brand"].isin(EXCLUDED_BRANDS) & ~df["model"].isin(EXCLUDED_MODELS)].reset_index(drop=True)
    return df, stats


def match_model(raw: str, keys: list[tuple[str, int]]) -> int | None:
    """Самое длинное известное название, целиком входящее в текст поста."""
    r = key(raw)
    best = None
    for k, idx in keys:
        if k and k in r and (best is None or len(k) > best[0]):
            best = (len(k), idx)
    return best[1] if best else None


def collect_prices(df: pd.DataFrame) -> tuple[dict, list]:
    keys = []
    for i, row in df.iterrows():
        for k in {key(row["model"]), key(f"{row['brand']} {row['model']}")} | set(ALIASES.get(key(row["model"]), [])):
            if len(k) >= 5:
                keys.append((k, i))
    prices, unmatched = {}, []
    for channel in ("ishopper", "polzaskidki"):
        path = SRC / f"{channel}.json"
        if not path.exists():
            continue
        for it in json.loads(path.read_text(encoding="utf-8")):
            if EXCLUDED_RAW.search(it["name"]):
                continue
            idx = match_model(it["name"], keys)
            if idx is None:
                unmatched.append(it["name"])
                continue
            prices.setdefault(idx, []).append((it["date"], it["price"], channel))
    return prices, unmatched


def pick_price(obs: list[tuple[str, int, str]]) -> tuple[int, str] | None:
    vals = sorted(p for _, p, _ in obs)
    med = vals[len(vals) // 2]
    obs = [o for o in obs if 0.55 * med <= o[1] <= 1.8 * med]  # отсечь варианты с другой памятью и опечатки
    fresh = [o for o in obs if o[0] >= FRESH_FROM]
    channels = lambda xs: "+".join(sorted({c for *_, c in xs}))
    if fresh:
        v = sorted(p for _, p, _ in fresh)
        return int(v[len(v) // 2]), f"Telegram ({channels(fresh)}): медиана {len(fresh)} цен за {FRESH_FROM[5:]}–11-05.2026, без пошлины"
    if obs:
        last = max(obs)
        return int(last[1]), f"Telegram ({last[2]}): последняя цена {last[0]}, без пошлины"
    return None


def main():
    df, stats = load_specs()
    prices, unmatched = collect_prices(df)
    updated, history = 0, []
    for idx, obs in prices.items():
        vals = sorted(p for _, p, _ in obs)
        med = vals[len(vals) // 2]
        history += [{"model": df.loc[idx, "model"], "date": d, "price_rub": p, "source": f"telegram:{ch}"}
                    for d, p, ch in obs if 0.55 * med <= p <= 1.8 * med]
        res = pick_price(obs)
        if res:
            df.loc[idx, "price_rub"], df.loc[idx, "price_source"] = res
            updated += 1
    df = df[COLUMNS].sort_values(["brand", "price_rub"], ascending=[True, False])
    for c in ["year", "price_rub", "ram_gb", "rom_gb", "battery_mah", "refresh_hz", "weight_g", "camera_mp",
              "antutu_k", "charge_w", "has_5g", "nfc", "esim", "ir_port"]:
        df[c] = df[c].astype(int)
    assert df["model"].is_unique, df[df["model"].duplicated()]["model"].tolist()
    df.to_csv(OUT, index=False)
    hist = pd.DataFrame(history).sort_values(["model", "date"])
    hist = hist.groupby(["model", "date", "source"], as_index=False)["price_rub"].min()  # один день — одна цена
    hist.to_csv(HISTORY_OUT, index=False)
    print(f"История цен: {len(hist)} точек по {hist['model'].nunique()} моделям → {HISTORY_OUT.relative_to(ROOT)}")
    print(f"База августа: {stats['base']} моделей; характеристики обновлены у {stats['updated_specs']}, "
          f"добавлено новых {stats['added']}")
    print(f"Цены из Telegram: у {updated} моделей; с августовской оценкой осталось {len(df) - updated}")
    print(f"Итого: {len(df)} моделей → {OUT.relative_to(ROOT)}")
    if "--unmatched" in sys.argv:
        from collections import Counter
        brands = re.compile(r"(iphone|galaxy|xiaomi|redmi|poco|realme|oneplus|nothing|vivo|iqoo|oppo|honor|moto|"
                            r"pixel|tecno|infinix|nubia|redmagic|lenovo|zte|meizu|cmf)", re.I)
        c = Counter(" ".join(n.lower().split()[:5]) for n in unmatched if brands.search(n))
        print("\nНе сопоставлено (похоже на телефоны):")
        for n, cnt in c.most_common(80):
            print(f"  {cnt:3d}  {n}")


if __name__ == "__main__":
    main()

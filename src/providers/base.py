"""Общий интерфейс источников цен.

Официальные API Яндекс Маркета, Ozon, Wildberries и Avito — продавцовые: ключ выдаётся в кабинете
продавца, и API отдаёт цены только по товарам этого кабинета. Поэтому адаптер
обновляет цены тех моделей из нашей БД, которые нашлись среди товаров кабинета.
"""
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path

from dotenv import load_dotenv

from src.paths import ENV_PATH  # noqa: E402  (переэкспорт для UI)


class ProviderError(Exception):
    pass


def load_env() -> None:
    load_dotenv(ENV_PATH, override=True)


def env(name: str) -> str:
    return os.getenv(name, "").strip()


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9а-я+]", "", name.lower())


class PriceProvider(ABC):
    name: str = "base"

    @abstractmethod
    def is_configured(self) -> bool: ...

    @abstractmethod
    def fetch_offers(self) -> list[tuple[str, float]]:
        """Список (название товара, цена в ₽)."""

    def match_prices(self, phones) -> dict[int, float]:
        """Сопоставление товаров магазина с моделями из БД по нормализованному названию."""
        keys = [(normalize(m), int(i)) for m, i in zip(phones["model"], phones["id"]) if normalize(m)]
        found = {}
        for title, price in self.fetch_offers():
            if not price:
                continue
            t = normalize(title)
            # товар относится к самой длинной подходящей модели: «Galaxy S25 Ultra» — не к «Galaxy S25»
            hits = [(len(k), pid) for k, pid in keys if k in t]
            if hits:
                pid = max(hits)[1]
                found[pid] = min(found.get(pid, price), price)
        return found


def all_providers() -> list[PriceProvider]:
    from src.providers.avito import AvitoProvider
    from src.providers.ozon import OzonSellerProvider
    from src.providers.wildberries import WildberriesProvider
    from src.providers.yandex_market import YandexMarketProvider
    load_env()
    return [YandexMarketProvider(), OzonSellerProvider(), WildberriesProvider(), AvitoProvider()]

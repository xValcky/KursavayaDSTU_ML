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
        offers = [(normalize(title), price) for title, price in self.fetch_offers() if price]
        found = {}
        for _, p in phones.iterrows():
            key = normalize(p["model"])
            prices = [price for title, price in offers if key and key in title]
            if prices:
                found[int(p["id"])] = min(prices)
        return found


def all_providers() -> list[PriceProvider]:
    from src.providers.avito import AvitoProvider
    from src.providers.ozon import OzonSellerProvider
    from src.providers.wildberries import WildberriesProvider
    from src.providers.yandex_market import YandexMarketProvider
    load_env()
    return [YandexMarketProvider(), OzonSellerProvider(), WildberriesProvider(), AvitoProvider()]

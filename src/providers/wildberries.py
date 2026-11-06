"""Wildberries: API цен и скидок для продавцов (https://dev.wildberries.ru/openapi/work-with-products).

.env:
  WB_API_TOKEN=...   # токен из личного кабинета продавца (категория «Цены и скидки»)
"""
import requests

from src.providers.base import PriceProvider, ProviderError, env

URL = "https://discounts-prices-api.wildberries.ru/api/v2/list/goods/filter"


class WildberriesProvider(PriceProvider):
    name = "Wildberries"

    def is_configured(self) -> bool:
        return bool(env("WB_API_TOKEN"))

    def fetch_offers(self) -> list[tuple[str, float]]:
        headers = {"Authorization": env("WB_API_TOKEN")}
        offers, offset = [], 0
        for _ in range(50):
            try:
                r = requests.get(URL, headers=headers, params={"limit": 1000, "offset": offset}, timeout=20)
            except requests.RequestException as e:
                raise ProviderError(f"{self.name}: нет соединения ({e})") from e
            if r.status_code in (401, 403):
                raise ProviderError(f"{self.name}: ключ не принят (HTTP {r.status_code})")
            if not r.ok:
                raise ProviderError(f"{self.name}: HTTP {r.status_code} {r.text[:200]}")
            goods = (r.json().get("data") or {}).get("listGoods") or []
            for g in goods:
                sizes = g.get("sizes") or [{}]
                price = sizes[0].get("discountedPrice") or sizes[0].get("price")
                # в API цен нет названия — используем артикул продавца (vendorCode)
                offers.append((g.get("vendorCode", ""), float(price) if price else 0.0))
            if len(goods) < 1000:
                break
            offset += 1000
        return offers

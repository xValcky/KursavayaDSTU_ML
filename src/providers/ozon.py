"""Ozon Seller API (https://docs.ozon.ru/api/seller/).

.env:
  OZON_CLIENT_ID=...
  OZON_API_KEY=...
"""
import requests

from src.providers.base import PriceProvider, ProviderError, env

BASE_URL = "https://api-seller.ozon.ru"


class OzonSellerProvider(PriceProvider):
    name = "Ozon"

    def is_configured(self) -> bool:
        return bool(env("OZON_CLIENT_ID") and env("OZON_API_KEY"))

    def _post(self, path: str, body: dict) -> dict:
        headers = {"Client-Id": env("OZON_CLIENT_ID"), "Api-Key": env("OZON_API_KEY")}
        try:
            r = requests.post(BASE_URL + path, headers=headers, json=body, timeout=20)
        except requests.RequestException as e:
            raise ProviderError(f"{self.name}: нет соединения ({e})") from e
        if r.status_code in (401, 403):
            raise ProviderError(f"{self.name}: ключ не принят (HTTP {r.status_code})")
        if not r.ok:
            raise ProviderError(f"{self.name}: HTTP {r.status_code} {r.text[:200]}")
        return r.json()

    def fetch_offers(self) -> list[tuple[str, float]]:
        offers, cursor = [], ""
        for _ in range(50):
            data = self._post("/v5/product/info/prices",
                              {"filter": {"visibility": "ALL"}, "limit": 1000, "cursor": cursor})
            items = data.get("items", [])
            for it in items:
                price = (it.get("price") or {}).get("price")
                # в ответе цен нет названия — используем артикул продавца (offer_id)
                offers.append((it.get("offer_id", ""), float(price) if price else 0.0))
            cursor = data.get("cursor", "")
            if not cursor or not items:
                break
        return offers

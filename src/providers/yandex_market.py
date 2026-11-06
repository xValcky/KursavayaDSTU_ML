"""Яндекс Маркет Partner API (https://yandex.ru/dev/market/partner-api/).

.env:
  YANDEX_MARKET_API_KEY=...       # Api-Key из кабинета продавца
  YANDEX_MARKET_BUSINESS_ID=...   # ID кабинета (бизнеса)
"""
import requests

from src.providers.base import PriceProvider, ProviderError, env

BASE_URL = "https://api.partner.market.yandex.ru"


class YandexMarketProvider(PriceProvider):
    name = "Яндекс Маркет"

    def is_configured(self) -> bool:
        return bool(env("YANDEX_MARKET_API_KEY") and env("YANDEX_MARKET_BUSINESS_ID"))

    def fetch_offers(self) -> list[tuple[str, float]]:
        url = f"{BASE_URL}/v2/businesses/{env('YANDEX_MARKET_BUSINESS_ID')}/offer-mappings"
        headers = {"Api-Key": env("YANDEX_MARKET_API_KEY"), "Content-Type": "application/json"}
        offers, page_token = [], None
        for _ in range(50):  # защита от бесконечной пагинации
            params = {"limit": 200, **({"page_token": page_token} if page_token else {})}
            try:
                r = requests.post(url, headers=headers, params=params, json={}, timeout=20)
            except requests.RequestException as e:
                raise ProviderError(f"{self.name}: нет соединения ({e})") from e
            if r.status_code in (401, 403):
                raise ProviderError(f"{self.name}: ключ не принят (HTTP {r.status_code})")
            if not r.ok:
                raise ProviderError(f"{self.name}: HTTP {r.status_code} {r.text[:200]}")
            result = r.json().get("result", {})
            for item in result.get("offerMappings", []):
                offer = item.get("offer", {})
                price = (offer.get("basicPrice") or {}).get("value")
                offers.append((offer.get("name", ""), float(price) if price else 0.0))
            page_token = (result.get("paging") or {}).get("nextPageToken")
            if not page_token:
                break
        return offers

"""Avito API для бизнеса (https://developers.avito.ru/api-catalog).

.env:
  AVITO_CLIENT_ID=...
  AVITO_CLIENT_SECRET=...   # из раздела «Для профессионалов» → API
Возвращает объявления своего аккаунта с ценами.
"""
import requests

from src.providers.base import PriceProvider, ProviderError, env

TOKEN_URL = "https://api.avito.ru/token"
ITEMS_URL = "https://api.avito.ru/core/v1/items"


class AvitoProvider(PriceProvider):
    name = "Avito"

    def is_configured(self) -> bool:
        return bool(env("AVITO_CLIENT_ID") and env("AVITO_CLIENT_SECRET"))

    def _token(self) -> str:
        try:
            r = requests.post(TOKEN_URL, data={"grant_type": "client_credentials",
                                               "client_id": env("AVITO_CLIENT_ID"),
                                               "client_secret": env("AVITO_CLIENT_SECRET")}, timeout=20)
        except requests.RequestException as e:
            raise ProviderError(f"{self.name}: нет соединения ({e})") from e
        if not r.ok or "access_token" not in r.text:
            detail = r.text[:120] if r.ok else f"HTTP {r.status_code}"
            raise ProviderError(f"{self.name}: ключ не принят ({detail})")
        return r.json()["access_token"]

    def fetch_offers(self) -> list[tuple[str, float]]:
        headers = {"Authorization": f"Bearer {self._token()}"}
        offers = []
        for page in range(1, 51):
            try:
                r = requests.get(ITEMS_URL, headers=headers, params={"per_page": 100, "page": page}, timeout=20)
            except requests.RequestException as e:
                raise ProviderError(f"{self.name}: нет соединения ({e})") from e
            if r.status_code in (401, 403):
                raise ProviderError(f"{self.name}: ключ не принят (HTTP {r.status_code})")
            if not r.ok:
                raise ProviderError(f"{self.name}: HTTP {r.status_code} {r.text[:200]}")
            items = r.json().get("resources") or []
            offers += [(it.get("title", ""), float(it.get("price") or 0)) for it in items]
            if len(items) < 100:
                break
        return offers

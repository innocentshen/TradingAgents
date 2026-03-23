from typing import Any

import requests

from tradingagents.asset_utils import normalize_symbol

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
REQUEST_TIMEOUT_SECONDS = 15

QUOTE_SUFFIXES = [
    "USDT",
    "USDC",
    "FDUSD",
    "BUSD",
    "USD",
    "BTC",
    "ETH",
    "BNB",
    "EUR",
    "GBP",
    "JPY",
]

COIN_ID_ALIASES = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "DOGE": "dogecoin",
    "ADA": "cardano",
    "TRX": "tron",
    "DOT": "polkadot",
    "AVAX": "avalanche-2",
    "LINK": "chainlink",
    "MATIC": "matic-network",
    "BCH": "bitcoin-cash",
    "LTC": "litecoin",
    "ATOM": "cosmos",
}


class CoinGeckoAPIError(Exception):
    """Raised when CoinGecko public API requests fail."""


def _request_json(endpoint: str, params: dict[str, Any] | None = None) -> Any:
    url = f"{COINGECKO_BASE_URL}{endpoint}"
    try:
        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={"accept": "application/json"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise CoinGeckoAPIError(
            f"CoinGecko request failed for {endpoint}: {exc}"
        ) from exc

    try:
        return response.json()
    except ValueError as exc:
        raise CoinGeckoAPIError(
            f"CoinGecko returned a non-JSON response for {endpoint}: {response.text[:300]}"
        ) from exc


def _extract_base_symbol(symbol: str) -> str:
    normalized = normalize_symbol(symbol)
    if normalized.startswith("CRYPTO:"):
        normalized = normalized.split(":", 1)[1]

    normalized = normalized.replace("/", "-")

    if "-" in normalized:
        return normalized.split("-", 1)[0]

    for quote_suffix in QUOTE_SUFFIXES:
        if normalized.endswith(quote_suffix) and len(normalized) > len(quote_suffix):
            return normalized[: -len(quote_suffix)]

    return normalized


def _pick_best_search_result(coins: list[dict[str, Any]], base_symbol: str) -> dict[str, Any] | None:
    exact_matches = [
        coin for coin in coins if str(coin.get("symbol", "")).upper() == base_symbol
    ]

    candidates = exact_matches or coins
    if not candidates:
        return None

    def sort_key(coin: dict[str, Any]) -> tuple[int, int]:
        rank = coin.get("market_cap_rank")
        if rank in (None, 0):
            return (1, 10**9)
        return (0, int(rank))

    return sorted(candidates, key=sort_key)[0]


def _resolve_coin_id(symbol: str) -> str:
    base_symbol = _extract_base_symbol(symbol)
    if base_symbol in COIN_ID_ALIASES:
        return COIN_ID_ALIASES[base_symbol]

    payload = _request_json("/search", {"query": base_symbol})
    best_match = _pick_best_search_result(payload.get("coins", []), base_symbol)
    if not best_match:
        raise CoinGeckoAPIError(
            f"CoinGecko could not resolve a coin ID for symbol '{symbol}'"
        )
    return best_match["id"]


def get_coin_snapshot(symbol: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return CoinGecko market and metadata snapshots for a crypto symbol."""
    coin_id = _resolve_coin_id(symbol)

    markets = _request_json(
        "/coins/markets",
        {
            "vs_currency": "usd",
            "ids": coin_id,
            "sparkline": "false",
            "price_change_percentage": "24h,7d,30d",
            "precision": "full",
        },
    )
    market_data = markets[0] if isinstance(markets, list) and markets else None

    details = _request_json(
        f"/coins/{coin_id}",
        {
            "localization": "false",
            "tickers": "false",
            "market_data": "false",
            "community_data": "true",
            "developer_data": "true",
            "sparkline": "false",
        },
    )

    return market_data, details

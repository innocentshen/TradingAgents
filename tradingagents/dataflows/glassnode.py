import os
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from tradingagents.asset_utils import normalize_symbol

GLASSNODE_BASE_URL = "https://api.glassnode.com"
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

GLASSNODE_ASSET_ALIASES = {
    "XBT": "BTC",
}


class GlassnodeAPIError(Exception):
    """Raised when Glassnode API requests fail or are not configured."""


def _get_api_key() -> str:
    api_key = os.getenv("GLASSNODE_API_KEY") or os.getenv(
        "TRADINGAGENTS_GLASSNODE_API_KEY"
    )
    if not api_key:
        raise GlassnodeAPIError(
            "Glassnode skipped: GLASSNODE_API_KEY environment variable is not set."
        )
    return api_key


def _extract_asset_id(symbol: str) -> str:
    normalized = normalize_symbol(symbol)
    if normalized.startswith("CRYPTO:"):
        normalized = normalized.split(":", 1)[1]

    normalized = normalized.replace("/", "-")

    if "-" in normalized:
        base_symbol = normalized.split("-", 1)[0]
    else:
        base_symbol = normalized
        for quote_suffix in QUOTE_SUFFIXES:
            if normalized.endswith(quote_suffix) and len(normalized) > len(quote_suffix):
                base_symbol = normalized[: -len(quote_suffix)]
                break

    return GLASSNODE_ASSET_ALIASES.get(base_symbol, base_symbol)


def _request_json(endpoint: str, params: dict[str, Any]) -> Any:
    response = None
    try:
        response = requests.get(
            f"{GLASSNODE_BASE_URL}{endpoint}",
            params={
                **params,
                "api_key": _get_api_key(),
                "f": "json",
                "timestamp_format": "unix",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={"accept": "application/json"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        response_text = ""
        if response is not None:
            response_text = response.text[:300]
        raise GlassnodeAPIError(
            f"Glassnode request failed for {endpoint}: {exc}. {response_text}".strip()
        ) from exc

    try:
        return response.json()
    except ValueError as exc:
        raise GlassnodeAPIError(
            f"Glassnode returned a non-JSON response for {endpoint}: {response.text[:300]}"
        ) from exc


def _safe_request_json(
    endpoint: str, params: dict[str, Any]
) -> tuple[Any | None, str | None]:
    try:
        return _request_json(endpoint, params), None
    except GlassnodeAPIError as exc:
        return None, str(exc)


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_points(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [point for point in payload if isinstance(point, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _latest_numeric_value(points: list[dict[str, Any]]) -> float | None:
    for point in reversed(points):
        numeric_value = _to_float(point.get("v"))
        if numeric_value is not None:
            return numeric_value
    return None


def _window_change(values: list[float], periods: int) -> float | None:
    if len(values) <= periods:
        return None
    start_value = values[-(periods + 1)]
    end_value = values[-1]
    if start_value in (None, 0):
        return None
    return (end_value / start_value - 1) * 100


def get_onchain_snapshot(
    symbol: str, lookback_days: int = 35
) -> dict[str, Any]:
    """Return optional Glassnode on-chain metrics for a crypto asset."""
    asset_id = _extract_asset_id(symbol)
    snapshot: dict[str, Any] = {
        "asset_id": asset_id,
        "configured": False,
        "metrics": {},
        "warnings": [],
    }

    try:
        _get_api_key()
    except GlassnodeAPIError as exc:
        snapshot["warnings"].append(str(exc))
        return snapshot

    snapshot["configured"] = True

    now = datetime.now(timezone.utc)
    since = int((now - timedelta(days=max(lookback_days, 35))).timestamp())
    until = int(now.timestamp())

    metric_specs = [
        (
            "active_addresses",
            "/v1/metrics/addresses/active_count",
            {},
        ),
        (
            "mvrv",
            "/v1/metrics/market/mvrv",
            {},
        ),
        (
            "realized_price_usd",
            "/v1/metrics/market/price_realized_usd",
            {},
        ),
        (
            "nupl",
            "/v1/metrics/indicators/net_unrealized_profit_loss",
            {},
        ),
        (
            "sopr",
            "/v1/metrics/indicators/sopr",
            {},
        ),
        (
            "exchange_inflow_usd",
            "/v1/metrics/transactions/transfers_volume_to_exchanges_sum",
            {"c": "USD"},
        ),
        (
            "exchange_outflow_usd",
            "/v1/metrics/transactions/transfers_volume_from_exchanges_sum",
            {"c": "USD"},
        ),
    ]

    if asset_id == "BTC":
        metric_specs.append(
            (
                "ssr",
                "/v1/metrics/indicators/ssr",
                {},
            )
        )

    base_params = {
        "a": asset_id,
        "s": since,
        "u": until,
        "i": "24h",
    }

    for index, (metric_name, endpoint, extra_params) in enumerate(metric_specs):
        payload, error = _safe_request_json(endpoint, {**base_params, **extra_params})
        if error:
            error_lower = error.lower()
            if index == 0 and (
                "unsupported asset" in error_lower
                or "invalid parameters" in error_lower
                or "400 client error" in error_lower
            ):
                snapshot["warnings"].append(
                    f"Glassnode skipped for {asset_id}: asset or plan does not support the requested on-chain metrics."
                )
                snapshot["configured"] = True
                return snapshot
            snapshot["warnings"].append(f"Glassnode {metric_name}: {error}")
            continue

        points = _extract_points(payload)
        values = [
            _to_float(point.get("v"))
            for point in points
            if _to_float(point.get("v")) is not None
        ]
        snapshot["metrics"][metric_name] = {
            "value": _latest_numeric_value(points),
            "timestamp": points[-1].get("t") if points else None,
            "points": len(values),
        }

        if metric_name == "active_addresses":
            snapshot["metrics"][metric_name]["change_30d"] = _window_change(values, 30)

    exchange_inflow = (
        snapshot["metrics"].get("exchange_inflow_usd", {}).get("value")
    )
    exchange_outflow = (
        snapshot["metrics"].get("exchange_outflow_usd", {}).get("value")
    )
    if exchange_inflow is not None and exchange_outflow is not None:
        snapshot["metrics"]["exchange_netflow_usd"] = {
            "value": exchange_inflow - exchange_outflow,
            "timestamp": max(
                snapshot["metrics"].get("exchange_inflow_usd", {}).get("timestamp") or 0,
                snapshot["metrics"].get("exchange_outflow_usd", {}).get("timestamp") or 0,
            ),
            "points": min(
                snapshot["metrics"].get("exchange_inflow_usd", {}).get("points") or 0,
                snapshot["metrics"].get("exchange_outflow_usd", {}).get("points") or 0,
            ),
        }

    return snapshot

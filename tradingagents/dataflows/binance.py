import math
import os
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

import pandas as pd
import requests
from stockstats import wrap

from tradingagents.asset_utils import normalize_symbol

from .coingecko import CoinGeckoAPIError, get_coin_snapshot
from .config import get_config
from .glassnode import get_onchain_snapshot
from .stockstats_utils import _clean_dataframe

SPOT_BASE_URL = "https://api.binance.com"
FUTURES_BASE_URL = "https://fapi.binance.com"
REQUEST_TIMEOUT_SECONDS = 15
KLINE_LIMIT = 1000

QUOTE_NORMALIZATION = {
    "USD": "USDT",
    "USDT": "USDT",
    "USDC": "USDC",
    "FDUSD": "FDUSD",
    "BTC": "BTC",
    "ETH": "ETH",
    "BNB": "BNB",
    "EUR": "EUR",
    "JPY": "JPY",
    "GBP": "GBP",
}

INTERVAL_TO_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
    "3d": 3 * 24 * 60 * 60_000,
    "1w": 7 * 24 * 60 * 60_000,
    "1M": 30 * 24 * 60 * 60_000,
}


class BinanceAPIError(Exception):
    """Exception raised when Binance public market-data requests fail."""


def _normalize_binance_symbol(symbol: str) -> str:
    normalized = normalize_symbol(symbol)
    if normalized.startswith("CRYPTO:"):
        normalized = normalized.split(":", 1)[1]

    normalized = normalized.replace("/", "-")

    if "-" in normalized:
        base, quote = normalized.rsplit("-", 1)
        quote = QUOTE_NORMALIZATION.get(quote, quote)
        return f"{base}{quote}"

    return normalized


def _request_json(base_url: str, endpoint: str, params: dict[str, Any]) -> Any:
    url = f"{base_url}{endpoint}"
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise BinanceAPIError(f"Binance request failed for {endpoint}: {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise BinanceAPIError(
            f"Binance returned a non-JSON response for {endpoint}: {response.text[:300]}"
        ) from exc

    if isinstance(data, dict) and isinstance(data.get("code"), int) and data["code"] < 0:
        raise BinanceAPIError(
            f"Binance API error for {endpoint}: {data.get('code')} {data.get('msg')}"
        )

    return data


def _safe_request_json(base_url: str, endpoint: str, params: dict[str, Any]) -> tuple[Any | None, str | None]:
    try:
        return _request_json(base_url, endpoint, params), None
    except BinanceAPIError as exc:
        return None, str(exc)


def _date_to_millis(date_str: str, end_of_day: bool = False) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end_of_day:
        dt = dt + timedelta(days=1) - timedelta(milliseconds=1)
    return int(dt.timestamp() * 1000)


def _format_utc_timestamp(ms: int | str | None) -> str:
    if ms in (None, ""):
        return "N/A"
    dt = datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_float(value: Any, precision: int = 6, suffix: str = "") -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "N/A"
    return f"{numeric:,.{precision}f}{suffix}"


def _format_percent(value: Any, precision: int = 2) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "N/A"
    return f"{numeric:.{precision}f}%"


def _format_ratio(value: Any, precision: int = 4) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "N/A"
    return f"{numeric:.{precision}f}"


def _format_int(value: Any) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "N/A"
    return f"{int(round(numeric)):,}"


def _format_text(value: Any) -> str:
    if value in (None, ""):
        return "N/A"
    text = str(value).strip()
    return text or "N/A"


def _format_list(values: Any, limit: int = 5) -> str:
    if not values:
        return "N/A"
    if isinstance(values, str):
        return _format_text(values)

    items = [str(value).strip() for value in values if str(value).strip()]
    if not items:
        return "N/A"
    if len(items) > limit:
        return f"{', '.join(items[:limit])}, +{len(items) - limit} more"
    return ", ".join(items)


def _fetch_klines(
    symbol: str,
    interval: str,
    start_time: int,
    end_time: int,
) -> list[list[Any]]:
    binance_symbol = _normalize_binance_symbol(symbol)
    interval_ms = INTERVAL_TO_MS[interval]
    current_start = start_time
    all_rows: list[list[Any]] = []

    while current_start <= end_time:
        batch = _request_json(
            SPOT_BASE_URL,
            "/api/v3/klines",
            {
                "symbol": binance_symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_time,
                "limit": KLINE_LIMIT,
            },
        )

        if not batch:
            break

        all_rows.extend(batch)

        last_open_time = int(batch[-1][0])
        next_start = last_open_time + interval_ms
        if next_start <= current_start:
            break
        current_start = next_start

        if len(batch) < KLINE_LIMIT:
            break

    deduped_rows = {int(row[0]): row for row in all_rows}
    return [deduped_rows[key] for key in sorted(deduped_rows)]


def _fetch_recent_klines(symbol: str, interval: str, limit: int) -> list[list[Any]]:
    return _request_json(
        SPOT_BASE_URL,
        "/api/v3/klines",
        {
            "symbol": _normalize_binance_symbol(symbol),
            "interval": interval,
            "limit": limit,
        },
    )


def _klines_to_dataframe(klines: list[list[Any]]) -> pd.DataFrame:
    if not klines:
        return pd.DataFrame(
            columns=[
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
                "QuoteVolume",
                "TradeCount",
                "TakerBuyBaseVolume",
                "TakerBuyQuoteVolume",
            ]
        )

    df = pd.DataFrame(
        klines,
        columns=[
            "OpenTime",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "CloseTime",
            "QuoteVolume",
            "TradeCount",
            "TakerBuyBaseVolume",
            "TakerBuyQuoteVolume",
            "Ignore",
        ],
    )

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "QuoteVolume",
        "TradeCount",
        "TakerBuyBaseVolume",
        "TakerBuyQuoteVolume",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Date"] = (
        pd.to_datetime(df["OpenTime"], unit="ms", utc=True).dt.tz_convert(None)
    )
    df["Adj Close"] = df["Close"]

    return df[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
            "QuoteVolume",
            "TradeCount",
            "TakerBuyBaseVolume",
            "TakerBuyQuoteVolume",
        ]
    ].copy()


def _cache_file_path(symbol: str, interval: str, start_date: str, end_date: str) -> str:
    config = get_config()
    os.makedirs(config["data_cache_dir"], exist_ok=True)
    binance_symbol = _normalize_binance_symbol(symbol)
    return os.path.join(
        config["data_cache_dir"],
        f"{binance_symbol}-BINANCE-{interval}-data-{start_date}-{end_date}.csv",
    )


def _load_or_download_klines(symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
    file_path = _cache_file_path(symbol, interval, start_date, end_date)

    if os.path.exists(file_path):
        return pd.read_csv(file_path, on_bad_lines="skip")

    klines = _fetch_klines(
        symbol,
        interval,
        _date_to_millis(start_date),
        _date_to_millis(end_date, end_of_day=True),
    )

    if not klines:
        raise BinanceAPIError(
            f"No Binance kline data found for {_normalize_binance_symbol(symbol)} between "
            f"{start_date} and {end_date}"
        )

    df = _klines_to_dataframe(klines)
    df.to_csv(file_path, index=False)
    return df


def _realized_volatility(close_series: pd.Series, periods_per_year: int) -> float | None:
    returns = close_series.pct_change().dropna()
    if returns.empty:
        return None
    return float(returns.std(ddof=0) * math.sqrt(periods_per_year) * 100)


def _window_return(close_series: pd.Series, periods: int) -> float | None:
    if len(close_series) <= periods:
        return None
    start_value = close_series.iloc[-(periods + 1)]
    end_value = close_series.iloc[-1]
    if start_value in (0, None):
        return None
    return float((end_value / start_value - 1) * 100)


def _range_percent(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None
    low = _to_float(df["Low"].min())
    high = _to_float(df["High"].max())
    if low in (None, 0) or high is None:
        return None
    return float((high / low - 1) * 100)


def _latest_dict_entry(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, list) and payload:
        return payload[-1]
    if isinstance(payload, dict):
        return payload
    return None


def _first_non_empty(values: list[Any]) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _depth_metrics(
    depth_snapshot: dict[str, Any] | None, levels: int = 10
) -> tuple[float | None, float | None, float | None]:
    if not isinstance(depth_snapshot, dict):
        return None, None, None

    bid_notional = 0.0
    ask_notional = 0.0
    has_bid_data = False
    has_ask_data = False

    for price, quantity, *_ in depth_snapshot.get("bids", [])[:levels]:
        numeric_price = _to_float(price)
        numeric_quantity = _to_float(quantity)
        if numeric_price is None or numeric_quantity is None:
            continue
        bid_notional += numeric_price * numeric_quantity
        has_bid_data = True

    for price, quantity, *_ in depth_snapshot.get("asks", [])[:levels]:
        numeric_price = _to_float(price)
        numeric_quantity = _to_float(quantity)
        if numeric_price is None or numeric_quantity is None:
            continue
        ask_notional += numeric_price * numeric_quantity
        has_ask_data = True

    if not has_bid_data and not has_ask_data:
        return None, None, None

    total_notional = bid_notional + ask_notional
    imbalance_pct = None
    if total_notional:
        imbalance_pct = (bid_notional - ask_notional) / total_notional * 100

    return (
        bid_notional if has_bid_data else None,
        ask_notional if has_ask_data else None,
        imbalance_pct,
    )


def _agg_trade_metrics(
    agg_trades: Any,
) -> tuple[float | None, float | None, float | None, float | None]:
    if not isinstance(agg_trades, list) or not agg_trades:
        return None, None, None, None

    taker_buy_notional = 0.0
    taker_sell_notional = 0.0

    for trade in agg_trades:
        price = _to_float(trade.get("p"))
        quantity = _to_float(trade.get("q"))
        if price is None or quantity is None:
            continue

        notional = price * quantity
        if trade.get("m"):
            taker_sell_notional += notional
        else:
            taker_buy_notional += notional

    total_notional = taker_buy_notional + taker_sell_notional
    if not total_notional:
        return None, None, None, None

    taker_buy_share = taker_buy_notional / total_notional * 100
    net_aggressor_balance = (
        (taker_buy_notional - taker_sell_notional) / total_notional * 100
    )

    return (
        taker_buy_notional,
        taker_sell_notional,
        taker_buy_share,
        net_aggressor_balance,
    )


def get_stock(
    symbol: Annotated[str, "ticker symbol of the asset"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Retrieve Binance spot OHLCV data for a crypto ticker."""
    binance_symbol = _normalize_binance_symbol(symbol)
    klines = _fetch_klines(
        symbol,
        "1d",
        _date_to_millis(start_date),
        _date_to_millis(end_date, end_of_day=True),
    )

    if not klines:
        return f"No Binance spot data found for symbol '{binance_symbol}' between {start_date} and {end_date}"

    df = _klines_to_dataframe(klines)
    csv_string = df.to_csv(index=False)

    header = f"# Binance spot price data for {binance_symbol} from {start_date} to {end_date}\n"
    header += f"# Original input symbol: {symbol}\n"
    header += f"# Total records: {len(df)}\n"
    header += f"# Data retrieved on: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
    return header + csv_string


def get_indicator(
    symbol: Annotated[str, "ticker symbol of the asset"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """Calculate stockstats indicators from cached Binance spot daily klines."""
    best_ind_params = {
        "close_50_sma": "50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.",
        "close_200_sma": "200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.",
        "close_10_ema": "10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.",
        "macd": "MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.",
        "macds": "MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.",
        "macdh": "MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.",
        "rsi": "RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.",
        "boll": "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.",
        "boll_ub": "Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.",
        "boll_lb": "Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.",
        "atr": "ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.",
        "vwma": "VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses.",
        "mfi": "MFI: A momentum indicator using price and volume to measure buying and selling pressure. Usage: Identify overbought (>80) or oversold (<20) conditions and confirm trend strength or reversals. Tips: Use alongside RSI or MACD to confirm signals; divergence can indicate reversals.",
    }

    if indicator not in best_ind_params:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(best_ind_params.keys())}"
        )

    today = pd.Timestamp.now(tz="UTC").tz_convert(None)
    start_date = (today - pd.DateOffset(years=10)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    data = _load_or_download_klines(symbol, "1d", start_date, end_date)
    data = _clean_dataframe(data)

    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df[indicator]

    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - timedelta(days=look_back_days)

    lines = []
    current_dt = curr_date_dt
    while current_dt >= before:
        date_str = current_dt.strftime("%Y-%m-%d")
        matching_rows = df[df["Date"] == date_str]
        if not matching_rows.empty:
            value = matching_rows.iloc[0][indicator]
            if pd.isna(value):
                lines.append(f"{date_str}: N/A")
            else:
                lines.append(f"{date_str}: {value}")
        else:
            lines.append(f"{date_str}: N/A")
        current_dt -= timedelta(days=1)

    return (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {curr_date} "
        f"(Binance spot daily):\n\n"
        + "\n".join(lines)
        + "\n\n"
        + best_ind_params[indicator]
    )


def _not_applicable_report(symbol: str, dataset_name: str) -> str:
    return (
        f"# {dataset_name} for {_normalize_binance_symbol(symbol)}\n"
        f"# Data retrieved on: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
        "Not applicable: Binance public market data does not provide public-company financial "
        f"statement style {dataset_name.lower()} data for crypto assets.\n\n"
        "Use the Binance crypto market-structure overview instead."
    )


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol of the asset"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """Build a crypto market-structure overview from Binance, with CoinGecko token metadata."""
    binance_symbol = _normalize_binance_symbol(ticker)
    warnings: list[str] = []

    spot_24h = _request_json(SPOT_BASE_URL, "/api/v3/ticker/24hr", {"symbol": binance_symbol})
    book_ticker = _request_json(SPOT_BASE_URL, "/api/v3/ticker/bookTicker", {"symbol": binance_symbol})
    avg_price, avg_price_error = _safe_request_json(
        SPOT_BASE_URL, "/api/v3/avgPrice", {"symbol": binance_symbol}
    )
    if avg_price_error:
        warnings.append(avg_price_error)

    depth_snapshot, depth_snapshot_error = _safe_request_json(
        SPOT_BASE_URL, "/api/v3/depth", {"symbol": binance_symbol, "limit": 20}
    )
    if depth_snapshot_error:
        warnings.append(depth_snapshot_error)

    agg_trades, agg_trades_error = _safe_request_json(
        SPOT_BASE_URL, "/api/v3/aggTrades", {"symbol": binance_symbol, "limit": 500}
    )
    if agg_trades_error:
        warnings.append(agg_trades_error)

    recent_daily = _klines_to_dataframe(_fetch_recent_klines(binance_symbol, "1d", 60))
    recent_4h = _klines_to_dataframe(_fetch_recent_klines(binance_symbol, "4h", 42))
    recent_1h = _klines_to_dataframe(_fetch_recent_klines(binance_symbol, "1h", 48))

    mark_price, mark_price_error = _safe_request_json(
        FUTURES_BASE_URL, "/fapi/v1/premiumIndex", {"symbol": binance_symbol}
    )
    if mark_price_error:
        warnings.append(mark_price_error)

    funding_history, funding_history_error = _safe_request_json(
        FUTURES_BASE_URL, "/fapi/v1/fundingRate", {"symbol": binance_symbol, "limit": 20}
    )
    if funding_history_error:
        warnings.append(funding_history_error)

    open_interest, open_interest_error = _safe_request_json(
        FUTURES_BASE_URL, "/fapi/v1/openInterest", {"symbol": binance_symbol}
    )
    if open_interest_error:
        warnings.append(open_interest_error)

    oi_history, oi_history_error = _safe_request_json(
        FUTURES_BASE_URL,
        "/futures/data/openInterestHist",
        {"symbol": binance_symbol, "period": "1d", "limit": 30},
    )
    if oi_history_error:
        warnings.append(oi_history_error)

    top_position_ratio, top_position_ratio_error = _safe_request_json(
        FUTURES_BASE_URL,
        "/futures/data/topLongShortPositionRatio",
        {"symbol": binance_symbol, "period": "1d", "limit": 30},
    )
    if top_position_ratio_error:
        warnings.append(top_position_ratio_error)

    top_account_ratio, top_account_ratio_error = _safe_request_json(
        FUTURES_BASE_URL,
        "/futures/data/topLongShortAccountRatio",
        {"symbol": binance_symbol, "period": "1d", "limit": 30},
    )
    if top_account_ratio_error:
        warnings.append(top_account_ratio_error)

    global_long_short_ratio, global_long_short_ratio_error = _safe_request_json(
        FUTURES_BASE_URL,
        "/futures/data/globalLongShortAccountRatio",
        {"symbol": binance_symbol, "period": "1d", "limit": 30},
    )
    if global_long_short_ratio_error:
        warnings.append(global_long_short_ratio_error)

    taker_flow, taker_flow_error = _safe_request_json(
        FUTURES_BASE_URL,
        "/futures/data/takerlongshortRatio",
        {"symbol": binance_symbol, "period": "1d", "limit": 30},
    )
    if taker_flow_error:
        warnings.append(taker_flow_error)

    coingecko_market = None
    coingecko_details = None
    try:
        coingecko_market, coingecko_details = get_coin_snapshot(ticker)
    except CoinGeckoAPIError as exc:
        warnings.append(str(exc))

    glassnode_snapshot = get_onchain_snapshot(ticker)
    warnings.extend(glassnode_snapshot.get("warnings", []))

    last_price = _to_float(spot_24h.get("lastPrice"))
    bid_price = _to_float(book_ticker.get("bidPrice"))
    ask_price = _to_float(book_ticker.get("askPrice"))
    mid_price = None
    spread_bps = None
    if bid_price is not None and ask_price is not None:
        mid_price = (bid_price + ask_price) / 2
        if mid_price:
            spread_bps = (ask_price - bid_price) / mid_price * 10_000

    daily_vol_30d = _realized_volatility(recent_daily["Close"].tail(31), 365)
    hourly_vol_24h = _realized_volatility(recent_1h["Close"].tail(25), 24 * 365)
    return_7d = _window_return(recent_daily["Close"], 7)
    return_30d = _window_return(recent_daily["Close"], 30)
    return_24h = _window_return(recent_1h["Close"], 24)
    range_7d_4h = _range_percent(recent_4h.tail(42))

    latest_mark = _latest_dict_entry(mark_price)
    latest_funding = _latest_dict_entry(funding_history)
    latest_oi = _latest_dict_entry(open_interest)
    latest_oi_hist = _latest_dict_entry(oi_history)
    latest_top_position = _latest_dict_entry(top_position_ratio)
    latest_top_account = _latest_dict_entry(top_account_ratio)
    latest_global_ratio = _latest_dict_entry(global_long_short_ratio)
    latest_taker_flow = _latest_dict_entry(taker_flow)

    latest_funding_rate = _first_non_empty(
        [
            _to_float((latest_mark or {}).get("lastFundingRate")),
            _to_float((latest_funding or {}).get("fundingRate")),
        ]
    )
    latest_funding_pct = (
        latest_funding_rate * 100 if latest_funding_rate is not None else None
    )

    funding_values = [
        _to_float(entry.get("fundingRate"))
        for entry in funding_history or []
        if _to_float(entry.get("fundingRate")) is not None
    ]
    avg_funding_7 = None
    if funding_values:
        sample = funding_values[-7:] if len(funding_values) >= 7 else funding_values
        avg_funding_7 = sum(sample) / len(sample) * 100

    oi_history_values = [
        _to_float(entry.get("sumOpenInterestValue"))
        for entry in oi_history or []
        if _to_float(entry.get("sumOpenInterestValue")) is not None
    ]
    oi_change_7d = None
    if len(oi_history_values) >= 8 and oi_history_values[-8] not in (None, 0):
        oi_change_7d = (oi_history_values[-1] / oi_history_values[-8] - 1) * 100

    binance_circulating_supply_proxy = None
    binance_estimated_market_cap = None
    if latest_oi_hist:
        binance_circulating_supply_proxy = _to_float(
            latest_oi_hist.get("CMCCirculatingSupply")
        )
        if binance_circulating_supply_proxy is not None and last_price is not None:
            binance_estimated_market_cap = binance_circulating_supply_proxy * last_price

    circulating_supply = _first_non_empty(
        [
            _to_float((coingecko_market or {}).get("circulating_supply")),
            binance_circulating_supply_proxy,
        ]
    )
    estimated_market_cap = _first_non_empty(
        [
            _to_float((coingecko_market or {}).get("market_cap")),
            binance_estimated_market_cap,
        ]
    )
    supply_source = "CoinGecko"
    if (
        circulating_supply is not None
        and circulating_supply == binance_circulating_supply_proxy
        and _to_float((coingecko_market or {}).get("circulating_supply")) is None
    ):
        supply_source = "Binance OI history proxy"
    elif circulating_supply is None:
        supply_source = "N/A"

    basis_pct = None
    if latest_mark and last_price not in (None, 0):
        mark_price_value = _to_float(latest_mark.get("markPrice"))
        if mark_price_value is not None:
            basis_pct = (mark_price_value / last_price - 1) * 100

    depth_bid_notional, depth_ask_notional, depth_imbalance_pct = _depth_metrics(
        depth_snapshot
    )
    (
        agg_taker_buy_notional,
        agg_taker_sell_notional,
        agg_taker_buy_share,
        agg_net_aggressor_balance,
    ) = _agg_trade_metrics(agg_trades)
    recent_agg_quote_notional = None
    if agg_taker_buy_notional is not None and agg_taker_sell_notional is not None:
        recent_agg_quote_notional = agg_taker_buy_notional + agg_taker_sell_notional

    coingecko_links = (coingecko_details or {}).get("links", {})
    coingecko_community = (coingecko_details or {}).get("community_data", {})
    coingecko_developer = (coingecko_details or {}).get("developer_data", {})
    coingecko_homepage = _first_non_empty(
        [homepage for homepage in coingecko_links.get("homepage", []) if homepage]
    )

    glassnode_metrics = glassnode_snapshot.get("metrics", {})
    active_addresses = _to_float(
        (glassnode_metrics.get("active_addresses") or {}).get("value")
    )
    active_addresses_change_30d = _to_float(
        (glassnode_metrics.get("active_addresses") or {}).get("change_30d")
    )
    onchain_realized_price = _to_float(
        (glassnode_metrics.get("realized_price_usd") or {}).get("value")
    )
    onchain_realized_price_gap_pct = None
    if onchain_realized_price not in (None, 0) and last_price not in (None, 0):
        onchain_realized_price_gap_pct = (last_price / onchain_realized_price - 1) * 100

    onchain_mvrv = _to_float((glassnode_metrics.get("mvrv") or {}).get("value"))
    onchain_nupl = _to_float((glassnode_metrics.get("nupl") or {}).get("value"))
    onchain_sopr = _to_float((glassnode_metrics.get("sopr") or {}).get("value"))
    onchain_exchange_inflow = _to_float(
        (glassnode_metrics.get("exchange_inflow_usd") or {}).get("value")
    )
    onchain_exchange_outflow = _to_float(
        (glassnode_metrics.get("exchange_outflow_usd") or {}).get("value")
    )
    onchain_exchange_netflow = _to_float(
        (glassnode_metrics.get("exchange_netflow_usd") or {}).get("value")
    )
    onchain_ssr = _to_float((glassnode_metrics.get("ssr") or {}).get("value"))
    onchain_status = "Enabled" if glassnode_metrics else "Skipped / unavailable"
    if glassnode_snapshot.get("configured") and not glassnode_metrics:
        onchain_status = "Configured, but no supported metrics returned"
    elif not glassnode_snapshot.get("configured"):
        onchain_status = "Disabled (set GLASSNODE_API_KEY to enable)"

    sections = [
        f"# Binance Crypto Market Structure Overview for {binance_symbol}",
        f"# Original input symbol: {ticker}",
        f"# Data retrieved on: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "This report uses Binance Spot market data for price discovery and Binance USD-M Futures public market data for leverage, positioning, and funding structure. It is intended as a crypto-native fundamentals proxy for trading research.",
        "",
        "## Spot Market Snapshot",
        f"- Last price: {_format_float(last_price, precision=6)}",
        f"- 24h open / high / low: {_format_float(spot_24h.get('openPrice'), precision=6)} / {_format_float(spot_24h.get('highPrice'), precision=6)} / {_format_float(spot_24h.get('lowPrice'), precision=6)}",
        f"- 24h base volume: {_format_float(spot_24h.get('volume'), precision=2)}",
        f"- 24h quote volume: {_format_float(spot_24h.get('quoteVolume'), precision=2)}",
        f"- Weighted average price: {_format_float(spot_24h.get('weightedAvgPrice'), precision=6)}",
        f"- 5-minute average price: {_format_float((avg_price or {}).get('price'), precision=6)}",
        f"- Best bid / ask: {_format_float(bid_price, precision=6)} / {_format_float(ask_price, precision=6)}",
        f"- Approximate spread: {_format_float(spread_bps, precision=2, suffix=' bps')}",
        "",
        "## Spot Liquidity and Recent Order Flow",
        f"- Top-10 bid notional from depth snapshot: {_format_float(depth_bid_notional, precision=2)}",
        f"- Top-10 ask notional from depth snapshot: {_format_float(depth_ask_notional, precision=2)}",
        f"- Top-10 book imbalance: {_format_percent(depth_imbalance_pct)}",
        f"- Recent aggregate trades sampled: {_format_int(len(agg_trades) if isinstance(agg_trades, list) else None)}",
        f"- Recent aggregate-trade quote notional: {_format_float(recent_agg_quote_notional, precision=2)}",
        f"- Recent taker-buy quote notional: {_format_float(agg_taker_buy_notional, precision=2)}",
        f"- Recent taker-sell quote notional: {_format_float(agg_taker_sell_notional, precision=2)}",
        f"- Recent taker-buy share: {_format_percent(agg_taker_buy_share)}",
        f"- Recent net aggressor balance: {_format_percent(agg_net_aggressor_balance)}",
        "",
        "## Multi-Timeframe Price Structure",
        f"- 24h return from 1h bars: {_format_percent(return_24h)}",
        f"- 7d return from daily bars: {_format_percent(return_7d)}",
        f"- 30d return from daily bars: {_format_percent(return_30d)}",
        f"- 7d intraperiod range from 4h bars: {_format_percent(range_7d_4h)}",
        f"- 30d realized volatility from daily bars: {_format_percent(daily_vol_30d)}",
        f"- 24h realized volatility from 1h bars: {_format_percent(hourly_vol_24h)}",
        "",
        "## Futures Leverage and Positioning",
        f"- Mark / index price: {_format_float((latest_mark or {}).get('markPrice'), precision=6)} / {_format_float((latest_mark or {}).get('indexPrice'), precision=6)}",
        f"- Spot-to-mark basis: {_format_percent(basis_pct)}",
        f"- Latest funding rate: {_format_percent(latest_funding_pct, precision=4)}",
        f"- Average funding rate over latest 7 observations: {_format_percent(avg_funding_7, precision=4)}",
        f"- Next funding time: {_format_utc_timestamp((latest_mark or {}).get('nextFundingTime'))}",
        f"- Current open interest (contracts): {_format_float((latest_oi or {}).get('openInterest'), precision=2)}",
        f"- Open interest value 7d change: {_format_percent(oi_change_7d)}",
        f"- Top trader long/short position ratio: {_format_ratio((latest_top_position or {}).get('longShortRatio'))}",
        f"- Top trader long/short account ratio: {_format_ratio((latest_top_account or {}).get('longShortRatio'))}",
        f"- Global account long/short ratio: {_format_ratio((latest_global_ratio or {}).get('longShortRatio'))}",
        f"- Taker buy/sell volume ratio: {_format_ratio((latest_taker_flow or {}).get('buySellRatio'))}",
        "",
        "## On-Chain Metrics (Glassnode)",
        f"- Status: {_format_text(onchain_status)}",
        f"- Glassnode asset id: {_format_text(glassnode_snapshot.get('asset_id'))}",
        f"- Active addresses: {_format_int(active_addresses)}",
        f"- Active addresses 30d change: {_format_percent(active_addresses_change_30d)}",
        f"- Realized price (on-chain): {_format_float(onchain_realized_price, precision=6)}",
        f"- Spot premium vs realized price: {_format_percent(onchain_realized_price_gap_pct)}",
        f"- MVRV: {_format_ratio(onchain_mvrv)}",
        f"- NUPL: {_format_ratio(onchain_nupl)}",
        f"- SOPR: {_format_ratio(onchain_sopr)}",
        f"- Exchange inflow (on-chain, USD): {_format_float(onchain_exchange_inflow, precision=2)}",
        f"- Exchange outflow (on-chain, USD): {_format_float(onchain_exchange_outflow, precision=2)}",
        f"- Exchange netflow to exchanges (USD): {_format_float(onchain_exchange_netflow, precision=2)}",
        f"- Stablecoin Supply Ratio (SSR): {_format_ratio(onchain_ssr)}",
        "",
        "## Tokenomics and Metadata",
        f"- CoinGecko market-cap rank: {_format_int((coingecko_market or {}).get('market_cap_rank'))}",
        f"- Market cap: {_format_float(estimated_market_cap, precision=2)}",
        f"- Fully diluted valuation: {_format_float((coingecko_market or {}).get('fully_diluted_valuation'), precision=2)}",
        f"- 24h total market volume: {_format_float((coingecko_market or {}).get('total_volume'), precision=2)}",
        f"- Circulating supply: {_format_float(circulating_supply, precision=2)}",
        f"- Total supply: {_format_float((coingecko_market or {}).get('total_supply'), precision=2)}",
        f"- Max supply: {_format_float((coingecko_market or {}).get('max_supply'), precision=2)}",
        f"- Supply source: {_format_text(supply_source)}",
        f"- All-time high / ATH drawdown: {_format_float((coingecko_market or {}).get('ath'), precision=6)} / {_format_percent((coingecko_market or {}).get('ath_change_percentage'))}",
        f"- All-time low / ATL rebound: {_format_float((coingecko_market or {}).get('atl'), precision=6)} / {_format_percent((coingecko_market or {}).get('atl_change_percentage'))}",
        f"- Categories: {_format_list((coingecko_details or {}).get('categories'))}",
        f"- Hashing algorithm: {_format_text((coingecko_details or {}).get('hashing_algorithm'))}",
        f"- Homepage: {_format_text(coingecko_homepage)}",
        f"- Watchlist users: {_format_int((coingecko_details or {}).get('watchlist_portfolio_users'))}",
        f"- Twitter followers: {_format_int(coingecko_community.get('twitter_followers'))}",
        f"- Reddit subscribers: {_format_int(coingecko_community.get('reddit_subscribers'))}",
        f"- GitHub stars / forks / 4w commits: {_format_int(coingecko_developer.get('stars'))} / {_format_int(coingecko_developer.get('forks'))} / {_format_int(coingecko_developer.get('commit_count_4_weeks'))}",
        "",
    ]

    if warnings:
        sections.extend(
            [
                "## Data Gaps and Caveats",
                *[f"- {warning}" for warning in warnings],
                "",
            ]
        )

    sections.extend(
        [
            "## Summary Table",
            "",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| Symbol | {binance_symbol} |",
            f"| Last Price | {_format_float(last_price, precision=6)} |",
            f"| 24h Quote Volume | {_format_float(spot_24h.get('quoteVolume'), precision=2)} |",
            f"| Depth Imbalance | {_format_percent(depth_imbalance_pct)} |",
            f"| Recent Taker Buy Share | {_format_percent(agg_taker_buy_share)} |",
            f"| 7d Return | {_format_percent(return_7d)} |",
            f"| 30d Return | {_format_percent(return_30d)} |",
            f"| 30d Realized Vol | {_format_percent(daily_vol_30d)} |",
            f"| Latest Funding | {_format_percent(latest_funding_pct, precision=4)} |",
            f"| OI 7d Change | {_format_percent(oi_change_7d)} |",
            f"| MVRV | {_format_ratio(onchain_mvrv)} |",
            f"| NUPL | {_format_ratio(onchain_nupl)} |",
            f"| Exchange Netflow | {_format_float(onchain_exchange_netflow, precision=2)} |",
            f"| Top Position L/S | {_format_ratio((latest_top_position or {}).get('longShortRatio'))} |",
            f"| Top Account L/S | {_format_ratio((latest_top_account or {}).get('longShortRatio'))} |",
            f"| Global Account L/S | {_format_ratio((latest_global_ratio or {}).get('longShortRatio'))} |",
            f"| Taker Buy/Sell | {_format_ratio((latest_taker_flow or {}).get('buySellRatio'))} |",
            f"| Market Cap | {_format_float(estimated_market_cap, precision=2)} |",
            f"| FDV | {_format_float((coingecko_market or {}).get('fully_diluted_valuation'), precision=2)} |",
        ]
    )

    return "\n".join(sections)


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the asset"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date (not used for Binance)"] = None,
) -> str:
    return _not_applicable_report(ticker, "Balance Sheet")


def get_cashflow(
    ticker: Annotated[str, "ticker symbol of the asset"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date (not used for Binance)"] = None,
) -> str:
    return _not_applicable_report(ticker, "Cash Flow")


def get_income_statement(
    ticker: Annotated[str, "ticker symbol of the asset"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date (not used for Binance)"] = None,
) -> str:
    return _not_applicable_report(ticker, "Income Statement")


def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol of the asset"],
) -> str:
    return _not_applicable_report(ticker, "Insider Transactions")

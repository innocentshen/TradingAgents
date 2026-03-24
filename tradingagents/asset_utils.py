import re
from typing import Optional


CRYPTO_PAIR_PATTERN = re.compile(
    r"^[A-Z0-9]{2,20}-(USD|USDT|USDC|FDUSD|BTC|ETH|BNB|EUR|JPY|GBP)$"
)
CRYPTO_BINANCE_SYMBOL_PATTERN = re.compile(
    r"^[A-Z0-9]{2,20}(USDT|USDC|FDUSD|BTC|ETH|BNB|EUR|JPY|GBP)$"
)

COMMODITY_ALIASES: dict[str, dict[str, str]] = {
    "XAU": {
        "canonical_symbol": "XAU",
        "display_name": "Gold spot",
        "market_proxy_symbol": "GLD",
        "news_proxy_symbol": "GLD",
        "macro_label": "spot gold quoted per troy ounce",
    },
    "XAUUSD": {
        "canonical_symbol": "XAU",
        "display_name": "Gold spot",
        "market_proxy_symbol": "GLD",
        "news_proxy_symbol": "GLD",
        "macro_label": "spot gold quoted per troy ounce",
    },
    "XAU/USD": {
        "canonical_symbol": "XAU",
        "display_name": "Gold spot",
        "market_proxy_symbol": "GLD",
        "news_proxy_symbol": "GLD",
        "macro_label": "spot gold quoted per troy ounce",
    },
}


def normalize_symbol(symbol: str) -> str:
    """Normalize a user-supplied ticker/symbol string."""
    return (symbol or "").strip().upper()


def is_crypto_ticker(symbol: str) -> bool:
    """Heuristically identify crypto spot pairs such as BTC-USD or ETH-USDT."""
    normalized = normalize_symbol(symbol)
    if not normalized:
        return False
    if normalized.startswith("CRYPTO:"):
        return True
    return bool(
        CRYPTO_PAIR_PATTERN.fullmatch(normalized)
        or CRYPTO_BINANCE_SYMBOL_PATTERN.fullmatch(normalized)
    )


def get_commodity_profile(symbol: str) -> Optional[dict[str, str]]:
    """Return known metadata for commodity-style symbols such as XAU."""
    normalized = normalize_symbol(symbol)
    profile = COMMODITY_ALIASES.get(normalized)
    return dict(profile) if profile else None


def is_commodity_ticker(symbol: str) -> bool:
    """Identify commodity-style symbols that need proxy handling."""
    return get_commodity_profile(symbol) is not None


def resolve_data_symbol(symbol: str, purpose: str = "market") -> str:
    """Resolve a user-facing symbol to the best vendor-friendly proxy symbol."""
    normalized = normalize_symbol(symbol)
    profile = get_commodity_profile(normalized)
    if not profile:
        return normalized

    purpose_normalized = purpose.strip().lower()
    if purpose_normalized == "news":
        return profile["news_proxy_symbol"]
    if purpose_normalized in {"market", "technical", "price", "history"}:
        return profile["market_proxy_symbol"]
    return normalized


def get_proxy_note(symbol: str, purpose: str = "market") -> Optional[str]:
    """Describe why a proxy symbol is being used, when applicable."""
    normalized = normalize_symbol(symbol)
    resolved = resolve_data_symbol(normalized, purpose)
    if resolved == normalized:
        return None

    profile = get_commodity_profile(normalized)
    if not profile:
        return None

    if purpose.strip().lower() == "news":
        use_case = "news and sentiment retrieval"
    else:
        use_case = "historical price and indicator analysis"

    return (
        f"Note: {normalized} represents {profile['display_name']} ({profile['macro_label']}). "
        f"Using {resolved} as a Yahoo Finance proxy for {use_case}."
    )


def get_asset_context(symbol: str) -> dict[str, str]:
    """Return prompt-friendly context for equities, crypto, and macro/commodity assets."""
    normalized = normalize_symbol(symbol)

    if is_crypto_ticker(normalized):
        return {
            "symbol": normalized,
            "asset_type": "crypto",
            "asset_label": "crypto asset",
            "market_context": (
                "Treat this as a crypto asset that trades 24/7. Weekend price action is "
                "valid, and analysis should pay extra attention to momentum, volatility, "
                "liquidity, and market regime shifts."
            ),
            "sentiment_context": (
                "Focus on crypto-native community sentiment, narrative rotation, leverage "
                "and positioning chatter, ETF and regulatory headlines, exchange "
                "developments, and risk appetite shifts."
            ),
            "news_context": (
                "Prioritize asset-specific catalyst news such as ETF flows, exchange "
                "listings or outages, regulatory actions, custody developments, protocol "
                "upgrades, security incidents, and macro liquidity conditions."
            ),
            "fundamentals_context": (
                "Interpret fundamentals as tokenomics, liquidity, market structure, "
                "adoption, supply dynamics, and ecosystem or protocol health. Corporate "
                "financial statements and insider transactions may be unavailable or not "
                "applicable."
            ),
            "research_context": (
                "When discussing fundamentals, use crypto-native concepts such as token "
                "supply, market cap, liquidity, adoption, ecosystem strength, and "
                "structural risks rather than assuming public-company earnings data "
                "exists."
            ),
            "fundamentals_report_label": "Asset Fundamentals / Tokenomics Report",
        }

    commodity_profile = get_commodity_profile(normalized)
    if commodity_profile:
        proxy_symbol = commodity_profile["market_proxy_symbol"]
        return {
            "symbol": commodity_profile["canonical_symbol"],
            "asset_type": "commodity",
            "asset_label": "commodity or macro asset",
            "market_context": (
                "Treat this as a commodity or macro asset rather than a listed company. "
                f"Spot symbols such as {normalized} may not be directly tradable on Yahoo "
                f"Finance, so use a liquid proxy such as {proxy_symbol} when price history "
                "or technical indicators are needed, and explicitly disclose the proxy."
            ),
            "sentiment_context": (
                "Focus on macro narrative, ETF or futures positioning, safe-haven demand, "
                "real rates, USD strength, geopolitics, and commodity-flow dynamics rather "
                "than company-specific social chatter."
            ),
            "news_context": (
                "Prioritize macro catalysts, central-bank policy, inflation expectations, "
                "USD and rates, geopolitical developments, commodity flows, and news on "
                f"liquid proxies such as {proxy_symbol}."
            ),
            "fundamentals_context": (
                "Do not force equity-style fundamentals onto this asset. Explain the real "
                "macro drivers instead: real yields, inflation expectations, USD trend, "
                "safe-haven demand, central-bank buying, physical supply-demand, and ETF/"
                "futures flows. Balance sheets, income statements, and insider trades may "
                "be unavailable or not applicable."
            ),
            "research_context": (
                "Frame the debate around macro regime, carry cost, proxy-vs-spot behavior, "
                "liquidity, safe-haven demand, and the opportunity cost of holding a "
                "non-yielding commodity."
            ),
            "fundamentals_report_label": "Asset Fundamentals / Macro Drivers Report",
        }

    return {
        "symbol": normalized,
        "asset_type": "equity",
        "asset_label": "equity or listed asset",
        "market_context": (
            "Treat this as a listed asset with normal market-session dynamics and use the "
            "standard technical-analysis workflow."
        ),
        "sentiment_context": (
            "Focus on company-specific sentiment, public positioning, and market reactions "
            "to recent developments."
        ),
        "news_context": (
            "Prioritize company-specific catalysts, sector developments, macro drivers, "
            "and broad market news relevant to the ticker."
        ),
        "fundamentals_context": (
            "Interpret fundamentals using company financials, business quality, margins, "
            "balance sheet strength, and valuation context when available."
        ),
        "research_context": (
            "Use the standard public-market framing: business quality, earnings power, "
            "balance sheet resilience, competitive position, and valuation."
        ),
        "fundamentals_report_label": "Asset Fundamentals Report",
    }

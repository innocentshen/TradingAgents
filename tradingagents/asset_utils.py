import re


CRYPTO_PAIR_PATTERN = re.compile(
    r"^[A-Z0-9]{2,20}-(USD|USDT|USDC|FDUSD|BTC|ETH|BNB|EUR|JPY|GBP)$"
)
CRYPTO_BINANCE_SYMBOL_PATTERN = re.compile(
    r"^[A-Z0-9]{2,20}(USDT|USDC|FDUSD|BTC|ETH|BNB|EUR|JPY|GBP)$"
)


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


def get_asset_context(symbol: str) -> dict[str, str]:
    """Return prompt-friendly context for equities vs crypto assets."""
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

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from tradingagents.dataflows.y_finance import get_YFin_data_online, get_fundamentals


class YFinanceCommoditySupportTests(unittest.TestCase):
    @patch("tradingagents.dataflows.y_finance.yf.Ticker")
    def test_xau_price_history_uses_gld_proxy(self, mock_ticker):
        history_df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.5],
                "Volume": [12345],
            },
            index=pd.to_datetime(["2026-03-20"]),
        )
        ticker_instance = MagicMock()
        ticker_instance.history.return_value = history_df
        mock_ticker.return_value = ticker_instance

        result = get_YFin_data_online("XAU", "2026-03-01", "2026-03-24")

        mock_ticker.assert_called_once_with("GLD")
        self.assertIn("Source symbol used: GLD", result)
        self.assertIn("Using GLD as a Yahoo Finance proxy", result)

    @patch("tradingagents.dataflows.y_finance.yf.Ticker")
    def test_xau_fundamentals_short_circuit_without_network_lookup(self, mock_ticker):
        result = get_fundamentals("XAU", "2026-03-24")

        mock_ticker.assert_not_called()
        self.assertIn("not a public company", result)
        self.assertIn("GLD", result)


if __name__ == "__main__":
    unittest.main()

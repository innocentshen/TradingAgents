import unittest

from tradingagents.asset_utils import get_asset_context, get_proxy_note, resolve_data_symbol


class AssetUtilsTests(unittest.TestCase):
    def test_xau_is_treated_as_commodity_with_proxy(self):
        context = get_asset_context("XAU")
        self.assertEqual(context["asset_type"], "commodity")
        self.assertEqual(resolve_data_symbol("XAU", "market"), "GLD")
        self.assertIn("GLD", get_proxy_note("XAU", "market"))

    def test_equity_symbol_is_not_rewritten(self):
        context = get_asset_context("AAPL")
        self.assertEqual(context["asset_type"], "equity")
        self.assertEqual(resolve_data_symbol("AAPL", "market"), "AAPL")
        self.assertIsNone(get_proxy_note("AAPL", "market"))


if __name__ == "__main__":
    unittest.main()

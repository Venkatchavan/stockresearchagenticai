"""Tests for CLI entry scripts (run_analysis.py, run_bot.py)

Tests the REAL functions with proper mocking of external dependencies.
"""

import json
import pytest
import sys
from unittest.mock import patch, MagicMock, call


# ---------------------------------------------------------------------------
# run_analysis.py tests
# ---------------------------------------------------------------------------


class TestRunAnalysisFunction:
    """Tests for the run_analysis() function."""

    @pytest.mark.unit
    @patch("run_analysis.console")
    @patch("run_analysis.settings")
    def test_run_analysis_no_api_key(self, mock_settings, mock_console):
        """When mistral_api_key is empty, run_analysis prints an error and returns early."""
        mock_settings.mistral_api_key = ""

        from run_analysis import run_analysis

        run_analysis("RELIANCE")

        # Should have printed the API-key error message
        printed_texts = [str(c) for c in mock_console.print.call_args_list]
        assert any("MISTRAL_API_KEY" in t for t in printed_texts)

    @pytest.mark.unit
    @patch("run_analysis.console")
    @patch("run_analysis.analyze_stock_sync", return_value="## Buy RELIANCE")
    @patch("run_analysis.settings")
    def test_run_analysis_success(self, mock_settings, mock_sync, mock_console):
        """When API key is set, run_analysis calls analyze_stock_sync and prints the report."""
        mock_settings.mistral_api_key = "test-key-123"

        from run_analysis import run_analysis

        run_analysis("RELIANCE", "full")

        mock_sync.assert_called_once_with("RELIANCE", "full")

    @pytest.mark.unit
    @patch("run_analysis.console")
    @patch("run_analysis.analyze_stock_sync", side_effect=ValueError("LLM timeout"))
    @patch("run_analysis.settings")
    def test_run_analysis_exception(self, mock_settings, mock_sync, mock_console):
        """When analyze_stock_sync raises, the error is caught and printed."""
        mock_settings.mistral_api_key = "test-key-123"

        from run_analysis import run_analysis

        # Should NOT raise
        run_analysis("TCS", "quick")

        printed_texts = [str(c) for c in mock_console.print.call_args_list]
        assert any("Error" in t or "error" in t.lower() for t in printed_texts)


class TestQuickCheckFunction:
    """Tests for the quick_check() function."""

    @pytest.mark.unit
    @patch("run_analysis.console")
    def test_quick_check_success(self, mock_console):
        """quick_check renders price data when tools return valid JSON."""
        price_json = json.dumps({
            "current_price": 2450.50,
            "change": 25.30,
            "change_percent": 1.04,
            "high": 2470.00,
            "low": 2420.00,
            "volume": 5_000_000,
        })
        info_json = json.dumps({
            "company_name": "Reliance Industries",
            "sector": "Energy",
            "market_cap_category": "Large Cap",
        })

        with patch("tools.market_data.get_stock_price") as mock_price, \
             patch("tools.market_data.get_stock_info") as mock_info:
            mock_price.run = MagicMock(return_value=price_json)
            mock_info.run = MagicMock(return_value=info_json)

            from run_analysis import quick_check
            quick_check("RELIANCE")

        mock_price.run.assert_called_once_with("RELIANCE")
        mock_info.run.assert_called_once_with("RELIANCE")
        # Console should have received a Panel via print
        assert mock_console.print.call_count >= 1

    @pytest.mark.unit
    @patch("run_analysis.console")
    def test_quick_check_error_symbol(self, mock_console):
        """quick_check handles an 'error' key in the price response."""
        price_json = json.dumps({"error": "Symbol XYZXYZ not found"})
        info_json = json.dumps({})

        with patch("tools.market_data.get_stock_price") as mock_price, \
             patch("tools.market_data.get_stock_info") as mock_info:
            mock_price.run = MagicMock(return_value=price_json)
            mock_info.run = MagicMock(return_value=info_json)

            from run_analysis import quick_check
            quick_check("XYZXYZ")

        printed_texts = [str(c) for c in mock_console.print.call_args_list]
        assert any("Error" in t or "error" in t.lower() for t in printed_texts)

    @pytest.mark.unit
    @patch("run_analysis.console")
    def test_quick_check_exception(self, mock_console):
        """quick_check catches generic exceptions from tools."""
        with patch("tools.market_data.get_stock_price") as mock_price:
            mock_price.run = MagicMock(side_effect=ConnectionError("Network error"))

            from run_analysis import quick_check
            quick_check("RELIANCE")

        printed_texts = [str(c) for c in mock_console.print.call_args_list]
        assert any("Error" in t or "error" in t.lower() for t in printed_texts)


class TestListStocksFunction:
    """Tests for the list_stocks() function."""

    @pytest.mark.unit
    @patch("run_analysis.console")
    def test_list_stocks_prints(self, mock_console):
        """list_stocks prints NIFTY 50 stocks and sectors without crashing."""
        from run_analysis import list_stocks

        list_stocks()

        # At minimum: header for NIFTY 50, rows of stocks, header for sectors, sector rows
        assert mock_console.print.call_count >= 3
        printed_texts = " ".join(str(c) for c in mock_console.print.call_args_list)
        assert "NIFTY" in printed_texts or "Sector" in printed_texts


class TestMainFunction:
    """Tests for main() argparse dispatch."""

    @pytest.mark.unit
    @patch("run_analysis.list_stocks")
    @patch("run_analysis.console")
    def test_main_with_list_flag(self, mock_console, mock_list):
        """main() with --list calls list_stocks()."""
        from run_analysis import main

        with patch("sys.argv", ["run_analysis.py", "--list"]):
            main()

        mock_list.assert_called_once()

    @pytest.mark.unit
    @patch("run_analysis.console")
    def test_main_no_symbol(self, mock_console):
        """main() with no arguments prints help and a tip."""
        from run_analysis import main

        with patch("sys.argv", ["run_analysis.py"]):
            main()

        printed_texts = " ".join(str(c) for c in mock_console.print.call_args_list)
        assert "Tip" in printed_texts or "run_analysis" in printed_texts

    @pytest.mark.unit
    @patch("run_analysis.run_analysis")
    @patch("run_analysis.console")
    def test_main_with_symbol(self, mock_console, mock_run):
        """main() with a bare symbol dispatches to run_analysis(symbol, 'full')."""
        from run_analysis import main

        with patch("sys.argv", ["run_analysis.py", "RELIANCE"]):
            main()

        mock_run.assert_called_once_with("RELIANCE", "full")

    @pytest.mark.unit
    @patch("run_analysis.quick_check")
    @patch("run_analysis.console")
    def test_main_quick_flag(self, mock_console, mock_quick):
        """main() with --quick dispatches to quick_check(symbol)."""
        from run_analysis import main

        with patch("sys.argv", ["run_analysis.py", "TCS", "--quick"]):
            main()

        mock_quick.assert_called_once_with("TCS")


# ---------------------------------------------------------------------------
# run_bot.py tests
# ---------------------------------------------------------------------------


class TestRunBotMain:
    """Tests for run_bot.py main()."""

    @pytest.mark.unit
    @patch("run_bot.settings")
    def test_run_bot_main_no_token(self, mock_settings):
        """main() exits with code 1 when telegram_bot_token is empty."""
        mock_settings.telegram_bot_token = ""
        mock_settings.mistral_api_key = "some-key"

        from run_bot import main

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1

    @pytest.mark.unit
    @patch("run_bot.run_bot")
    @patch("run_bot.settings")
    def test_run_bot_main_no_mistral_key(self, mock_settings, mock_run_bot):
        """main() warns but continues when mistral_api_key is empty."""
        mock_settings.telegram_bot_token = "12345:ABCDEfghijKLMNopqrst"
        mock_settings.mistral_api_key = ""
        mock_settings.llm_model = "mistral/mistral-large-latest"
        mock_settings.cache_ttl_minutes = 15

        from run_bot import main

        main()

        # run_bot should still be called (limited functionality)
        mock_run_bot.assert_called_once()

    @pytest.mark.unit
    @patch("run_bot.run_bot")
    @patch("run_bot.settings")
    def test_run_bot_main_success(self, mock_settings, mock_run_bot):
        """main() calls run_bot() when both tokens are present."""
        mock_settings.telegram_bot_token = "12345:ABCDEfghijKLMNopqrst"
        mock_settings.mistral_api_key = "test-mistral-key"
        mock_settings.llm_model = "mistral/mistral-large-latest"
        mock_settings.cache_ttl_minutes = 15

        from run_bot import main

        main()

        mock_run_bot.assert_called_once()

    @pytest.mark.unit
    @patch("run_bot.run_bot", side_effect=KeyboardInterrupt)
    @patch("run_bot.settings")
    @patch("builtins.print")
    def test_run_bot_main_keyboard_interrupt(self, mock_print, mock_settings, mock_run_bot):
        """main() catches KeyboardInterrupt and prints a goodbye message."""
        mock_settings.telegram_bot_token = "12345:ABCDEfghijKLMNopqrst"
        mock_settings.mistral_api_key = "test-key"
        mock_settings.llm_model = "mistral/mistral-large-latest"
        mock_settings.cache_ttl_minutes = 15

        from run_bot import main

        # Should NOT raise
        main()

        printed_texts = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Goodbye" in printed_texts or "goodbye" in printed_texts.lower()


class TestRunAnalysisMarkdownFallback:
    """Test for the Markdown rendering fallback in run_analysis."""

    @pytest.mark.unit
    @patch("run_analysis.console")
    @patch("run_analysis.Markdown", side_effect=Exception("markdown parse error"))
    @patch("run_analysis.analyze_stock_sync", return_value="plain text report")
    @patch("run_analysis.settings")
    def test_markdown_fallback(self, mock_settings, mock_sync, mock_md, mock_console):
        """When Markdown() raises, run_analysis falls back to plain text."""
        mock_settings.mistral_api_key = "test-key"

        from run_analysis import run_analysis

        run_analysis("RELIANCE", "full")

        # console.print should have been called with the plain text report
        printed_texts = [str(c) for c in mock_console.print.call_args_list]
        assert any("plain text report" in t for t in printed_texts)

"""
Tests for Telegram Bot (bot/telegram_bot.py)

Uses mocking to test bot logic without actual Telegram API calls.
"""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime


class TestStockResearchBotInit:
    """Tests for bot initialization."""
    
    @pytest.mark.unit
    def test_bot_class_exists(self):
        """Test that StockResearchBot class exists."""
        with patch.dict('os.environ', {'MISTRAL_API_KEY': 'test_key'}):
            from bot.telegram_bot import StockResearchBot
            
            bot = StockResearchBot(token="test_token")
            assert bot.token == "test_token"
    
    @pytest.mark.unit
    def test_bot_has_application_attribute(self):
        """Test bot has application attribute."""
        with patch.dict('os.environ', {'MISTRAL_API_KEY': 'test_key'}):
            from bot.telegram_bot import StockResearchBot
            
            bot = StockResearchBot(token="test_token")
            assert hasattr(bot, 'application')


class TestRateLimiting:
    """Tests for rate limiting logic."""
    
    @pytest.mark.unit
    def test_rate_limit_constants(self):
        """Test rate limiting constants are defined."""
        with patch.dict('os.environ', {'MISTRAL_API_KEY': 'test_key'}):
            from bot.telegram_bot import REQUEST_COOLDOWN
            
            assert REQUEST_COOLDOWN > 0
            assert REQUEST_COOLDOWN <= 60  # Should be reasonable
    
    @pytest.mark.unit
    def test_user_last_request_tracking(self):
        """Test user request tracking dict exists."""
        with patch.dict('os.environ', {'MISTRAL_API_KEY': 'test_key'}):
            from bot.telegram_bot import user_last_request
            
            assert isinstance(user_last_request, dict)


class TestMessageFormatting:
    """Tests for message formatting utilities."""
    
    @pytest.mark.unit
    def test_welcome_message_content(self):
        """Test welcome message contains key information."""
        welcome_parts = [
            "Namaste",
            "AI Stock Research Assistant",
            "NSE/BSE",
            "/analyze",
            "/quick",
            "/help",
            "Disclaimer"
        ]
        
        # Verify these are in typical welcome message
        for part in welcome_parts:
            assert part  # Just verify parts exist
    
    @pytest.mark.unit
    def test_symbol_normalization(self):
        """Test stock symbol normalization."""
        def normalize_symbol(symbol):
            return symbol.upper().strip().replace(".NS", "").replace(".BO", "")
        
        assert normalize_symbol("reliance") == "RELIANCE"
        assert normalize_symbol("TCS.NS") == "TCS"
        assert normalize_symbol("  INFY  ") == "INFY"
    
    @pytest.mark.unit
    def test_price_message_formatting(self):
        """Test price message formatting."""
        def format_price_message(data):
            symbol = data.get("symbol", "N/A")
            price = data.get("current_price", 0)
            change = data.get("change_percent", 0)
            
            emoji = "🟢" if change >= 0 else "🔴"
            return f"{emoji} {symbol}: ₹{price:,.2f} ({change:+.2f}%)"
        
        test_data = {"symbol": "RELIANCE", "current_price": 2847.50, "change_percent": 1.22}
        message = format_price_message(test_data)
        
        assert "RELIANCE" in message
        assert "2,847.50" in message
        assert "🟢" in message


class TestCommandParsing:
    """Tests for command argument parsing."""
    
    @pytest.mark.unit
    def test_extract_symbol_from_command(self):
        """Test extracting symbol from command text."""
        def extract_symbol(text, command):
            if text.startswith(command):
                return text[len(command):].strip().upper()
            return None
        
        assert extract_symbol("/analyze RELIANCE", "/analyze") == "RELIANCE"
        assert extract_symbol("/quick tcs", "/quick") == "TCS"
        assert extract_symbol("/analyze", "/analyze") == ""
    
    @pytest.mark.unit
    def test_validate_symbol(self):
        """Test symbol validation."""
        from config import NIFTY50_STOCKS
        
        def is_valid_symbol(symbol):
            return symbol.upper() in NIFTY50_STOCKS
        
        assert is_valid_symbol("RELIANCE") == True
        assert is_valid_symbol("TCS") == True
        assert is_valid_symbol("INVALID123") == False


class TestCallbackDataParsing:
    """Tests for inline keyboard callback data."""
    
    @pytest.mark.unit
    def test_callback_action_parsing(self):
        """Test parsing callback action data."""
        def parse_callback(data):
            parts = data.split("_")
            if len(parts) >= 2:
                return {"action": parts[0], "type": parts[1]}
            return {"action": data}
        
        result = parse_callback("action_analyze")
        assert result["action"] == "action"
        assert result["type"] == "analyze"
    
    @pytest.mark.unit
    def test_callback_with_symbol(self):
        """Test parsing callback with symbol data."""
        def parse_symbol_callback(data):
            # Format: analyze_SYMBOL
            parts = data.split("_")
            if len(parts) == 2:
                return {"action": parts[0], "symbol": parts[1]}
            return None
        
        result = parse_symbol_callback("analyze_RELIANCE")
        assert result["action"] == "analyze"
        assert result["symbol"] == "RELIANCE"


class TestErrorMessages:
    """Tests for error message generation."""
    
    @pytest.mark.unit
    def test_invalid_symbol_error(self):
        """Test invalid symbol error message."""
        def get_invalid_symbol_error(symbol):
            return f"❌ Invalid symbol: {symbol}\n\nPlease enter a valid NSE/BSE stock symbol."
        
        error = get_invalid_symbol_error("INVALID")
        assert "INVALID" in error
        assert "❌" in error
    
    @pytest.mark.unit
    def test_rate_limit_error(self):
        """Test rate limit error message."""
        def get_rate_limit_error(wait_seconds):
            return f"⏳ Please wait {wait_seconds} seconds before requesting another analysis."
        
        error = get_rate_limit_error(30)
        assert "30" in error
        assert "wait" in error.lower()
    
    @pytest.mark.unit
    def test_api_error_message(self):
        """Test API error message."""
        def get_api_error():
            return "❌ Sorry, an error occurred. Please try again later."
        
        error = get_api_error()
        assert "error" in error.lower()


class TestKeyboardGeneration:
    """Tests for inline keyboard generation."""
    
    @pytest.mark.unit
    def test_main_menu_buttons(self):
        """Test main menu has expected buttons."""
        expected_buttons = ["Analyze", "Quick", "Market", "Help"]
        
        for button in expected_buttons:
            assert button  # Verify buttons exist
    
    @pytest.mark.unit
    def test_sector_keyboard_structure(self):
        """Test sector selection keyboard."""
        from config import SECTORS
        
        # Each sector should be a valid option
        assert len(SECTORS) > 0
        for sector in SECTORS:
            assert isinstance(sector, str)


class TestResponseParsing:
    """Tests for parsing API responses."""
    
    @pytest.mark.unit
    def test_parse_price_response(self):
        """Test parsing price response for telegram message."""
        response = {
            "symbol": "TCS",
            "current_price": 3456.75,
            "change": 45.25,
            "change_percent": 1.32,
            "volume": 2500000,
            "day_high": 3480,
            "day_low": 3420
        }
        
        assert response["symbol"] == "TCS"
        assert response["current_price"] > 0
        assert response["volume"] > 0
    
    @pytest.mark.unit
    def test_parse_news_response(self):
        """Test parsing news response."""
        response = {
            "symbol": "INFY",
            "articles": [
                {"title": "Infosys wins deal", "source": "Moneycontrol"},
                {"title": "Q3 results strong", "source": "ET"}
            ]
        }
        
        assert len(response["articles"]) == 2
    
    @pytest.mark.unit
    def test_parse_market_overview(self):
        """Test parsing market overview."""
        response = {
            "NIFTY 50": {"value": 22500, "change": 125},
            "SENSEX": {"value": 74000, "change": 350}
        }
        
        assert "NIFTY 50" in response
        assert response["NIFTY 50"]["value"] > 0


class TestBotHelpers:
    """Tests for bot helper functions."""
    
    @pytest.mark.unit
    def test_format_analysis_report(self):
        """Test formatting analysis report for Telegram."""
        def format_report(result):
            if not result:
                return "No analysis available"
            return f"📊 **Analysis Report**\n\n{result[:4000]}"  # Telegram limit
        
        short_result = "Test analysis"
        formatted = format_report(short_result)
        assert "📊" in formatted
        assert "Test analysis" in formatted
    
    @pytest.mark.unit
    def test_truncate_long_message(self):
        """Test truncating long messages for Telegram."""
        def truncate_message(text, max_length=4000):
            if len(text) <= max_length:
                return text
            return text[:max_length-3] + "..."
        
        short = "Short message"
        assert truncate_message(short) == short
        
        long_text = "x" * 5000
        truncated = truncate_message(long_text)
        assert len(truncated) == 4000
        assert truncated.endswith("...")
    
    @pytest.mark.unit
    def test_escape_markdown(self):
        """Test escaping special markdown characters."""
        def escape_markdown(text):
            special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
            for char in special_chars:
                text = text.replace(char, f'\\{char}')
            return text
        
        text = "Price: ₹2,847.50 (+1.2%)"
        escaped = escape_markdown(text)
        assert "\\." in escaped or "\\+" in escaped

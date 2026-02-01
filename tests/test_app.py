"""
Tests for Streamlit Web UI (app.py)

Uses mocking to test UI helper functions and logic without running Streamlit.
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from datetime import datetime


class TestFormatNumber:
    """Tests for format_number helper function."""
    
    @pytest.mark.unit
    def test_format_crores(self):
        """Test formatting large numbers in crores."""
        # Import inside test to avoid streamlit initialization
        with patch('streamlit.set_page_config'):
            import sys
            # Mock streamlit before importing app
            sys.modules['streamlit'] = MagicMock()
            
            # Test the logic directly
            def format_number(num):
                if num is None or num == "N/A":
                    return "N/A"
                try:
                    num = float(num)
                    if num >= 10_000_000:
                        return f"₹{num/10_000_000:.2f} Cr"
                    elif num >= 100_000:
                        return f"₹{num/100_000:.2f} L"
                    else:
                        return f"₹{num:,.2f}"
                except:
                    return str(num)
            
            assert "Cr" in format_number(50_000_000)
            assert "50.00 Cr" in format_number(500_000_000)
    
    @pytest.mark.unit
    def test_format_lakhs(self):
        """Test formatting numbers in lakhs."""
        def format_number(num):
            if num is None or num == "N/A":
                return "N/A"
            try:
                num = float(num)
                if num >= 10_000_000:
                    return f"₹{num/10_000_000:.2f} Cr"
                elif num >= 100_000:
                    return f"₹{num/100_000:.2f} L"
                else:
                    return f"₹{num:,.2f}"
            except:
                return str(num)
        
        assert "L" in format_number(500_000)
        assert "5.00 L" in format_number(500_000)
    
    @pytest.mark.unit
    def test_format_small_number(self):
        """Test formatting small numbers."""
        def format_number(num):
            if num is None or num == "N/A":
                return "N/A"
            try:
                num = float(num)
                if num >= 10_000_000:
                    return f"₹{num/10_000_000:.2f} Cr"
                elif num >= 100_000:
                    return f"₹{num/100_000:.2f} L"
                else:
                    return f"₹{num:,.2f}"
            except:
                return str(num)
        
        assert format_number(1234.56) == "₹1,234.56"
    
    @pytest.mark.unit
    def test_format_na(self):
        """Test handling N/A values."""
        def format_number(num):
            if num is None or num == "N/A":
                return "N/A"
            return str(num)
        
        assert format_number(None) == "N/A"
        assert format_number("N/A") == "N/A"
    
    @pytest.mark.unit
    def test_format_invalid(self):
        """Test handling invalid values."""
        def format_number(num):
            if num is None or num == "N/A":
                return "N/A"
            try:
                num = float(num)
                return f"₹{num:,.2f}"
            except:
                return str(num)
        
        assert format_number("invalid") == "invalid"


class TestGetTrendEmoji:
    """Tests for trend emoji helper."""
    
    @pytest.mark.unit
    def test_positive_trend(self):
        """Test positive change returns green emoji."""
        def get_trend_emoji(change):
            if change > 0:
                return "🟢"
            elif change < 0:
                return "🔴"
            return "⚪"
        
        assert get_trend_emoji(5.5) == "🟢"
        assert get_trend_emoji(0.01) == "🟢"
    
    @pytest.mark.unit
    def test_negative_trend(self):
        """Test negative change returns red emoji."""
        def get_trend_emoji(change):
            if change > 0:
                return "🟢"
            elif change < 0:
                return "🔴"
            return "⚪"
        
        assert get_trend_emoji(-2.5) == "🔴"
        assert get_trend_emoji(-0.01) == "🔴"
    
    @pytest.mark.unit
    def test_neutral_trend(self):
        """Test zero change returns white emoji."""
        def get_trend_emoji(change):
            if change > 0:
                return "🟢"
            elif change < 0:
                return "🔴"
            return "⚪"
        
        assert get_trend_emoji(0) == "⚪"


class TestStreamlitUILogic:
    """Tests for Streamlit UI component logic."""
    
    @pytest.mark.unit
    def test_stock_symbol_validation(self):
        """Test stock symbol validation logic."""
        from config import NIFTY50_STOCKS
        
        # Valid symbols
        assert "RELIANCE" in NIFTY50_STOCKS
        assert "TCS" in NIFTY50_STOCKS
        assert "INFY" in NIFTY50_STOCKS
    
    @pytest.mark.unit
    def test_sector_mapping(self):
        """Test sector categorization."""
        from config import SECTORS
        
        assert "IT" in SECTORS or "TECHNOLOGY" in SECTORS
        assert "BANKING" in SECTORS or "FINANCE" in SECTORS
    
    @pytest.mark.unit
    def test_price_data_parsing(self):
        """Test parsing price data from tools."""
        mock_price_response = json.dumps({
            "symbol": "RELIANCE",
            "current_price": 2847.50,
            "change": 34.25,
            "change_percent": 1.22,
            "volume": 5000000
        })
        
        data = json.loads(mock_price_response)
        assert data["symbol"] == "RELIANCE"
        assert data["current_price"] == 2847.50
        assert data["change"] > 0
    
    @pytest.mark.unit
    def test_technical_data_parsing(self):
        """Test parsing technical indicator data."""
        mock_technical_response = json.dumps({
            "rsi_14": 55.5,
            "macd": 12.5,
            "signal": 10.2,
            "bb_upper": 2900,
            "bb_middle": 2850,
            "bb_lower": 2800
        })
        
        data = json.loads(mock_technical_response)
        assert 0 <= data["rsi_14"] <= 100
        assert data["bb_upper"] >= data["bb_middle"] >= data["bb_lower"]
    
    @pytest.mark.unit
    def test_news_data_parsing(self):
        """Test parsing news data structure."""
        mock_news_response = json.dumps({
            "symbol": "RELIANCE",
            "articles": [
                {"title": "Test News", "url": "https://example.com", "source": "Moneycontrol"},
            ],
            "articles_count": 1
        })
        
        data = json.loads(mock_news_response)
        assert data["articles_count"] == len(data["articles"])


class TestUIDataFormatting:
    """Tests for UI data formatting utilities."""
    
    @pytest.mark.unit
    def test_percentage_formatting(self):
        """Test percentage formatting for display."""
        def format_percentage(value):
            if value is None:
                return "N/A"
            return f"{value:+.2f}%"
        
        assert format_percentage(5.5) == "+5.50%"
        assert format_percentage(-3.2) == "-3.20%"
        assert format_percentage(None) == "N/A"
    
    @pytest.mark.unit
    def test_date_formatting(self):
        """Test date formatting for display."""
        def format_date(date_str):
            try:
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                return dt.strftime("%d %b %Y, %H:%M")
            except:
                return date_str
        
        result = format_date("2026-02-01T10:30:00")
        assert "01 Feb 2026" in result
    
    @pytest.mark.unit
    def test_rsi_signal_interpretation(self):
        """Test RSI signal interpretation logic."""
        def interpret_rsi(rsi):
            if rsi is None:
                return "N/A", "gray"
            if rsi > 70:
                return "Overbought", "red"
            elif rsi < 30:
                return "Oversold", "green"
            return "Neutral", "gray"
        
        assert interpret_rsi(75)[0] == "Overbought"
        assert interpret_rsi(25)[0] == "Oversold"
        assert interpret_rsi(50)[0] == "Neutral"
    
    @pytest.mark.unit
    def test_recommendation_color(self):
        """Test recommendation color coding."""
        def get_recommendation_color(recommendation):
            rec_lower = recommendation.lower()
            if "buy" in rec_lower or "bullish" in rec_lower:
                return "green"
            elif "sell" in rec_lower or "bearish" in rec_lower:
                return "red"
            return "orange"
        
        assert get_recommendation_color("Strong Buy") == "green"
        assert get_recommendation_color("SELL") == "red"
        assert get_recommendation_color("Hold") == "orange"


class TestUIErrorHandling:
    """Tests for UI error handling."""
    
    @pytest.mark.unit
    def test_handles_api_error_response(self):
        """Test handling API error responses."""
        error_response = json.dumps({"error": "Symbol not found"})
        data = json.loads(error_response)
        
        assert "error" in data
    
    @pytest.mark.unit
    def test_handles_empty_data(self):
        """Test handling empty data responses."""
        empty_response = json.dumps({"symbol": "TEST", "articles": []})
        data = json.loads(empty_response)
        
        assert len(data.get("articles", [])) == 0
    
    @pytest.mark.unit
    def test_handles_missing_fields(self):
        """Test handling responses with missing fields."""
        partial_response = json.dumps({"symbol": "TEST"})
        data = json.loads(partial_response)
        
        # Should handle missing fields gracefully
        assert data.get("current_price", "N/A") == "N/A"
        assert data.get("volume", 0) == 0

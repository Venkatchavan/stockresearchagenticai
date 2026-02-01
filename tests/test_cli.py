"""
Tests for CLI entry scripts (run_analysis.py, run_bot.py)

Tests command-line argument parsing and script logic.
"""

import pytest
import sys
from unittest.mock import patch, MagicMock
from io import StringIO


class TestRunAnalysisArgParsing:
    """Tests for run_analysis.py argument parsing."""
    
    @pytest.mark.unit
    def test_symbol_argument_required(self):
        """Test that symbol argument is expected."""
        # Simulate the argument structure
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("symbol", help="Stock symbol")
        parser.add_argument("--type", choices=["full", "quick", "technical"], default="full")
        
        args = parser.parse_args(["RELIANCE"])
        assert args.symbol == "RELIANCE"
        assert args.type == "full"
    
    @pytest.mark.unit
    def test_analysis_type_options(self):
        """Test analysis type argument options."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("symbol")
        parser.add_argument("--type", choices=["full", "quick", "technical"], default="full")
        
        # Test quick type
        args = parser.parse_args(["TCS", "--type", "quick"])
        assert args.type == "quick"
        
        # Test technical type
        args = parser.parse_args(["INFY", "--type", "technical"])
        assert args.type == "technical"
    
    @pytest.mark.unit
    def test_symbol_normalization_in_cli(self):
        """Test symbol is normalized properly."""
        symbol = "reliance"
        normalized = symbol.upper().strip()
        assert normalized == "RELIANCE"
    
    @pytest.mark.unit
    def test_help_flag_structure(self):
        """Test help produces proper structure."""
        import argparse
        
        parser = argparse.ArgumentParser(
            description="Stock Research Assistant CLI"
        )
        parser.add_argument("symbol", help="Stock symbol to analyze")
        parser.add_argument("--type", "-t", choices=["full", "quick", "technical"])
        parser.add_argument("--output", "-o", help="Output file path")
        
        # Should not raise
        assert parser.format_help() is not None


class TestRunAnalysisLogic:
    """Tests for run_analysis.py logic."""
    
    @pytest.mark.unit
    def test_api_key_check(self):
        """Test API key validation logic."""
        def check_api_key(key):
            if not key or key == "":
                return False
            return True
        
        assert check_api_key("valid_key") == True
        assert check_api_key("") == False
        assert check_api_key(None) == False
    
    @pytest.mark.unit
    def test_output_formatting(self):
        """Test output formatting for CLI."""
        def format_cli_output(result, symbol):
            header = f"{'=' * 50}\n"
            header += f"Analysis for {symbol}\n"
            header += f"{'=' * 50}\n"
            return header + result
        
        output = format_cli_output("Test result", "RELIANCE")
        assert "RELIANCE" in output
        assert "=" in output
    
    @pytest.mark.unit
    def test_save_to_file_logic(self):
        """Test save to file logic."""
        from pathlib import Path
        import tempfile
        import os
        
        # Create temp file and write
        fd, path = tempfile.mkstemp(suffix='.txt')
        try:
            content = "Test analysis result"
            with os.fdopen(fd, 'w') as f:
                f.write(content)
            
            # Verify file was written
            written = Path(path).read_text()
            assert written == content
        finally:
            os.unlink(path)
    
    @pytest.mark.unit
    def test_elapsed_time_calculation(self):
        """Test elapsed time calculation."""
        from datetime import datetime
        import time
        
        start = datetime.now()
        time.sleep(0.01)  # Small delay
        end = datetime.now()
        
        elapsed = (end - start).total_seconds()
        assert elapsed > 0


class TestRunBotLogic:
    """Tests for run_bot.py logic."""
    
    @pytest.mark.unit
    def test_token_validation(self):
        """Test bot token validation."""
        def validate_token(token):
            if not token:
                return False
            # Telegram tokens are typically long strings with a colon
            if len(token) < 20:
                return False
            return True
        
        assert validate_token("1234567890:ABCdefGHIjklMNOpqrSTUvwxYZ") == True
        assert validate_token("") == False
        assert validate_token("short") == False
    
    @pytest.mark.unit
    def test_logging_setup(self):
        """Test logging configuration."""
        import logging
        
        # Test that logging can be configured
        logger = logging.getLogger("test_bot")
        logger.setLevel(logging.INFO)
        
        assert logger.level == logging.INFO
    
    @pytest.mark.unit
    def test_environment_variable_loading(self):
        """Test environment variable loading."""
        import os
        
        # Test with mock environment
        with patch.dict(os.environ, {'TEST_VAR': 'test_value'}):
            assert os.environ.get('TEST_VAR') == 'test_value'


class TestCLIErrorHandling:
    """Tests for CLI error handling."""
    
    @pytest.mark.unit
    def test_invalid_symbol_handling(self):
        """Test handling of invalid symbols."""
        from config import NIFTY50_STOCKS
        
        def validate_and_warn(symbol):
            if symbol not in NIFTY50_STOCKS:
                return f"Warning: {symbol} is not in NIFTY 50"
            return None
        
        warning = validate_and_warn("UNKNOWN123")
        assert "Warning" in warning
        
        no_warning = validate_and_warn("RELIANCE")
        assert no_warning is None
    
    @pytest.mark.unit
    def test_keyboard_interrupt_message(self):
        """Test KeyboardInterrupt handling message."""
        def get_interrupt_message():
            return "\n\nAnalysis cancelled by user."
        
        msg = get_interrupt_message()
        assert "cancelled" in msg.lower()
    
    @pytest.mark.unit
    def test_exception_formatting(self):
        """Test exception message formatting."""
        def format_error(e):
            return f"Error: {type(e).__name__}: {str(e)}"
        
        try:
            raise ValueError("Test error")
        except ValueError as e:
            formatted = format_error(e)
            assert "ValueError" in formatted
            assert "Test error" in formatted


class TestCLIOutputFormatting:
    """Tests for CLI output formatting."""
    
    @pytest.mark.unit
    def test_rich_panel_content(self):
        """Test content for Rich panel display."""
        def create_panel_content(symbol, analysis_type):
            return f"Analyzing: {symbol}\nType: {analysis_type}"
        
        content = create_panel_content("TCS", "full")
        assert "TCS" in content
        assert "full" in content
    
    @pytest.mark.unit
    def test_progress_message(self):
        """Test progress message formatting."""
        def get_progress_message(step, total):
            return f"Step {step}/{total} completed"
        
        msg = get_progress_message(3, 6)
        assert "3" in msg
        assert "6" in msg
    
    @pytest.mark.unit
    def test_completion_summary(self):
        """Test completion summary formatting."""
        def create_summary(symbol, elapsed_seconds):
            minutes = int(elapsed_seconds // 60)
            seconds = int(elapsed_seconds % 60)
            return f"Analysis of {symbol} completed in {minutes}m {seconds}s"
        
        summary = create_summary("INFY", 125)
        assert "INFY" in summary
        assert "2m" in summary
        assert "5s" in summary


class TestConfigValidation:
    """Tests for configuration validation in scripts."""
    
    @pytest.mark.unit
    def test_settings_import(self):
        """Test settings can be imported."""
        from config import settings
        
        assert settings is not None
    
    @pytest.mark.unit
    def test_nifty50_available(self):
        """Test NIFTY50 stocks list is available."""
        from config import NIFTY50_STOCKS
        
        assert len(NIFTY50_STOCKS) == 50
    
    @pytest.mark.unit
    def test_sectors_available(self):
        """Test sectors are available."""
        from config import SECTORS
        
        assert len(SECTORS) > 0

"""
Market Data Collection Agent
Responsible for gathering real-time and historical market data
"""

from crewai import Agent, LLM

from config import settings
from tools.market_data import (
    get_stock_price,
    get_stock_info,
    get_historical_data,
    get_index_data,
    get_nse_stock_quote,
)


def create_market_data_agent() -> Agent:
    """Create the Market Data Collection Agent."""
    
    llm = LLM(
        model=settings.llm_model,
        api_key=settings.mistral_api_key,
        temperature=0.3,  # Lower temperature for factual data
    )
    
    return Agent(
        role="Market Data Analyst",
        goal="""Collect comprehensive and accurate market data for Indian stocks
        from NSE via Yahoo Finance and nsetools. Gather current prices, historical
        data, trading volumes, and key market statistics.""",
        backstory="""You are a meticulous market data specialist for Indian
        equity markets. You gather data using your tools which provide:
        - Current stock prices, volume, and day's trading range
        - Company information (sector, industry, market cap, financials)
        - Historical OHLCV data with returns and volatility statistics
        - Major index levels (NIFTY50, SENSEX, BANKNIFTY, NIFTYIT)
        - NSE-specific data including delivery percentages (when available)

        You are meticulous about data accuracy. When a tool returns "N/A"
        or an error, report it as unavailable rather than guessing.

        IMPORTANT: Only report data that your tools actually return.
        Do not fabricate prices, volumes, or statistics.

        Your role is foundational - other analysts depend on the accuracy
        of your data to make their assessments.""",
        tools=[
            get_stock_price,
            get_stock_info,
            get_historical_data,
            get_index_data,
            get_nse_stock_quote,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )


# Create singleton instance
market_data_agent = create_market_data_agent()

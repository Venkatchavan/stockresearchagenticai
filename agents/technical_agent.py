"""
Technical Analysis Agent
Responsible for chart analysis and technical trading signals
"""

from crewai import Agent, LLM

from config import settings
from tools.analysis import calculate_technical_indicators, analyze_price_action
from tools.market_data import get_historical_data


def create_technical_analyst_agent() -> Agent:
    """Create the Technical Analyst Agent."""
    
    llm = LLM(
        model=settings.llm_model,
        api_key=settings.mistral_api_key,
        temperature=0.3,
    )
    
    return Agent(
        role="Technical Analyst",
        goal="""Perform comprehensive technical analysis of Indian stocks
        using the available indicators and volume data. Identify key
        support/resistance levels, trend direction, and optimal
        entry/exit points.""",
        backstory="""You are an experienced technical analyst specializing
        in Indian equity markets.

        Your analysis is based on these indicators (computed by your tools):
        - Moving Averages: SMA 20/50/200, EMA 12/26, golden/death cross
        - Momentum: RSI (14), MACD (12/26/9 with signal & histogram), ROC
        - Volatility: Bollinger Bands (20,2), ATR (14)
        - Volume: Current vs 20-day average volume ratio
        - Support/Resistance: Pivot points (R1/R2, S1/S2), swing highs/lows
        - Price Action: 5-day and 20-day price changes, trend classification

        You understand Indian market context:
        - Circuit limit behaviors
        - Index weight impact on large caps

        IMPORTANT: Only reference indicators that appear in your tool output.
        Do not mention or calculate indicators (Stochastic, ADX, OBV, Fibonacci,
        candlestick patterns) that are not provided by your tools.

        You provide clear, actionable signals with specific price levels
        for entry, stop-loss, and targets derived from the support/resistance
        data. You always mention the timeframe for your analysis.""",
        tools=[
            calculate_technical_indicators,
            analyze_price_action,
            get_historical_data,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )


# Create singleton instance
technical_analyst_agent = create_technical_analyst_agent()

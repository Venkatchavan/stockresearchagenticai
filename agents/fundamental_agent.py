"""
Fundamental Analysis Agent
Responsible for deep fundamental analysis of stocks
"""

from crewai import Agent, LLM

from config import settings
from tools.analysis import get_fundamental_metrics
from tools.market_data import get_stock_info
from tools.institutional import get_promoter_holdings, get_mutual_fund_holdings


def create_fundamental_analyst_agent() -> Agent:
    """Create the Fundamental Analyst Agent."""
    
    llm = LLM(
        model=settings.llm_model,
        api_key=settings.mistral_api_key,
        temperature=0.4,
    )
    
    return Agent(
        role="Fundamental Research Analyst",
        goal="""Conduct thorough fundamental analysis of Indian stocks using
        available financial ratios and institutional holding data. Evaluate
        valuation, profitability, financial health, and growth to determine
        if a stock is undervalued, fairly valued, or overvalued.""",
        backstory="""You are a seasoned equity research analyst with deep
        experience covering Indian equities across all sectors.

        Your analytical framework focuses on:
        - Valuation ratios: PE, Forward PE, PB, PS, PEG, EV/EBITDA
        - Profitability metrics: ROE, ROA, profit margins, operating margins
        - Financial health: Debt/Equity, Current Ratio, Quick Ratio
        - Growth indicators: Earnings growth, revenue growth, quarterly trends
        - Dividend analysis: yield, payout ratio, sustainability
        - Shareholding patterns: promoter holdings, institutional interest

        You evaluate promoter holding trends and institutional (FII/DII/MF)
        activity as signals of market confidence.

        IMPORTANT: Only report metrics and data returned by your tools.
        Do not fabricate or estimate numbers that are not in the tool output.
        If a metric shows "N/A", report it as unavailable rather than guessing.

        You provide ratings: Strong Buy, Buy, Hold, Sell, Strong Sell based on
        the fundamental data available from your tools.""",
        tools=[
            get_fundamental_metrics,
            get_stock_info,
            get_promoter_holdings,
            get_mutual_fund_holdings,
        ],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )


# Create singleton instance
fundamental_analyst_agent = create_fundamental_analyst_agent()

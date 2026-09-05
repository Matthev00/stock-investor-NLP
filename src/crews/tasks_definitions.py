from crewai import Task, Agent
from enum import Enum
from typing import Optional

from src.crews.models import StockReportOutput

# Fragments of the single-agent prompt that only make sense while the AlphaVantage tool is wired in.
_ALPHAVANTAGE_STEP = "(7) assess news sentiment from AlphaVantage (analyse_alphavantage_sentiment_tool). "
_SENTIMENT_SOURCES = "Reddit, news, Finnhub, and AlphaVantage"
_SENTIMENT_SOURCES_NO_ALPHAVANTAGE = "Reddit, news, and Finnhub"


class TaskType(Enum):
    """Enumeration of available task types."""
    RESEARCH = "research"
    TECHNICAL_ANALYSIS = "technical_analysis"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    REPORTING = "reporting"
    SCEPTIC = "sceptic"
    TRUST = "trust"
    SYNTHESIS = "synthesis"
    SINGLE_AGENT_ANALYSIS = "single_agent_analysis"


# Task configurations
TASK_CONFIGS = {
    TaskType.RESEARCH: {
        "description": (
            "Gather and analyze qualitative data and public sentiment for '{stock_symbol}' "
            "by processing Reddit discussions (using analyse_reddit_tool), "
            "at least 20 Yahoo News articles (using fetch_yahoo_news_tool), "
            "and Yahoo financial analyses (using fetch_yahoo_analysis_tool). "
            "Focus on identifying the *drivers* of sentiment and the *impact* of news. "
            "Synthesize this information to build a comprehensive picture of the current "
            "sentiment and news landscape surrounding the stock."
        ),
        "expected_output": (
            "A textual report summarizing the current public sentiment, relevant news, "
            "and analyst opinions surrounding '{stock_symbol}'. This report **must**:\n"
            "1.  Provide a concise overview of the general sentiment (positive, negative, mixed) and **highlight the key themes or topics driving this sentiment** based on Reddit analysis.\n"
            "2.  Summarize the **most impactful recent news articles** from Yahoo Finance, explaining their potential implications for the stock.\n"
            "3.  Briefly discuss the consensus from Yahoo financial analyses (e.g., earnings estimates, analyst ratings) and **point out any notable shifts or strong opinions**.\n"
            "4.  Conclude with a **short, insightful synthesis** of these findings, focusing on the overall sentiment narrative.\n"
            "The report should be purely textual and focus on insights and highlights, not raw data dumps."
        ),
    },
    TaskType.TECHNICAL_ANALYSIS: {
        "description": (
            "Perform an in-depth technical analysis of '{stock_symbol}' by fetching "
            "historical market data and calculating a wide array of technical indicators "
            "(using analyse_technical_indicators_tool). Your primary goal is to **interpret these indicators** "
            "to identify trends, patterns, support/resistance levels, and potential trading signals, "
            "explaining their significance."
        ),
        "expected_output": (
            "A detailed textual technical analysis report for '{stock_symbol}'. This report **must**:\n"
            "1.  Begin with an overview of the current price trend (e.g., uptrend, downtrend, consolidation) and its strength.\n"
            "2.  **Interpret the signals from key technical indicators** (like SMAs, EMAs, MACD, RSI, Bollinger Bands, Stochastics). For each indicator, explain what its current state suggests for the stock (e.g., 'RSI at 75 indicates overbought conditions, suggesting a potential pullback.').\n"
            "3.  Identify and describe any significant chart patterns observed (e.g., head and shoulders, double top/bottom) and their implications.\n"
            "4.  Clearly state key support and resistance levels and their importance.\n"
            "5.  Highlight any notable bullish or bearish signals or divergences, explaining the reasoning.\n"
            "6.  Conclude with a **summary of the overall technical outlook** for the stock (e.g., bullish, bearish, neutral with key levels to watch).\n"
            "The report should be purely textual and focus on insights and highlights, not raw data dumps."
        ),
    },
    TaskType.FUNDAMENTAL_ANALYSIS: {
        "description": (
            "Conduct a comprehensive fundamental analysis of '{stock_symbol}' by fetching "
            "and analyzing its financial statements (income, balance sheet, cash flow), "
            "earnings reports, and company overview from Yahoo Finance (using analyse_fundamentals_tool). "
            "Your focus is to **interpret this data** to assess its financial health, profitability, "
            "growth prospects, valuation, and overall intrinsic value, highlighting key strengths and weaknesses."
        ),
        "expected_output": (
            "A detailed textual fundamental analysis report for '{stock_symbol}'. This report **must**:\n"
            "1.  Provide a summary of the company's business model and its current market position based on the overview.\n"
            "2.  Analyze and **interpret key financial metrics and ratios** (e.g., P/E, P/S, Debt-to-Equity, ROE, profit margins) and compare them to historical values or industry peers if possible (even if qualitatively).\n"
            "3.  Discuss trends in revenue, earnings, and cash flow, **highlighting growth drivers or areas of concern**.\n"
            "4.  Assess the company's financial health, focusing on liquidity, solvency, and profitability.\n"
            "5.  Provide an **assessment of the stock's valuation** (e.g., appearing overvalued, undervalued, or fairly valued based on the analysis) and explain the reasoning.\n"
            "6.  Conclude with a **summary of the key fundamental strengths and weaknesses** and the overall fundamental outlook for the company.\n"
            "The report should be purely textual and focus on insights and highlights, not raw data dumps."
        ),
    },
    TaskType.REPORTING: {
        "output_pydantic": StockReportOutput,
        "description": (
            "Synthesize the sentiment analysis, technical analysis, and fundamental analysis "
            "for '{stock_symbol}', drawing from the outputs of the Stock Sentiment Agent, "
            "Technical Analyst, and Fundamental Analyst. Your goal is to formulate a cohesive, "
            "insightful, and definitive investment report. **Do not simply concatenate previous reports.** "
            "Instead, integrate these diverse insights, critically evaluate them, "
            "highlight the most crucial findings and any convergences or divergences, "
            "discuss potential risks and rewards, and provide a clear, actionable investment thesis."
        ),
        "expected_output": (
            "A comprehensive and well-structured textual investment report for '{stock_symbol}'. "
            "This report **must**:\n"
            "1.  Begin with a concise **Executive Summary** that states the overall investment thesis (e.g., Buy, Sell, Hold with price targets if applicable) and the key reasons supporting it.\n"
            "2.  **Synthesize and critically evaluate** key findings from the sentiment analysis. Highlight how public perception and news flow might impact the stock, going beyond just stating the sentiment.\n"
            "3.  **Synthesize and critically evaluate** key findings from the technical analysis. Explain how the technical outlook aligns or contrasts with other analyses and what key levels are critical.\n"
            "4.  **Synthesize and critically evaluate** key findings from the fundamental analysis. Discuss the core financial health and valuation in the context of the overall investment thesis.\n"
            "5.  **Identify and discuss convergences or divergences** between the sentiment, technical, and fundamental analyses. For example, 'While fundamentals appear strong, technical indicators suggest short-term caution.'\n"
            "6.  Clearly outline the **primary catalysts and risk factors** for the stock.\n"
            "7.  Conclude with a **well-reasoned investment outlook and a specific recommendation**, reiterating the main supporting points. The recommendation should be actionable.\n"
            "This report should be a narrative, not just a list of points. Strive for clarity, conciseness, and actionable insights."
        ),
    },
    TaskType.SCEPTIC: {
        "description": (
            "Play devil's advocate for '{stock_symbol}' investment thesis. "
            "Identify risks, weaknesses, potential downsides, and challenges to the bullish case. "
            "Challenge assumptions made by other analysts. Be critical and thorough."
        ),
        "expected_output": (
            "Critical analysis highlighting risks, weaknesses, potential problems, and downsides "
            "for '{stock_symbol}'. Focus on what could go wrong and alternative bearish scenarios."
        ),
    },
    TaskType.TRUST: {
        "description": (
            "Verify data credibility and validate claims about '{stock_symbol}'. "
            "Cross-check information across sentiment, technical, and fundamental analyses. "
            "Confirm source reliability and data accuracy."
        ),
        "expected_output": (
            "Data verification report confirming accuracy and reliability of sources used "
            "in the analysis of '{stock_symbol}'."
        ),
    },
    TaskType.SYNTHESIS: {
        "output_pydantic": StockReportOutput,
        "description": (
            "You are the investment committee leader. Synthesize all perspectives on '{stock_symbol}' "
            "from Researcher, Technical Analyst, Fundamental Analyst, Sceptic, and Trust Specialist. "
            "Integrate their insights, resolve conflicts, address sceptic concerns, and create a final consensus report. "
            "Provide clear investment recommendation (Buy/Sell/Hold) with price target."
        ),
        "expected_output": (
            "Comprehensive investment report including executive summary, sentiment analysis, technical analysis, "
            "fundamental analysis, risk assessment, convergences/divergences, catalysts, and final recommendation."
        ),
    },
    TaskType.SINGLE_AGENT_ANALYSIS: {
        "output_pydantic": StockReportOutput,
        "description": (
            "Conduct a complete, independent investment analysis of '{stock_symbol}'. "
            "Use all available tools to: "
            "(1) gather Reddit sentiment (analyse_reddit_tool), "
            "(2) fetch recent news articles (fetch_yahoo_news_tool), "
            "(3) retrieve analyst estimates (fetch_yahoo_analysis_tool), "
            "(4) compute technical indicators (analyse_technical_indicators_tool), "
            "(5) analyse fundamentals and financials (analyse_fundamentals_tool), "
            "(6) assess market sentiment from Finnhub (analyse_finnhub_sentiment_tool), "
            "(7) assess news sentiment from AlphaVantage (analyse_alphavantage_sentiment_tool). "
            "Synthesize all gathered data into a cohesive, professional investment report."
        ),
        "expected_output": (
            "A comprehensive, well-structured investment report for '{stock_symbol}' covering all of the following sections:\n"
            "1. **Executive Summary** — overall investment thesis (Buy/Sell/Hold) with key supporting reasons and price target.\n"
            "2. **Sentiment Analysis** — synthesis of Reddit, news, Finnhub, and AlphaVantage sentiment; dominant themes driving public perception.\n"
            "3. **Technical Analysis** — interpretation of key indicators (RSI, MACD, SMAs, Bollinger Bands, Stochastics); trend, support/resistance levels, and trading signals.\n"
            "4. **Fundamental Analysis** — financial health assessment (P/E, EPS, revenue growth, margins, cash flow, debt); valuation assessment and key metrics.\n"
            "5. **Risk Assessment** — primary risks, downside scenarios, and factors that could invalidate the thesis.\n"
            "6. **Convergences/Divergences** — where sentiment, technical, and fundamental analyses agree or conflict.\n"
            "7. **Catalysts** — key upcoming events or structural drivers (earnings, product launches, macro factors).\n"
            "8. **Final Recommendation** — actionable Buy/Sell/Hold with specific price target and time horizon.\n"
            "The report must use professional financial language, cite specific data points, and be actionable."
        ),
    },
}


def create_task(
    task_type: TaskType,
    agent: Agent,
    context: Optional[list] = None,
    skip_alphavantage: bool = False,
) -> Task:
    """
    Factory function to create tasks based on type.

    Args:
        task_type: The type of task to create (TaskType enum).
        agent: The agent that will execute the task.
        context: Optional context list for tasks that require previous outputs.
        skip_alphavantage: Drop AlphaVantage instructions when the tool is not wired in.

    Returns:
        A configured Task instance.

    Raises:
        ValueError: If task_type is not a valid TaskType.
    """
    if not isinstance(task_type, TaskType):
        raise ValueError(f"Invalid task type: {task_type}. Must be a TaskType enum value.")

    config = TASK_CONFIGS.get(task_type)
    if not config:
        raise ValueError(f"No configuration found for task type: {task_type}")

    description = config["description"]
    expected_output = config["expected_output"]
    if skip_alphavantage:
        description = description.replace(_ALPHAVANTAGE_STEP, "")
        expected_output = expected_output.replace(_SENTIMENT_SOURCES, _SENTIMENT_SOURCES_NO_ALPHAVANTAGE)

    task_kwargs = {
        "description": description,
        "expected_output": expected_output,
        "agent": agent,
    }

    if "output_pydantic" in config:
        task_kwargs["output_pydantic"] = config["output_pydantic"]

    # Add context if provided
    if context is not None:
        task_kwargs["context"] = context

    return Task(**task_kwargs)

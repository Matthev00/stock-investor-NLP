import logging
import os
import time

from src.utils import dumps
from crewai.tools import tool
from src.services.alphavantage.alphavantage_client import AlphaVantageClient, AlphaVantageRateLimitError
from src.experiments import tool_capture

logger = logging.getLogger(__name__)

_client: AlphaVantageClient | None = None


def _get_client() -> AlphaVantageClient:
    global _client
    if _client is None:
        _client = AlphaVantageClient()
    return _client


def _is_skipped() -> bool:
    return os.getenv("SKIP_ALPHAVANTAGE", "false").lower() == "true"


def _rate_limit_wait_seconds() -> int:
    return int(os.getenv("ALPHAVANTAGE_RATE_LIMIT_WAIT", "3600"))


def _max_retries() -> int:
    return int(os.getenv("ALPHAVANTAGE_MAX_RETRIES", "3"))


@tool
def analyse_alphavantage_sentiment(
    stock_symbol: str,
    limit: int = 50
) -> str:
    """
    Analyzes news sentiment and provides company overview using Alpha Vantage API.
    
    This tool fetches comprehensive sentiment analysis from financial news articles
    and provides detailed company information including fundamental metrics.
    
    Alpha Vantage provides sentiment scores ranging from -1 (most bearish) to +1 (most bullish),
    with labels like 'Bullish', 'Somewhat-Bullish', 'Neutral', 'Somewhat-Bearish', 'Bearish'.
    
    Args:
        stock_symbol (str): The stock ticker symbol to analyze (e.g., 'AAPL', 'TSLA').
        limit (int): Maximum number of news articles to analyze (default: 50, max: 1000).
    
    Returns:
        str: A JSON string containing:
            - News sentiment analysis with average scores
            - Sentiment distribution across categories
            - Recent news articles with individual sentiment scores
            - Company overview with fundamental metrics
            - Earnings data
    
    Example:
        >>> analyse_alphavantage_sentiment("AAPL", limit=50)
        {
            "sentiment_analysis": {
                "ticker": "AAPL",
                "articles_analyzed": 50,
                "average_sentiment_score": 0.234,
                "sentiment_distribution": {
                    "Bullish": 25,
                    "Neutral": 20,
                    "Bearish": 5
                }
            },
            "company_overview": {
                "name": "Apple Inc.",
                "sector": "Technology",
                "market_cap": "3000000000000"
            }
        }
    """
    if _is_skipped():
        skipped = {"skipped": True, "message": "AlphaVantage skipped (SKIP_ALPHAVANTAGE=true)"}
        tool_capture.record("alphavantage_sentiment", dumps(skipped))
        return dumps(skipped, indent=2)

    max_retries = _max_retries()
    wait_seconds = _rate_limit_wait_seconds()

    for attempt in range(max_retries + 1):
        try:
            client = _get_client()
            sentiment_data = client.get_news_sentiment(stock_symbol, limit=limit)
            company_data = client.get_company_overview(stock_symbol)
            earnings_data = client.get_earnings(stock_symbol)
            result = {
                'sentiment_analysis': sentiment_data,
                'company_overview': company_data,
                'earnings': earnings_data,
                'summary': client.get_market_sentiment_summary(stock_symbol)
            }
            json_str = dumps(result, indent=2)
            tool_capture.record("alphavantage_sentiment", json_str)
            return json_str

        except AlphaVantageRateLimitError as e:
            if attempt < max_retries:
                logger.warning(
                    "AlphaVantage rate limited (attempt %d/%d). Waiting %.0fs before retry…",
                    attempt + 1, max_retries, wait_seconds,
                )
                time.sleep(wait_seconds)
            else:
                logger.error("AlphaVantage rate limit exceeded after %d retries: %s", max_retries, e)
                error_response = {"error": "rate_limit", "message": str(e)}
                tool_capture.record("alphavantage_sentiment", dumps(error_response))
                return dumps(error_response, indent=2)

        except Exception as e:
            error_response = {
                "error": str(e),
                "message": f"Failed to analyze data for {stock_symbol} using Alpha Vantage. "
                           "Please check if the stock symbol is valid and API key is configured.",
            }
            return dumps(error_response, indent=2)

    return dumps({"error": "unknown", "message": "Unexpected exit from retry loop"}, indent=2)


@tool
def get_company_fundamentals_alpha(stock_symbol: str) -> str:
    """
    Fetches detailed company fundamentals and overview from Alpha Vantage.
    
    This tool provides comprehensive fundamental analysis data including:
    - Company description and business information
    - Key financial metrics (P/E, PEG, Dividend Yield)
    - Profitability metrics (Profit Margin, ROE, ROA)
    - Valuation metrics (Market Cap, EPS, Beta)
    - 52-week price range
    - Analyst price targets
    
    Args:
        stock_symbol (str): The stock ticker symbol to analyze (e.g., 'AAPL').
    
    Returns:
        str: A JSON string with detailed company fundamentals.
    
    Example:
        >>> get_company_fundamentals_alpha("AAPL")
        {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "sector": "Technology",
            "pe_ratio": "28.5",
            "market_cap": "3000000000000"
        }
    """
    try:
        company_data = _get_client().get_company_overview(stock_symbol)
        return dumps(company_data, indent=2)
    except Exception as e:
        error_response = {
            "error": str(e),
            "message": f"Failed to fetch fundamentals for {stock_symbol}."
        }
        return dumps(error_response, indent=2)

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ExperimentRun(BaseModel):
    instrument: str
    timestamp: datetime
    mode: str
    provider: str
    model: str
    temperature: float
    execution_time: float

    yahoo_news: list[dict[str, Any]] | None = None
    yahoo_fundamentals: dict[str, Any] | None = None
    yahoo_technical_summary: dict[str, Any] | None = None
    yahoo_analysis: dict[str, Any] | None = None
    finnhub_sentiment: dict[str, Any] | None = None
    alphavantage_sentiment: dict[str, Any] | None = None
    reddit_sentiment: dict[str, Any] | None = None

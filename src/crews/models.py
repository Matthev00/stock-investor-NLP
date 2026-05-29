from typing import Literal

from pydantic import BaseModel


class StockReportOutput(BaseModel):
    """Structured output for the final reporting/synthesis task."""

    report: str
    recommendation: Literal["BUY", "SELL", "HOLD"]

from enum import Enum
from typing import Optional

from src.crews.sequential import SequentialStockAnalysisCrew
from src.crews.group_chat import GroupChatStockAnalysisCrew
from src.crews.single_agent import SingleAgentStockAnalysisCrew
from src.config import load_config


class CrewMode(Enum):
    """Enum for crew execution modes."""

    SEQUENTIAL = "sequential"
    GROUP_CHAT = "group_chat"
    SINGLE_AGENT = "single_agent"


class StockAnalysisCrewFactory:
    """Factory for creating crew instances based on mode."""

    @staticmethod
    def create(mode: str, provider: Optional[str] = None, skip_alphavantage: bool = False):
        """
        Create a crew instance based on mode.

        Args:
            mode: "sequential" or "group_chat"
            provider: "gemini" or "openai". If None, uses LLM_PROVIDER from env.
            skip_alphavantage: Omit the AlphaVantage tool so runs record no data for it
                instead of the placeholder values its client returns once the quota is spent.

        Returns:
            Instance of SequentialStockAnalysisCrew or GroupChatStockAnalysisCrew

        Raises:
            ValueError: If mode is not recognized or provider is invalid
        """
        config = load_config(provider)

        if mode.lower() == CrewMode.SEQUENTIAL.value:
            return SequentialStockAnalysisCrew(config, skip_alphavantage=skip_alphavantage)
        elif mode.lower() == CrewMode.GROUP_CHAT.value:
            return GroupChatStockAnalysisCrew(config, skip_alphavantage=skip_alphavantage)
        elif mode.lower() == CrewMode.SINGLE_AGENT.value:
            return SingleAgentStockAnalysisCrew(config, skip_alphavantage=skip_alphavantage)
        else:
            raise ValueError(f"Unknown crew mode: {mode}. Use 'sequential', 'group_chat', or 'single_agent'.")

from enum import Enum

from src.crews.sequential import SequentialStockAnalysisCrew
from src.crews.group_chat import GroupChatStockAnalysisCrew


class CrewMode(Enum):
    """Enum for crew execution modes."""

    SEQUENTIAL = "sequential"
    GROUP_CHAT = "group_chat"


class StockAnalysisCrewFactory:
    """Factory for creating crew instances based on mode."""

    @staticmethod
    def create(mode: str, api_key: str):
        """
        Create a crew instance based on mode.

        Args:
            mode: "sequential" or "group_chat"
            api_key: Google Gemini API key

        Returns:
            Instance of SequentialStockAnalysisCrew or GroupChatStockAnalysisCrew

        Raises:
            ValueError: If mode is not recognized
        """
        if mode.lower() == CrewMode.SEQUENTIAL.value:
            return SequentialStockAnalysisCrew(api_key)
        elif mode.lower() == CrewMode.GROUP_CHAT.value:
            return GroupChatStockAnalysisCrew(api_key)
        else:
            raise ValueError(f"Unknown crew mode: {mode}. Use 'sequential' or 'group_chat'.")

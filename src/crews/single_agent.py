from time import time

from crewai import Crew, LLM

from src.crews.agents_definitions import create_single_agent
from src.crews.tasks_definitions import create_task, TaskType
from src.config import LLMConfig


class SingleAgentStockAnalysisCrew:
    """Single-agent mode — one autonomous analyst with all tools performs the full analysis."""

    def __init__(self, config: LLMConfig, skip_alphavantage: bool = False):
        self.config = config
        self.skip_alphavantage = skip_alphavantage
        self.llm = LLM(
            model=self.config.model_single_agent,
            api_key=self.config.api_key,
            temperature=self.config.temperature,
        )
        self._initialize_agent_and_task()

    def _initialize_agent_and_task(self):
        """Initialize the single agent and its comprehensive analysis task."""
        analyst = create_single_agent(self.llm, skip_alphavantage=self.skip_alphavantage)
        analysis_task = create_task(
            TaskType.SINGLE_AGENT_ANALYSIS, analyst, skip_alphavantage=self.skip_alphavantage
        )

        self.crew = Crew(
            agents=[analyst],
            tasks=[analysis_task],
            cache=True,
            verbose=True,
        )

    def run(self, stock_symbol: str) -> dict:
        """
        Run single-agent analysis and return report with metadata.

        Args:
            stock_symbol: Stock ticker symbol

        Returns:
            Dictionary with mode, provider, execution_time, and report
        """
        start_time = time()
        result = self.crew.kickoff(inputs={"stock_symbol": stock_symbol.upper()})
        execution_time = time() - start_time

        pydantic_out = result.pydantic
        report_text = pydantic_out.report if pydantic_out else str(result)
        recommendation = pydantic_out.recommendation if pydantic_out else None

        return {
            "mode": "single_agent",
            "provider": self.config.provider.value,
            "execution_time": execution_time,
            "report": report_text,
            "recommendation": recommendation,
        }

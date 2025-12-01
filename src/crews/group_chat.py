from time import time

from crewai import Crew, LLM
from crewai.process import Process

from src.crews.agents_definitions import (
    create_researcher_agent,
    create_technical_analyst_agent,
    create_fundamental_analyst_agent,
    create_sceptic_agent,
    create_trust_agent,
    create_leader_agent,
)
from src.crews.tasks_definitions import create_task, TaskType


class GroupChatStockAnalysisCrew:
    """Group Chat mode - 6 agents in hierarchical debate with Leader orchestrating."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm = LLM(
            model="gemini/gemini-2.0-flash",
            api_key=self.api_key,
            temperature=0.5,
        )
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all 6 agents for group chat mode."""
        # Original 3 specialist agents
        self.researcher = create_researcher_agent(self.llm)
        self.technical_analyst = create_technical_analyst_agent(self.llm)
        self.fundamental_analyst = create_fundamental_analyst_agent(self.llm)

        # New 3 agents for debate and synthesis
        self.sceptic = create_sceptic_agent(self.llm)
        self.trust_agent = create_trust_agent(self.llm)
        self.leader = create_leader_agent(self.llm)

    def run(self, stock_symbol: str) -> dict:
        """
        Run group chat mode with hierarchical process - leader orchestrates 6 specialist agents.

        Args:
            stock_symbol: Stock ticker symbol

        Returns:
            Dictionary with mode, execution_time, and report
        """
        start_time = time()

        research_task = create_task(TaskType.RESEARCH, self.researcher)
        technical_task = create_task(TaskType.TECHNICAL_ANALYSIS, self.technical_analyst)
        fundamental_task = create_task(TaskType.FUNDAMENTAL_ANALYSIS, self.fundamental_analyst)
        sceptic_task = create_task(TaskType.SCEPTIC, self.sceptic)
        trust_task = create_task(TaskType.TRUST, self.trust_agent)
        
        synthesis_task = create_task(
            TaskType.SYNTHESIS,
            self.leader,
            context=[research_task, technical_task, fundamental_task, sceptic_task, trust_task],
        )

        crew = Crew(
            agents=[
                self.researcher,
                self.technical_analyst,
                self.fundamental_analyst,
                self.sceptic,
                self.trust_agent,
            ],
            tasks=[
                research_task,
                technical_task,
                fundamental_task,
                sceptic_task,
                trust_task,
                synthesis_task,
            ],
            manager_agent=self.leader,
            process=Process.hierarchical,
            verbose=True,
            cache=True,
            
        )

        try:
            result = crew.kickoff(inputs={"stock_symbol": stock_symbol.upper()})
            execution_time = time() - start_time

            return {
                "mode": "group_chat",
                "execution_time": execution_time,
                "report": str(result),
            }
        except Exception as e:
            execution_time = time() - start_time
            return {
                "mode": "group_chat",
                "execution_time": execution_time,
                "report": f"Group Chat mode failed: {str(e)}. Recommend using Sequential mode.",
                "error": str(e),
            }

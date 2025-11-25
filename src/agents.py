from crewai import LLM, Agent, Crew, Task

from src.tools.reddit_sentiment_analysis_tool import analyse_reddit
from src.tools.yahoo_analysis_tool import fetch_yahoo_analysis
from src.tools.yahoo_fundamental_analysis_tool import analyse_fundamentals
from src.tools.yahoo_news_tool import fetch_yahoo_news
from src.tools.yahoo_technical_analysis_tool import analyse_technical_indicators


class StockAnalysisCrew:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm = LLM(
            model="gemini/gemini-2.0-flash",
            api_key=self.api_key,
            temperature=0.2,
        )
        self._initialize_agents_and_tasks()

    def _initialize_agents_and_tasks(self):
        researcher = Agent(
            role="Senior Stock Market Researcher",
            goal="Gather and analyze comprehensive data about {stock_symbol}",
            backstory="With a Ph.D.in Financial Economics and 15 years of experience in equity research, you're known for your meticulous data collection and insightful analysis.",
            llm=self.llm,
            tools=[analyse_reddit, fetch_yahoo_news, fetch_yahoo_analysis],
            verbose=True,
            memory=True,
        )

        technical_analyst = Agent(
            role="Expert Technical Analyst",
            goal="Perform an in-depth technical analysis on {stock_symbol}",
            verbose=True,
            memory=True,
            backstory="As a Chartered Market Technician (CMT) with 15 years of experience, you have a keen eye for chart patterns and market trends.",
            tools=[analyse_technical_indicators],
            llm=self.llm,
        )

        fundamental_analyst = Agent(
            role="Senior Fundamental Analyst",
            goal="Conduct a comprehensive fundamental analysis of {stock_symbol}",
            verbose=True,
            memory=True,
            backstory="With a CFA charter and 15 years of experience in value investing, you dissect financial statements and identify key value drivers.",
            tools=[analyse_fundamentals],
            llm=self.llm,
        )

        reporter = Agent(
            role="Chief Investment Strategist",
            goal="Synthesize all analyses to create a definitive investment report on {stock_symbol}",
            verbose=True,
            memory=True,
            backstory="As a seasoned investment strategist with 20 years of experience, you weave complex financial data into compelling investment narratives.",
            llm=self.llm,
        )

        research_task = Task(
            description=(
                "Gather and analyze qualitative data and public sentiment for '{stock_symbol}' "
                "by processing Reddit discussions (using analyse_reddit_tool), "
                "at least 20 Yahoo News articles (using fetch_yahoo_news_tool), "
                "and Yahoo financial analyses (using fetch_yahoo_analysis_tool). "
                "Focus on identifying the *drivers* of sentiment and the *impact* of news. "
                "Synthesize this information to build a comprehensive picture of the current "
                "sentiment and news landscape surrounding the stock."
            ),
            expected_output=(
                "A textual report summarizing the current public sentiment, relevant news, "
                "and analyst opinions surrounding '{stock_symbol}'. This report **must**:\n"
                "1.  Provide a concise overview of the general sentiment (positive, negative, mixed) and **highlight the key themes or topics driving this sentiment** based on Reddit analysis.\n"
                "2.  Summarize the **most impactful recent news articles** from Yahoo Finance, explaining their potential implications for the stock.\n"
                "3.  Briefly discuss the consensus from Yahoo financial analyses (e.g., earnings estimates, analyst ratings) and **point out any notable shifts or strong opinions**.\n"
                "4.  Conclude with a **short, insightful synthesis** of these findings, focusing on the overall sentiment narrative.\n"
                "The report should be purely textual and focus on insights and highlights, not raw data dumps."
            ),
            agent=researcher,
        )

        technical_analysis_task = Task(
            description=(
                "Perform an in-depth technical analysis of '{stock_symbol}' by fetching "
                "historical market data and calculating a wide array of technical indicators "
                "(using analyse_technical_indicators_tool). Your primary goal is to **interpret these indicators** "
                "to identify trends, patterns, support/resistance levels, and potential trading signals, "
                "explaining their significance."
            ),
            expected_output=(
                "A detailed textual technical analysis report for '{stock_symbol}'. This report **must**:\n"
                "1.  Begin with an overview of the current price trend (e.g., uptrend, downtrend, consolidation) and its strength.\n"
                "2.  **Interpret the signals from key technical indicators** (like SMAs, EMAs, MACD, RSI, Bollinger Bands, Stochastics). For each indicator, explain what its current state suggests for the stock (e.g., 'RSI at 75 indicates overbought conditions, suggesting a potential pullback.').\n"
                "3.  Identify and describe any significant chart patterns observed (e.g., head and shoulders, double top/bottom) and their implications.\n"
                "4.  Clearly state key support and resistance levels and their importance.\n"
                "5.  Highlight any notable bullish or bearish signals or divergences, explaining the reasoning.\n"
                "6.  Conclude with a **summary of the overall technical outlook** for the stock (e.g., bullish, bearish, neutral with key levels to watch).\n"
                "The report should be purely textual and focus on insights and highlights, not raw data dumps."
            ),
            agent=technical_analyst,
            context=[research_task],
        )

        # Task for Fundamental Analyst
        fundamental_analysis_task = Task(
            description=(
                "Conduct a comprehensive fundamental analysis of '{stock_symbol}' by fetching "
                "and analyzing its financial statements (income, balance sheet, cash flow), "
                "earnings reports, and company overview from Alpha Vantage (using analyse_fundamentals_tool). "
                "Your focus is to **interpret this data** to assess its financial health, profitability, "
                "growth prospects, valuation, and overall intrinsic value, highlighting key strengths and weaknesses."
            ),
            expected_output=(
                "A detailed textual fundamental analysis report for '{stock_symbol}'. This report **must**:\n"
                "1.  Provide a summary of the company's business model and its current market position based on the overview.\n"
                "2.  Analyze and **interpret key financial metrics and ratios** (e.g., P/E, P/S, Debt-to-Equity, ROE, profit margins) and compare them to historical values or industry peers if possible (even if qualitatively).\n"
                "3.  Discuss trends in revenue, earnings, and cash flow, **highlighting growth drivers or areas of concern**.\n"
                "4.  Assess the company's financial health, focusing on liquidity, solvency, and profitability.\n"
                "5.  Provide an **assessment of the stock's valuation** (e.g., appearing overvalued, undervalued, or fairly valued based on the analysis) and explain the reasoning.\n"
                "6.  Conclude with a **summary of the key fundamental strengths and weaknesses** and the overall fundamental outlook for the company.\n"
                "The report should be purely textual and focus on insights and highlights, not raw data dumps."
            ),
            agent=fundamental_analyst,
            context=[research_task],
        )

        # Task for Reporter Agent
        reporting_task = Task(
            description=(
                "Synthesize the sentiment analysis, technical analysis, and fundamental analysis "
                "for '{stock_symbol}', drawing from the outputs of the Stock Sentiment Agent, "
                "Technical Analyst, and Fundamental Analyst. Your goal is to formulate a cohesive, "
                "insightful, and definitive investment report. **Do not simply concatenate previous reports.** "
                "Instead, integrate these diverse insights, critically evaluate them, "
                "highlight the most crucial findings and any convergences or divergences, "
                "discuss potential risks and rewards, and provide a clear, actionable investment thesis."
            ),
            expected_output=(
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
            agent=reporter,
            context=[research_task, technical_analysis_task, fundamental_analysis_task],
        )

        self.crew = Crew(
            agents=[researcher, technical_analyst, fundamental_analyst, reporter],
            tasks=[research_task, technical_analysis_task, fundamental_analysis_task, reporting_task],
            cache=True,
            verbose=True,
        )

    def run(self, stock_symbol: str):
        return self.crew.kickoff(inputs={"stock_symbol": stock_symbol.upper()})

import textwrap
from typing import List, Optional

from deepeval.metrics.faithfulness.template import FaithfulnessTemplate


class StockFaithfulnessTemplate(FaithfulnessTemplate):
    """Domain-specific faithfulness template for stock analysis reports.

    Overrides the generic DeepEval prompts with financial-domain language so
    that the LLM judge focuses on claims relevant to investment analysis:
    price targets, analyst ratings, revenue/earnings figures, sentiment scores,
    technical indicators, and risk assessments.
    """

    @staticmethod
    def generate_claims(actual_output: str, multimodal: bool = False):
        return textwrap.dedent(
            f"""You are a financial analyst reviewing an AI-generated stock investment report.
            Extract a comprehensive list of FACTUAL, VERIFIABLE claims made in the report.

            Focus specifically on:
            - Numerical figures (revenue, EPS, price targets, growth rates, P/E ratios)
            - Analyst ratings and consensus (Buy/Hold/Sell counts, target prices)
            - Sentiment characterizations (bullish, bearish, positive/negative sentiment)
            - Technical indicator signals (RSI levels, moving average crossovers, trend direction)
            - News events or developments cited as justification
            - Risk factors mentioned with specific supporting data

            Claims MUST be self-contained and coherent — include the full context (company name,
            metric name, time period) so they can be verified in isolation.
            Do NOT include subjective opinions or forward-looking statements framed as opinion.

            Example:
            Example Report Excerpt:
            "AAPL reported Q2 revenue of $94.8B, up 5% YoY. Analyst consensus shows 28 Buy
            ratings and 5 Hold ratings. Reddit sentiment is predominantly positive with a
            compound score of 0.72."

            Example JSON:
            {{
                "claims": [
                    "AAPL reported Q2 revenue of $94.8B.",
                    "AAPL Q2 revenue grew 5% year-over-year.",
                    "Analyst consensus shows 28 Buy ratings for AAPL.",
                    "Analyst consensus shows 5 Hold ratings for AAPL.",
                    "Reddit sentiment for AAPL is predominantly positive with a compound score of 0.72."
                ]
            }}
            ===== END OF EXAMPLE =====

            **
            IMPORTANT: Return ONLY valid JSON with a "claims" key containing a list of strings.
            Extract every verifiable factual claim — even if you suspect the claim is incorrect.
            Do NOT add prior knowledge. Take the report at face value.
            **

            Investment Report:
            {actual_output}

            JSON:
            """
        )

    @staticmethod
    def generate_truths(
        retrieval_context: str,
        extraction_limit: Optional[int] = None,
        multimodal: bool = False,
    ):
        if extraction_limit is None:
            limit = "all relevant FACTUAL, undisputed truths"
        elif extraction_limit == 1:
            limit = "the single most important FACTUAL, undisputed truth"
        else:
            limit = f"the {extraction_limit} most important FACTUAL, undisputed truths per document"

        return textwrap.dedent(
            f"""You are a financial data analyst. The text below contains raw data collected
            from financial APIs for a stock analysis task. Extract {limit} that can be
            directly verified from this data.

            Focus specifically on:
            - Exact numerical values (prices, revenue, EPS, growth rates, ratios)
            - Analyst rating counts and consensus price targets
            - Sentiment scores and their direction (positive/negative/neutral)
            - Technical indicator values and signals (RSI, MACD, moving averages)
            - News headlines and their reported dates
            - Reddit/social media discussion tone and volume metrics

            Truths MUST be coherent and include enough context to be verifiable
            (metric name + value + instrument + time period where available).

            Example:
            Example Data:
            "[yahoo_fundamentals] {{\\"trailingPE\\": 28.4, \\"revenueGrowth\\": 0.05,
            \\"recommendationKey\\": \\"buy\\", \\"numberOfAnalystOpinions\\": 33}}"

            Example JSON:
            {{
                "truths": [
                    "The trailing P/E ratio is 28.4.",
                    "Revenue growth is 5% (0.05).",
                    "The analyst recommendation key is 'buy'.",
                    "There are 33 analyst opinions."
                ]
            }}
            ===== END OF EXAMPLE =====

            **
            IMPORTANT: Return ONLY valid JSON with a "truths" key containing a list of strings.
            Only extract what is explicitly stated in the data. Do NOT add outside knowledge.
            **

            Raw Financial Data:
            {retrieval_context}

            JSON:
            """
        )

    @staticmethod
    def generate_verdicts(
        claims: List[str], retrieval_context: str, multimodal: bool = False
    ):
        return textwrap.dedent(
            f"""You are a financial fact-checker. For each claim extracted from an AI-generated
            stock investment report, determine whether it contradicts the raw financial data
            provided as retrieval context.

            Verdict rules:
            - "yes"  → the retrieval context supports or is consistent with the claim
            - "no"   → the retrieval context DIRECTLY contradicts the claim (e.g. wrong number,
                        opposite sentiment direction, incorrect analyst rating count)
            - "idk"  → the retrieval context does not mention this topic, so the claim
                        cannot be verified (treat as unverifiable, NOT as false)

            Important guidelines:
            - Generate EXACTLY one verdict per claim — length of 'verdicts' MUST equal number of claims.
            - Only use "no" when there is a direct, clear contradiction with a specific data point.
            - Rounding differences (e.g. 94.8B vs 94.83B) should NOT be marked "no".
            - Directional claims (e.g. "sentiment is positive") confirmed by a positive score are "yes".
            - Claims about analyst consensus direction that match the data are "yes" even if counts differ slightly.
            - Do NOT use prior financial knowledge — judge only against the provided retrieval context.
            - No reason needed for "yes" verdicts.

            Expected JSON format:
            {{
                "verdicts": [
                    {{"verdict": "yes"}},
                    {{"verdict": "no", "reason": "The report states revenue grew 12% but the data shows revenueGrowth of 0.05 (5%)."}},
                    {{"verdict": "idk", "reason": "The retrieval context contains no data about insider trading activity."}}
                ]
            }}

            **
            IMPORTANT: Return ONLY valid JSON with a 'verdicts' key.
            The number of verdicts MUST strictly equal the number of claims.
            **

            Retrieval Context (raw financial data truths):
            {retrieval_context}

            Claims (from the investment report):
            {claims}

            JSON:
            """
        )

    @staticmethod
    def generate_reason(
        score: float, contradictions: List[str], multimodal: bool = False
    ):
        return textwrap.dedent(
            f"""You are a financial quality assurance analyst reviewing an AI-generated
            stock investment report for factual accuracy.

            Below is a list of contradictions — specific instances where the report's claims
            conflict with the raw financial data the report was based on.

            The faithfulness score is {score} (scale 0–1, higher = more faithful to source data).

            Write a concise, professional assessment that:
            1. States the score and what it means in context of financial reporting accuracy
            2. Summarises the key contradictions found (if any), referencing specific metrics
            3. Notes any patterns (e.g. consistently inflated figures, wrong sentiment direction)

            If there are no contradictions, briefly confirm the report accurately reflects
            the underlying financial data with a professional tone.

            Expected JSON format:
            {{
                "reason": "The score is <faithfulness_score> because <your_reason>."
            }}

            **
            IMPORTANT: Return ONLY valid JSON with a 'reason' key.
            Be specific and reference actual financial metrics where possible.
            Keep the assessment under 3 sentences.
            **

            Faithfulness Score:
            {score}

            Contradictions:
            {contradictions}

            JSON:
            """
        )

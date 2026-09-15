import json
import logging
import os
from typing import Any

from deepeval.metrics import FaithfulnessMetric
from deepeval.models import GeminiModel, GPTModel
from deepeval.test_case import LLMTestCase

from src.config import LLMProvider
from src.utils.stock_faithfulness_template import StockFaithfulnessTemplate

logger = logging.getLogger(__name__)

# DeepEval's own default per-attempt timeout (~88.5s, derived from a 180s outer budget
# split across 2 retries) is tighter than gpt-5 needs on our ~20k-char retrieval context,
# even at low reasoning effort. setdefault so an explicit override (e.g. the one
# rescore_faithfulness.py's batch runs pass on the command line) still wins.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "600")

_RETRIEVAL_FIELDS = [
    "yahoo_news",
    "yahoo_fundamentals",
    "yahoo_technical_summary",
    "yahoo_analysis",
    "finnhub_sentiment",
    "alphavantage_sentiment",
    "reddit_sentiment",
]


def _build_retrieval_context(api_data: dict[str, Any]) -> list[str]:
    context = []
    for field in _RETRIEVAL_FIELDS:
        data = api_data.get(field)
        if data is not None:
            context.append(f"[{field}]\n{json.dumps(data, default=str)}")
    return context


class FaithfulnessEvaluator:
    """LLM-as-judge faithfulness evaluation using DeepEval's FaithfulnessMetric.

    Measures whether claims in the generated report are grounded in the raw
    data collected by the agents (retrieval context).
    """

    def __init__(self, model: str = "gpt-5", threshold: float = 0.5, provider: str = "openai"):
        # Configs use the LiteLLM "gemini/<model>" form for CrewAI; DeepEval wants the bare name.
        self.model_name = model.removeprefix("gemini/")
        self.model = self._build_model(provider, self.model_name)
        self.threshold = threshold

    @staticmethod
    def _build_model(provider: str, model: str) -> GPTModel | GeminiModel:
        """Resolve the judge model for the given provider.

        DeepEval treats a bare model string as an OpenAI model, so both providers are
        built explicitly here instead.
        """
        if LLMProvider(provider.lower()) is LLMProvider.GEMINI:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            return GeminiModel(model=model, api_key=api_key, temperature=0.0)

        # Extracting truths from ~20k characters of raw API data takes a GPT-5 judge past
        # DeepEval's per-attempt timeout at default reasoning; low effort lands it in ~110s.
        generation_kwargs = {"reasoning_effort": "low"} if model.startswith("gpt-5") else None
        return GPTModel(model=model, generation_kwargs=generation_kwargs)

    def evaluate(
        self,
        report_text: str,
        stock_symbol: str,
        api_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Run faithfulness evaluation and return score + reason.

        Args:
            report_text: The generated investment report.
            stock_symbol: Ticker symbol used as the LLM test-case input prompt.
            api_data: Raw data collected by agents (keys from _RETRIEVAL_FIELDS).

        Returns:
            Dict with 'score' (float 0-1 or None), 'reason' (str) and the
            'model' that judged, so results stay attributable to a judge.
        """
        retrieval_context = _build_retrieval_context(api_data)
        if not retrieval_context:
            logger.warning("No retrieval context for %s — skipping faithfulness evaluation", stock_symbol)
            return {"score": None, "reason": "No retrieval context available", "model": self.model_name}

        metric = FaithfulnessMetric(
            threshold=self.threshold,
            model=self.model,
            include_reason=True,
            # An unverifiable claim is a defect in a report meant to rest on the collected data,
            # so it costs the same as a contradicted one instead of being waved through.
            penalize_ambiguous_claims=True,
            evaluation_template=StockFaithfulnessTemplate,
        )
        test_case = LLMTestCase(
            input=f"Generate a comprehensive investment report for {stock_symbol}.",
            actual_output=report_text,
            retrieval_context=retrieval_context,
        )

        try:
            metric.measure(test_case)
            return {"score": metric.score, "reason": metric.reason, "model": self.model_name}
        except Exception as e:
            logger.exception("Faithfulness evaluation failed for %s", stock_symbol)
            return {"score": None, "reason": f"Evaluation failed: {e}", "model": self.model_name}

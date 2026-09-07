import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.crews import StockAnalysisCrewFactory, CrewMode
from src.experiments import tool_capture
from src.experiments.models import ExperimentRun
from src.experiments import serializer
from src.utils.report_evaluator import ReportEvaluator
from src.utils.faithfulness_evaluator import FaithfulnessEvaluator

logger = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self._cfg = yaml.safe_load(f)

        self._n_runs: int = self._cfg["experiment"]["n_runs"]
        exp_cfg = self._cfg["experiment"]
        if "crew_modes" in exp_cfg:
            self._modes: list[str] = exp_cfg["crew_modes"]
        else:
            self._modes = [exp_cfg["crew_mode"]]
        self._output_dir = Path(exp_cfg["output_dir"])
        self._results_csv = Path(self._cfg["experiment"]["results_csv"])

        self._llm_provider: str = self._cfg["llm"]["provider"]
        self._llm_model: str = self._cfg["llm"]["model"]
        self._llm_temperature: float = float(self._cfg["llm"]["temperature"])

        self._tickers: list[dict] = self._cfg["tickers"]

        rl = self._cfg.get("rate_limits", {})
        self._delay_between_runs: float = float(rl.get("delay_between_runs_seconds", 0))
        self._delay_between_tickers: float = float(rl.get("delay_between_tickers_seconds", 0))
        self._skip_alphavantage: bool = bool(rl.get("skip_alphavantage", False))

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._results_csv.parent.mkdir(parents=True, exist_ok=True)
        self._evaluator = ReportEvaluator()
        self._faithfulness_evaluator = FaithfulnessEvaluator()

    def run(self, *, dry_run: bool = False, only_ticker: str | None = None) -> None:
        tickers = self._tickers
        if only_ticker:
            tickers = [t for t in tickers if t["symbol"].upper() == only_ticker.upper()]
            if not tickers:
                raise ValueError(f"Ticker {only_ticker} not found in config.")

        logger.info(
            "Starting experiment: %d mode(s) × %d ticker(s) × %d run(s), provider=%s, model=%s, temperature=%s",
            len(self._modes),
            len(tickers),
            self._n_runs,
            self._llm_provider,
            self._llm_model,
            self._llm_temperature,
        )

        if dry_run:
            for mode in self._modes:
                for entry in tickers:
                    logger.info("[dry-run] Would run %s (%s) mode=%s × %d", entry["symbol"], entry.get("sector", ""), mode, self._n_runs)
            return

        self._set_llm_env()

        failures: list[str] = []
        for mode in self._modes:
            logger.info("=== Starting mode: %s ===", mode)
            for ticker_idx, entry in enumerate(tickers):
                symbol: str = entry["symbol"].upper()
                sector: str = entry.get("sector", "")
                if ticker_idx > 0 and self._delay_between_tickers > 0:
                    logger.info("Rate limit: sleeping %.0fs before next ticker…", self._delay_between_tickers)
                    time.sleep(self._delay_between_tickers)
                for run_idx in range(1, self._n_runs + 1):
                    logger.info("→ [%s] %s run %d/%d", mode, symbol, run_idx, self._n_runs)
                    try:
                        self._execute_run(symbol, sector, run_idx, mode)
                    except Exception:
                        # One failed run must not abandon the rest of the matrix; a long
                        # experiment is expected to hit the occasional API timeout or 429.
                        logger.exception("Run failed: [%s] %s run %d/%d", mode, symbol, run_idx, self._n_runs)
                        failures.append(f"{mode}/{symbol}#{run_idx}")
                    if run_idx < self._n_runs and self._delay_between_runs > 0:
                        logger.info("Rate limit: sleeping %.0fs before next run…", self._delay_between_runs)
                        time.sleep(self._delay_between_runs)

        if failures:
            logger.warning("%d run(s) failed and were skipped: %s", len(failures), ", ".join(failures))
        logger.info("Experiment complete. Results: %s", self._results_csv)

    def _set_llm_env(self) -> None:
        os.environ["LLM_MODEL"] = self._llm_model
        os.environ["LLM_TEMPERATURE"] = str(self._llm_temperature)

    def _execute_run(self, symbol: str, sector: str, run_idx: int, mode: str) -> None:
        timestamp = datetime.now()
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
        stem = f"{symbol}_{ts_str}"

        crew = StockAnalysisCrewFactory.create(mode, self._llm_provider, skip_alphavantage=self._skip_alphavantage)
        tool_capture.start()
        result = crew.run(symbol)
        api_data = tool_capture.collect()

        report_text: str = str(result.get("report", ""))
        evaluation: dict[str, Any] = self._evaluator.evaluate(report_text, symbol)
        faithfulness = self._faithfulness_evaluator.evaluate(report_text, symbol, api_data)
        evaluation["faithfulness"] = faithfulness

        run = ExperimentRun(
            instrument=symbol,
            timestamp=timestamp,
            mode=result.get("mode", mode),
            provider=result.get("provider", self._llm_provider),
            model=self._llm_model,
            temperature=self._llm_temperature,
            execution_time=result.get("execution_time", 0.0),
            recommendation=result.get("recommendation"),
            **api_data,
        )

        stem = serializer.save_run_json(run, self._output_dir)
        serializer.save_report_md(report_text, stem, self._output_dir)
        serializer.save_eval_json(evaluation, stem, self._output_dir)
        serializer.append_csv_row(run, sector, evaluation, self._results_csv)

        faithfulness_score = faithfulness.get("score")
        logger.info(
            "  ✓ saved %s (score=%.1f, grade=%s, recommendation=%s, faithfulness=%s)",
            stem,
            evaluation["overall_score"],
            evaluation["grade"],
            run.recommendation or "N/A",
            f"{faithfulness_score:.2f}" if faithfulness_score is not None else "N/A",
        )

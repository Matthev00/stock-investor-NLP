import csv
import json
import logging
from datetime import datetime
from pathlib import Path

from src.experiments.models import ExperimentRun

logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "timestamp",
    "instrument",
    "sector",
    "mode",
    "provider",
    "model",
    "temperature",
    "execution_time",
    "overall_score",
    "grade",
    "structure",
    "data_richness",
    "sophistication",
    "actionability",
    "sentiment_balance",
    "recommendation",
    "faithfulness_score",
    "judge_model",
]


def make_stem(instrument: str, timestamp: datetime) -> str:
    return f"{instrument}_{timestamp.strftime('%Y%m%d_%H%M%S')}"


def save_run_json(run: ExperimentRun, output_dir: Path) -> str:
    stem = make_stem(run.instrument, run.timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{stem}.json").write_text(
        json.dumps(run.model_dump(), indent=2, default=str), encoding="utf-8"
    )
    return stem


def save_report_md(report_text: str, stem: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{stem}.md").write_text(report_text, encoding="utf-8")


def save_eval_json(evaluation: dict, stem: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{stem}_eval.json").write_text(
        json.dumps(evaluation, indent=2, default=str), encoding="utf-8"
    )


def append_csv_row(run: ExperimentRun, sector: str, evaluation: dict, results_csv: Path) -> None:
    write_header = not results_csv.exists()
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    dim = evaluation.get("dimension_scores", {})
    row = {
        "timestamp": run.timestamp.isoformat(),
        "instrument": run.instrument,
        "sector": sector,
        "mode": run.mode,
        "provider": run.provider,
        "model": run.model,
        "temperature": run.temperature,
        "execution_time": round(run.execution_time, 2),
        "overall_score": round(evaluation.get("overall_score", 0), 2),
        "grade": evaluation.get("grade", ""),
        "structure": round(dim.get("structure", 0), 2),
        "data_richness": round(dim.get("data_richness", 0), 2),
        "sophistication": round(dim.get("sophistication", 0), 2),
        "actionability": round(dim.get("actionability", 0), 2),
        "sentiment_balance": round(dim.get("sentiment_balance", 0), 2),
        "recommendation": run.recommendation or evaluation.get("metrics", {}).get("actionability", {}).get("primary_action", ""),
        "faithfulness_score": evaluation.get("faithfulness", {}).get("score"),
        "judge_model": evaluation.get("faithfulness", {}).get("model", ""),
    }
    fieldnames = CSV_COLUMNS
    if not write_header:
        # A results file written before a column existed keeps its own header, so rows
        # stay aligned with it instead of silently gaining an extra field.
        with open(results_csv, newline="", encoding="utf-8") as f:
            existing = next(csv.reader(f), None)
        if existing and existing != CSV_COLUMNS:
            fieldnames = existing
            missing = [c for c in CSV_COLUMNS if c not in existing]
            if missing:
                logger.warning("%s predates column(s) %s — not recorded for these rows", results_csv, missing)

    with open(results_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

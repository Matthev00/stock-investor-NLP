"""Re-run faithfulness judging on saved experiment runs.

Every run already stores its raw API data and its report, so faithfulness can be
recomputed without paying for report generation again. Use this after a judge
change, or to fill in runs whose evaluation timed out.

    uv run python rescore_faithfulness.py --data-dir experiments/data_v10 \
        --results-csv experiments/results_v10.csv --only-missing --limit 30
"""

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

from src.utils.faithfulness_evaluator import FaithfulnessEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rescore")


def _key(instrument: str, timestamp: str) -> str:
    """Identify a run across files.

    The CSV writes isoformat() while the run json is serialised with str(), which
    separates date and time with a space — parse both back so they line up.
    """
    return f"{instrument}_{datetime.fromisoformat(timestamp).isoformat()}"


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect_runs(data_dir: Path, only_missing: bool) -> list[Path]:
    """Return the stems of runs to rescore, oldest first."""
    stems = []
    for eval_path in sorted(data_dir.glob("*_eval.json")):
        stem = Path(str(eval_path)[: -len("_eval.json")])
        if not stem.with_suffix(".json").exists() or not stem.with_suffix(".md").exists():
            logger.warning("%s: missing run json or report, skipping", stem.name)
            continue
        if only_missing:
            faithfulness = _load(eval_path).get("faithfulness") or {}
            if faithfulness.get("score") is not None:
                continue
        stems.append(stem)
    return stems


def rescore(stem: Path, evaluator: FaithfulnessEvaluator) -> dict:
    """Recompute faithfulness for one run and persist it into its eval json."""
    run = _load(stem.with_suffix(".json"))
    report = stem.with_suffix(".md").read_text(encoding="utf-8")
    eval_path = Path(f"{stem}_eval.json")
    evaluation = _load(eval_path)

    faithfulness = evaluator.evaluate(report, run["instrument"], run)
    evaluation["faithfulness"] = faithfulness
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, default=str)
    return faithfulness


def update_csv(results_csv: Path, scores: dict[str, dict]) -> int:
    """Write recomputed scores back into the results CSV, matched on timestamp."""
    if not results_csv.exists():
        logger.warning("%s does not exist — CSV not updated", results_csv)
        return 0

    with open(results_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    for extra in ("faithfulness_score", "judge_model"):
        if extra not in fieldnames:
            fieldnames.append(extra)

    updated = 0
    for row in rows:
        result = scores.get(_key(row["instrument"], row["timestamp"]))
        if result is None:
            continue
        row["faithfulness_score"] = result.get("score")
        row["judge_model"] = result.get("model", "")
        updated += 1

    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="experiments/data_v10")
    parser.add_argument("--results-csv", default="experiments/results_v10.csv")
    parser.add_argument("--model", default="gpt-5", help="Judge model (default: gpt-5)")
    parser.add_argument("--provider", default="openai", help="openai | gemini")
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Skip runs that already have a score. Omit to rescore everything, "
        "which is what a judge or prompt change requires for comparable results.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Rescore at most N runs. DeepEval opens a fresh HTTP client per call and never "
        "closes it, so long processes drift into timeouts — work in batches.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List what would be rescored.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    stems = collect_runs(data_dir, args.only_missing)
    if args.limit is not None:
        stems = stems[: args.limit]

    logger.info("Rescoring %d run(s) from %s with %s", len(stems), data_dir, args.model)
    if args.dry_run:
        for stem in stems:
            logger.info("[dry-run] would rescore %s", stem.name)
        return

    evaluator = FaithfulnessEvaluator(model=args.model, provider=args.provider)
    scores: dict[str, dict] = {}
    failures = []
    for idx, stem in enumerate(stems, 1):
        logger.info("→ %d/%d %s", idx, len(stems), stem.name)
        try:
            faithfulness = rescore(stem, evaluator)
        except Exception:
            logger.exception("Rescoring failed for %s", stem.name)
            failures.append(stem.name)
            continue
        run = _load(stem.with_suffix(".json"))
        scores[_key(run["instrument"], run["timestamp"])] = faithfulness
        score = faithfulness.get("score")
        logger.info("  score=%s", f"{score:.3f}" if score is not None else "N/A")

    updated = update_csv(Path(args.results_csv), scores)
    scored = sum(1 for v in scores.values() if v.get("score") is not None)
    logger.info("Done: %d scored, %d still unscored, %d CSV row(s) updated", scored, len(stems) - scored, updated)
    if failures:
        logger.warning("%d run(s) raised: %s", len(failures), ", ".join(failures))


if __name__ == "__main__":
    main()

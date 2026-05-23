"""CLI entry point for running the batch experiment."""

import argparse
import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

from src.experiments.runner import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch stock analysis experiment")
    parser.add_argument(
        "--config",
        default="experiment_config.yaml",
        help="Path to experiment YAML config (default: experiment_config.yaml)",
    )
    parser.add_argument(
        "--ticker",
        default=None,
        help="Run only a single ticker (must be present in config tickers list)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without making any API calls",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = ExperimentRunner(config_path=args.config)
    runner.run(dry_run=args.dry_run, only_ticker=args.ticker)


if __name__ == "__main__":
    main()

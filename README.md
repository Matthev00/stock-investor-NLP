# Stock Investment Analysis Platform

[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45.1-orange.svg)](https://streamlit.io)
[![CrewAI](https://img.shields.io/badge/CrewAI-0.119.0-green.svg)](https://www.crewai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Demo

<video src="https://private-user-images.githubusercontent.com/125884790/603565816-58eb803c-848f-4688-8d3b-b046c783c9aa.mp4" controls width="100%"></video>

## Overview

An AI-powered stock analysis platform that generates professional investment reports using a multi-agent system built on CrewAI. Agents collect data from Yahoo Finance, Finnhub, and AlphaVantage, then synthesize it into a structured BUY/HOLD/SELL report with evaluation metrics.

> **Note on Reddit:** The Reddit sentiment tool is currently disabled due to Reddit API access restrictions (OAuth app approval required). All other data sources are fully operational.

## Analysis Modes

| Mode | Agents | Description |
|------|--------|-------------|
| **Sequential** | 4 | Researcher → Technical Analyst → Fundamental Analyst → Reporter |
| **Group Chat** | 6 | Above specialists + Sceptic + Trust Agent, orchestrated by a Leader |
| **Single Agent** | 1 | One agent with all tools — fastest, most direct |

## LLM Configuration

Default: **OpenAI `gpt-4.1`** for all modes.  
Gemini is also supported — switch via UI dropdown or `LLM_PROVIDER` in `.env`.

## Evaluation

Every generated report can be evaluated via the **Evaluate Report Quality** button:

- **Rule-based evaluator** — scores 5 dimensions (Structure, Data Richness, Sophistication, Actionability, Sentiment Balance) visualized as a radar chart.
- **LLM-as-judge Faithfulness** (DeepEval) — extracts *truths* from raw API data and *claims* from the report, then uses `gpt-5` to check whether the report's claims are grounded in the source data. Returns a 0–1 score + written justification. Uses a custom financial-domain prompt template.

## Setup

1. **Clone & install:**
    ```bash
    git clone https://github.com/Matthev00/stock-investor-NLP.git
    cd stock-investor-NLP
    make create_environment && source .venv/bin/activate
    make requirements
    ```

2. **Create `.env`:**
    ```dotenv
    LLM_PROVIDER=openai
    OPENAI_API_KEY=your_openai_api_key
    GEMINI_API_KEY=your_gemini_api_key
    FINNHUB_API_KEY=your_finnhub_api_key
    ALPHA_VANTAGE_API_KEY=your_alphavantage_api_key
    ```

3. **Run:**
    ```bash
    uv run streamlit run src/app.py
    ```

## Using the App

- Enter a ticker (e.g. `AAPL`), select time period and chart type.
- Pick **Technical Indicators** (SMA 20/50/200, EMA 20/50, Bollinger Bands).
- **Update** — refreshes the price chart without running the AI pipeline.
- **Generate Report** — triggers the full multi-agent analysis.
- After the report is generated, click **Evaluate Report Quality** to run both evaluators.
- **Download Report as PDF** to export the analysis.

## Batch Experiment Runner

Run systematic experiments across all modes and tickers without the UI:

```bash
# Preview without API calls
uv run python run_experiment.py --dry-run

# Run with default config
uv run python run_experiment.py

# Custom config (supports multiple modes in one run)
uv run python run_experiment.py --config experiment_config_all_modes.yaml

# Single ticker
uv run python run_experiment.py --ticker AAPL
```

**Config example (`experiment_config_all_modes.yaml`):**
```yaml
experiment:
  n_runs: 5
  crew_modes: [sequential, group_chat, single_agent]
  output_dir: experiments/data_v3
  results_csv: experiments/results_v3.csv

llm:
  provider: openai
  model: gpt-4.1
  temperature: 0.2

rate_limits:
  delay_between_runs_seconds: 15
  delay_between_tickers_seconds: 45
  skip_alphavantage: false
```

**Outputs per run:**
- `<TICKER>_<timestamp>.json` — raw API responses captured from agent tools
- `<TICKER>_<timestamp>.md` — generated investment report
- `<TICKER>_<timestamp>_eval.json` — full evaluation breakdown (scores + faithfulness reason)
- `results_v3.csv` — one row per run with all scores including `faithfulness_score`

Run in background:
```bash
caffeinate -i uv run python run_experiment.py \
  --config experiment_config_all_modes.yaml \
  > experiments/run.log 2>&1 &
tail -f experiments/run.log
```

## Disclaimer

This platform is for informational purposes only and does not constitute financial or investment advice.

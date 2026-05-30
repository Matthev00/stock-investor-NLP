import hashlib
import hmac
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from markdown_it import MarkdownIt

from src.crews import StockAnalysisCrewFactory, CrewMode
from src.config import get_default_provider, LLMProvider, load_config
from src.utils.pdf_exporter import PDFReportExporter
from src.utils.chart_builder import ChartBuilder
from src.utils.report_evaluator import ReportEvaluator
from src.utils.faithfulness_evaluator import FaithfulnessEvaluator
from src.experiments import tool_capture
from src.experiments.models import ExperimentRun
from src.experiments import serializer as exp_serializer

logger = logging.getLogger(__name__)

_APP_OUTPUT_DIR = Path("experiments/data_v2")
_APP_RESULTS_CSV = Path("experiments/results_v2.csv")

INTERVAL_MAPPING = [
    {"period": "1d", "interval": "1m"},
    {"period": "5d", "interval": "30m"},
    {"period": "1mo", "interval": "1d"},
    {"period": "6mo", "interval": "1d"},
    {"period": "ytd", "interval": "1d"},
    {"period": "1y", "interval": "1d"},
    {"period": "5y", "interval": "1wk"},
    {"period": "max", "interval": "1wk"},
]


# Calculate basic metrics from the stock data
def calculate_metrics(data):
    last_close = data["Close"].iloc[-1]
    prev_close = data["Close"].iloc[0]
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100
    high = data["High"].max()
    low = data["Low"].min()
    volume = data["Volume"].sum()
    return last_close, change, pct_change, high, low, volume


# Add simple moving average (SMA) and exponential moving average (EMA) indicators
def add_technical_indicators(data: pd.DataFrame, indicators: dict) -> pd.DataFrame:
    """Add selected technical indicators to the dataframe."""
    return ChartBuilder.add_indicators(data, indicators)


def format_markdown(text):
    md = MarkdownIt()
    tokens = md.parse(text)
    formatted = ""
    for token in tokens:
        if token.type == "paragraph_open":
            formatted += "\n\n"
        formatted += token.content
    return formatted.strip()


def escape_markdown_specials(text: str) -> str:
    text = text.replace("$", r"\$")
    return text


def load_stock_data(symbol: str, period: dict) -> pd.DataFrame:
    """Load and process stock data - wrapper for ChartBuilder."""
    return ChartBuilder.load_and_process_data(symbol, period["period"])


if "stock_fig" not in st.session_state:
    st.session_state.stock_fig = None
if "stock_metrics" not in st.session_state:
    st.session_state.stock_metrics = None
if "report" not in st.session_state:
    st.session_state.report = None
if "report_mode" not in st.session_state:
    st.session_state.report_mode = None
if "report_provider" not in st.session_state:
    st.session_state.report_provider = None
if "execution_time" not in st.session_state:
    st.session_state.execution_time = None
if "report_recommendation" not in st.session_state:
    st.session_state.report_recommendation = None
if "selected_indicators" not in st.session_state:
    st.session_state.selected_indicators = {}
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None
if "run_stem" not in st.session_state:
    st.session_state.run_stem = None
if "run_record" not in st.session_state:
    st.session_state.run_record = None
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

st.set_page_config("Stock Investment Report", layout="wide")

if not st.session_state.authenticated:
    st.title("🔐 Stock Investment Analysis Platform")
    st.subheader("Login")
    password = st.text_input("Password", type="password")
    if st.button("Login", type="primary"):
        expected = st.secrets["password_hash"]
        entered = hashlib.sha256(password.encode()).hexdigest()
        if hmac.compare_digest(entered, expected):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

st.title("📈 Stock Investment Analysis Platform")


if st.sidebar.button("Logout", type="secondary"):
    st.session_state.authenticated = False
    st.rerun()

st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Stock symbol (eg. AAPL)")
time_period = st.sidebar.selectbox("Time period", [period["period"] for period in INTERVAL_MAPPING])
chart_type = st.sidebar.selectbox("Chart Type", ["Candlestick", "Line"])

llm_provider = st.sidebar.selectbox(
    "LLM Provider",
    options=[provider.value for provider in LLMProvider],
    index=0 if get_default_provider().lower() == "gemini" else 1,
    format_func=lambda x: "Gemini" if x == "gemini" else "OpenAI",
)

crew_mode = st.sidebar.radio(
    "Analysis Mode",
    options=[CrewMode.SEQUENTIAL.value, CrewMode.GROUP_CHAT.value, CrewMode.SINGLE_AGENT.value],
    format_func=lambda x: {
        CrewMode.SEQUENTIAL.value: "Sequential",
        CrewMode.GROUP_CHAT.value: "Group Chat",
        CrewMode.SINGLE_AGENT.value: "Single Agent",
    }[x],
    horizontal=True
)

st.sidebar.subheader("📊 Technical Indicators")
with st.sidebar.expander("Select Indicators", expanded=True):
    st.write("**Moving Averages**")
    indicators = {
        "SMA 20": st.checkbox("SMA 20", value=True),
        "SMA 50": st.checkbox("SMA 50", value=False),
        "SMA 200": st.checkbox("SMA 200", value=False),
        "EMA 20": st.checkbox("EMA 20", value=False),
        "EMA 50": st.checkbox("EMA 50", value=False),
    }
    
    st.write("**Volatility**")
    indicators.update({
        "Bollinger Bands": st.checkbox("Bollinger Bands", value=False),
    })
    
sidebar_col1, sidebar_col2 = st.sidebar.columns(spec=[0.4, 0.6], gap="small")

if sidebar_col1.button("Update", type="primary", use_container_width=True):
    data = load_stock_data(ticker, next(filter(lambda x: x["period"] == time_period, INTERVAL_MAPPING)))

    last_close, change, pct_change, high, low, volume = calculate_metrics(data)
    st.session_state.stock_metrics = {
        "last_close": last_close,
        "change": change,
        "pct_change": pct_change,
        "high": high,
        "low": low,
        "volume": volume,
    }

    # Add selected technical indicators
    data = add_technical_indicators(data, indicators)

    # Build chart using ChartBuilder
    fig = ChartBuilder.build_chart(data, ticker, indicators, time_period, chart_type)

    st.session_state.stock_fig = fig
    st.session_state.selected_indicators = indicators

if sidebar_col2.button("Generate report", type="primary", use_container_width=True):
    with st.spinner("Running multi-agent analysis…"):
        try:
            crew = StockAnalysisCrewFactory.create(crew_mode, llm_provider)
            tool_capture.start()
            result = crew.run(ticker)
            _api_data = tool_capture.collect()

            report_md = format_markdown(str(result["report"]))
            report_cleaned = escape_markdown_specials(report_md)
            st.session_state.report = report_cleaned
            st.session_state.report_mode = result["mode"]
            st.session_state.report_provider = result["provider"]
            st.session_state.execution_time = result["execution_time"]
            st.session_state.report_recommendation = result.get("recommendation")
            try:
                cfg = load_config(llm_provider)
                model = cfg.model_group_chat if crew_mode == CrewMode.GROUP_CHAT.value else cfg.model_sequential
                api_data = _api_data
                run_record = ExperimentRun(
                    instrument=ticker.upper(),
                    timestamp=datetime.now(),
                    mode=result["mode"],
                    provider=result["provider"],
                    model=model,
                    temperature=cfg.temperature,
                    execution_time=result["execution_time"],
                    recommendation=result.get("recommendation"),
                    **api_data,
                )
                stem = exp_serializer.save_run_json(run_record, _APP_OUTPUT_DIR)
                exp_serializer.save_report_md(report_cleaned, stem, _APP_OUTPUT_DIR)
                st.session_state.run_stem = stem
                st.session_state.run_record = run_record
            except Exception as e:
                logger.exception("Failed to serialize run data: %s", e)
                st.warning(f"Report saved but serialization failed: {e}")
        except ValueError as e:
            st.error(f"Configuration Error: {str(e)}\n\nPlease ensure API keys are set in your .env file.")

if st.session_state.stock_metrics is not None:
    last_close = st.session_state.stock_metrics["last_close"]
    change = st.session_state.stock_metrics["change"]
    pct_change = st.session_state.stock_metrics["pct_change"]
    high = st.session_state.stock_metrics["high"]
    low = st.session_state.stock_metrics["low"]
    volume = st.session_state.stock_metrics["volume"]

    st.metric(
        label=f"{ticker} Last Price",
        value=f"{last_close:.2f} USD",
        delta=f"{change:.2f} ({pct_change:.2f}%)",
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("High", f"{high:.2f} USD")
    col2.metric("Low", f"{low:.2f} USD")
    col3.metric("Volume", f"{volume:,}")

if st.session_state.stock_fig is not None:
    st.plotly_chart(st.session_state.stock_fig, use_container_width=True, key="chart_display")

if st.session_state.report is not None:
    st.header("Investment Report")
    
    # Display metadata
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        mode_label = {
            CrewMode.SEQUENTIAL.value: "Sequential (Original)",
            CrewMode.GROUP_CHAT.value: "Group Chat (FinDebate)",
            CrewMode.SINGLE_AGENT.value: "Single Agent (Autonomous)",
        }.get(st.session_state.report_mode, st.session_state.report_mode)
        st.metric("Analysis Mode", mode_label)
    with col2:
        provider_label = "Gemini" if st.session_state.report_provider == "gemini" else "OpenAI"
        st.metric("LLM Provider", provider_label)
    with col3:
        st.metric("Execution Time", f"{st.session_state.execution_time:.1f}s")
    with col4:
        rec = st.session_state.report_recommendation
        rec_icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(rec, "⚪")
        st.metric("Recommendation", f"{rec_icon} {rec}" if rec else "N/A")
    
    st.divider()
    
    # Export to PDF button
    exporter = PDFReportExporter()
    pdf_buffer = exporter.export(
        ticker=ticker,
        report_text=st.session_state.report,
        fig=st.session_state.stock_fig,
        metrics=st.session_state.stock_metrics,
        indicators=st.session_state.selected_indicators,
        mode=st.session_state.report_mode,
        provider=st.session_state.report_provider,
        execution_time=st.session_state.execution_time
    )

    col_pdf, col_eval = st.columns(2)
    
    with col_pdf:
        st.download_button(
            label="📥 Download Report as PDF",
            data=pdf_buffer,
            file_name=f"{ticker.upper()}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            key="download_pdf_btn"
        )
    
    with col_eval:
        if st.button("📊 Evaluate Report Quality", type="secondary", use_container_width=True):
            with st.spinner("Evaluating report quality..."):
                evaluator = ReportEvaluator()
                st.session_state.evaluation_results = evaluator.evaluate(
                    st.session_state.report,
                    ticker.upper()
                )
                if st.session_state.run_record:
                    fa_evaluator = FaithfulnessEvaluator()
                    faithfulness = fa_evaluator.evaluate(
                        st.session_state.report,
                        ticker.upper(),
                        st.session_state.run_record.model_dump(),
                    )
                    st.session_state.evaluation_results["faithfulness"] = faithfulness
                try:
                    if st.session_state.run_stem and st.session_state.run_record:
                        exp_serializer.save_eval_json(
                            st.session_state.evaluation_results,
                            st.session_state.run_stem,
                            _APP_OUTPUT_DIR,
                        )
                        exp_serializer.append_csv_row(
                            st.session_state.run_record,
                            "",
                            st.session_state.evaluation_results,
                            _APP_RESULTS_CSV,
                        )
                except Exception as e:
                    logger.exception("Failed to serialize evaluation: %s", e)
                    st.warning(f"Evaluation displayed but not saved: {e}")

    st.divider()
    
    # Display evaluation results if available
    if st.session_state.evaluation_results is not None:
        eval_data = st.session_state.evaluation_results
        
        st.subheader("📊 Report Quality Evaluation")
        
        # Overall score display
        score_col1, score_col2, score_col3 = st.columns([1, 1, 1])
        
        with score_col1:
            score = eval_data['overall_score']
            grade = eval_data['grade']
            
            # Color based on grade
            if grade == 'A':
                score_color = "🟢"
            elif grade == 'B':
                score_color = "🟡"
            elif grade == 'C':
                score_color = "🟠"
            else:
                score_color = "🔴"
            
            st.metric("Overall Quality Score", f"{score:.1f}/100", delta=f"Grade: {grade}")
            st.markdown(f"{score_color} **Quality Level:** {grade}")
        
        with score_col2:
            best_dim = max(eval_data['dimension_scores'].items(), key=lambda x: x[1])
            st.metric("Strongest Dimension", best_dim[0].replace('_', ' ').title(), f"{best_dim[1]:.1f}/100")
        
        with score_col3:
            worst_dim = min(eval_data['dimension_scores'].items(), key=lambda x: x[1])
            st.metric("Needs Improvement", worst_dim[0].replace('_', ' ').title(), f"{worst_dim[1]:.1f}/100")

        st.markdown("**Dimension Scores**")
        dim_cols = st.columns(len(eval_data['dimension_scores']))
        for col, (dim_name, dim_val) in zip(dim_cols, eval_data['dimension_scores'].items()):
            delta_color = "normal" if dim_val >= 70 else "inverse"
            col.metric(
                label=dim_name.replace('_', ' ').title(),
                value=f"{dim_val:.0f} / 100",
                delta="✓ OK" if dim_val >= 70 else "⚠ Low",
                delta_color=delta_color,
            )

        st.divider()
        
        # Dimension scores with visualization
        st.subheader("Quality Dimensions")
        
        dim_scores = eval_data['dimension_scores']
        
        # Create radar chart data
        import plotly.graph_objects as go
        
        categories = [k.replace('_', ' ').title() for k in dim_scores.keys()]
        values = list(dim_scores.values())
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Score',
            line_color='rgb(31, 119, 180)',
            fillcolor='rgba(31, 119, 180, 0.3)'
        ))
        
        # Add target line at 70
        fig_radar.add_trace(go.Scatterpolar(
            r=[70] * len(categories),
            theta=categories,
            mode='lines',
            name='Target (70)',
            line=dict(color='green', dash='dash', width=2)
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title="Quality Dimensions Radar Chart",
            height=600
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Detailed metrics in expandable sections
        with st.expander("📋 Detailed Metrics", expanded=False):
            metrics_tabs = st.tabs([
                "Structure",
                "Data Richness",
                "Professional Sophistication",
                "Actionability",
                "Sentiment"
            ])
            
            with metrics_tabs[0]:
                st.json(eval_data['metrics']['structure'])
            
            with metrics_tabs[1]:
                st.json(eval_data['metrics']['data_richness'])
            
            with metrics_tabs[2]:
                st.json(eval_data['metrics']['sophistication'])
            
            with metrics_tabs[3]:
                st.json(eval_data['metrics']['actionability'])
            
            with metrics_tabs[4]:
                st.json(eval_data['metrics']['sentiment'])
        
        # Recommendations
        if eval_data['recommendations']:
            st.subheader("💡 Improvement Recommendations")
            
            for i, rec in enumerate(eval_data['recommendations'], 1):
                priority_emoji = {
                    'Critical': '🔴',
                    'High': '🟠',
                    'Medium': '🟡',
                    'Low': '🟢',
                    'Info': 'ℹ️'
                }.get(rec['priority'], 'ℹ️')
                
                with st.container():
                    st.markdown(f"**{priority_emoji} {rec['category']} ({rec['priority']} Priority)**")
                    st.markdown(f"*Issue:* {rec['issue']}")
                    st.markdown(f"*Suggestion:* {rec['suggestion']}")
                    if i < len(eval_data['recommendations']):
                        st.divider()

        # Faithfulness (LLM-as-judge)
        if "faithfulness" in eval_data:
            st.divider()
            faithfulness = eval_data["faithfulness"]
            f_score = faithfulness.get("score")
            st.subheader("🔍 Faithfulness (LLM-as-judge)")
            f_col1, f_col2 = st.columns([1, 3])
            with f_col1:
                st.metric(
                    "Faithfulness Score",
                    f"{f_score:.2f}" if f_score is not None else "N/A",
                    "DeepEval",
                )
            with f_col2:
                st.info(faithfulness.get("reason") or "No reason available")
    
    st.divider()
    st.markdown(st.session_state.report)


if not st.session_state.get("stock_fig") and not st.session_state.get("report"):
    st.markdown(
        """
## AI-Powered Stock Analysis Platform

Welcome to stock analysis platform! It uses Artificial Intelligence and Large Language Models (LLMs) to provide professional investment insights.

**Key Features:**

* Analyzes sentiment from Reddit (wallstreetbets, stocks, investing subreddits).
* Performs detailed fundamental and technical analysis.
* Integrates research from the web and news sources.

To get a detailed, AI-generated report, select a stock symbol and provide Google Gemini API key. This platform is designed to help investors make data-driven decisions in the stock market.

**Disclaimer:** This analysis is for informational purposes only and is not financial or investment advice."""
    )

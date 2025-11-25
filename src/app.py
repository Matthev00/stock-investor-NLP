import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import talib as ta
import yfinance as yf
from markdown_it import MarkdownIt

from src.agents import StockAnalysisCrew

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


def process_data(ticker, data):
    data = data.xs(ticker, axis=1, level=1)
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize("UTC")
    data.index = data.index.tz_convert("US/Eastern")
    data.reset_index(inplace=True)
    data.rename(columns={"Date": "Datetime"}, inplace=True)
    return data


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
def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    # data["SMA_20"] = ta.trend.sma_indicator(data["Close"], window=20)
    # data["EMA_20"] = ta.trend.ema_indicator(data["Close"], window=20)
    data["SMA_20"] = ta.SMA(data["Close"].to_numpy().flatten(), timeperiod=20)
    data["EMA_20"] = ta.EMA(data["Close"].to_numpy().flatten(), timeperiod=20)
    return data


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
    return yf.download(
        symbol,
        period=period["period"],
        interval=period["interval"],
        auto_adjust=True,
        progress=False,
    )


if "stock_fig" not in st.session_state:
    st.session_state.stock_fig = None
if "stock_metrics" not in st.session_state:
    st.session_state.stock_metrics = None
if "report" not in st.session_state:
    st.session_state.report = None


st.set_page_config("Stock Investment Report", layout="wide")
st.title("📈 Stock Investment Analysis Platform")


st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Stock symbol (eg. AAPL)")
time_period = st.sidebar.selectbox("Time period", [period["period"] for period in INTERVAL_MAPPING])
chart_type = st.sidebar.selectbox("Chart Type", ["Candlestick", "Line"])
api_key = st.sidebar.text_input("Gemini API key", type="password")
sidebar_col1, sidebar_col2 = st.sidebar.columns(spec=[0.4, 0.6], gap="small")

if sidebar_col1.button("Update", type="primary", use_container_width=True):
    data = load_stock_data(ticker, next(filter(lambda x: x["period"] == time_period, INTERVAL_MAPPING)))
    data = process_data(ticker, data)

    last_close, change, pct_change, high, low, volume = calculate_metrics(data)
    st.session_state.stock_metrics = {
        "last_close": last_close,
        "change": change,
        "pct_change": pct_change,
        "high": high,
        "low": low,
        "volume": volume,
    }

    fig = go.Figure()
    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=data["Datetime"],
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
            )
        )
    else:
        fig = px.line(data, x="Datetime", y="Close")

    fig.update_layout(
        title=f"{ticker} {time_period.upper()} Chart",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        height=600,
    )

    st.session_state.stock_fig = fig

if sidebar_col2.button("Generate report", type="primary", use_container_width=True):
    with st.spinner("Running multi-agent analysis…"):
        result = StockAnalysisCrew(api_key).run(ticker)

        report_md = format_markdown(str(result))
        report_cleaned = escape_markdown_specials(report_md)
        st.session_state.report = report_cleaned

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
    st.plotly_chart(st.session_state.stock_fig, use_container_width=True)

if st.session_state.report is not None:
    st.header("Investment Report")
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

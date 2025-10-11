"""
Streamlit app for technical analysis of financial data.
"""

from typing import Any
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime, timedelta
import os
import time
import yfinance as yf

from src.analysis.technical_indicators import TechnicalAnalysis
from src.visualization.visualizer import Visualizer
from src.data.data_fetcher import DataFetcher

st.set_page_config(
    page_title="Financial Technical Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme" not in st.session_state:
    st.session_state.theme = "light"

def toggle_theme():
    """Toggle between light and dark theme."""
    if st.session_state.theme == "light":
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"

def apply_custom_css():
    """Apply custom CSS based on the selected theme."""
    if st.session_state.theme == "dark":
        st.markdown("""
        <style>
        .main {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .stApp {
            background-color: #0E1117;
        }
        .css-1d391kg, .css-12oz5g7 {
            background-color: #262730;
        }
        .st-bq {
            background-color: #262730;
        }
        .st-c0 {
            background-color: #0E1117;
        }
        .st-bw {
            color: #FAFAFA;
        }
        .st-bs {
            color: #FAFAFA;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .main {
            background-color: #FFFFFF;
            color: #31333F;
        }
        .stApp {
            background-color: #FFFFFF;
        }
        .css-1d391kg, .css-12oz5g7 {
            background-color: #F0F2F6;
        }
        .st-bq {
            background-color: #F0F2F6;
        }
        .st-c0 {
            background-color: #FFFFFF;
        }
        .st-bw {
            color: #31333F;
        }
        .st-bs {
            color: #31333F;
        }
        </style>
        """, unsafe_allow_html=True)

apply_custom_css()


def prepare_indicator_configs(
    use_sma,
    sma_short_length,
    sma_long_length,
    use_ema,
    ema_short_length,
    ema_long_length,
    use_rsi,
    rsi_length,
    use_macd,
    macd_fast,
    macd_slow,
    macd_signal,
    use_bbands,
    bb_length,
    bb_std,
    use_stoch,
    stoch_k,
    stoch_d,
    use_adx,
    adx_length,
    use_willr,
    willr_length,
):
    """Build the list of indicator configuration dictionaries."""

    indicators = []

    if use_sma:
        indicators.append({"name": "SMA", "params": {"length": sma_short_length}})
        indicators.append({"name": "SMA", "params": {"length": sma_long_length}})

    if use_ema:
        indicators.append({"name": "EMA", "params": {"length": ema_short_length}})
        indicators.append({"name": "EMA", "params": {"length": ema_long_length}})

    if use_rsi:
        indicators.append({"name": "RSI", "params": {"length": rsi_length}})

    if use_macd:
        indicators.append(
            {
                "name": "MACD",
                "params": {"fast": macd_fast, "slow": macd_slow, "signal": macd_signal},
            }
        )

    if use_bbands:
        indicators.append(
            {
                "name": "BBands",
                "params": {"length": bb_length, "std": bb_std},
            }
        )

    if use_stoch:
        indicators.append({"name": "Stochastic", "params": {"k": stoch_k, "d": stoch_d}})

    if use_adx:
        indicators.append({"name": "ADX", "params": {"length": adx_length}})

    if use_willr:
        indicators.append({"name": "WILLR", "params": {"length": willr_length}})

    return indicators


def sanitize_symbol(symbol):
    """Normalize symbol strings entered by users."""

    return symbol.replace(" ", "").upper()


def parse_symbol_list(raw_symbols, primary_symbol):
    """Turn a raw comma/newline separated string into a list of unique symbols."""

    if not raw_symbols:
        return []

    candidates = [sanitize_symbol(s) for s in raw_symbols.replace("\n", ",").split(",")]
    symbols = [s for s in candidates if s]

    # De-duplicate while preserving order and avoiding the primary symbol twice
    unique_symbols: list[Any] = []
    for sym in symbols:
        if sym and sym not in unique_symbols and sym != primary_symbol:
            unique_symbols.append(sym)

    return unique_symbols


def summarize_performance(df):
    """Create a summary of recent performance metrics for a dataframe."""

    if df is None or df.empty or "Close" not in df.columns:
        return np.nan, np.nan, np.nan

    last_close = df["Close"].iloc[-1]
    first_close = df["Close"].iloc[0]
    period_return = (last_close / first_close - 1) if first_close else np.nan

    daily_returns = df["Close"].pct_change().dropna()
    annualized_volatility = daily_returns.std() * np.sqrt(252) if not daily_returns.empty else np.nan

    return last_close, period_return, annualized_volatility


def fetch_real_time_data(symbol, period="1d", interval="1m"):
    """Fetch real-time data for a symbol."""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        return data
    except Exception as e:
        st.error(f"Error fetching real-time data for {symbol}: {e}")
        return pd.DataFrame()


col1, col2 = st.columns([5, 1])
with col1:
    st.title("📊 Financial Technical Analysis")
with col2:
    theme_icon = "🌙" if st.session_state.theme == "light" else "☀️"
    st.button(theme_icon, on_click=toggle_theme, key="theme_toggle")

st.markdown("Analyze single or multiple stocks, visualize indicators, and backtest trading ideas.")

# Main analysis section (merged from Single Symbol and Multi-Symbol tabs)
st.subheader("Stock Analysis")
st.caption("Analyze one or multiple stocks with technical indicators. Enter a primary symbol and additional comparison symbols.")

with st.sidebar:
    st.header("Configuration")

    raw_symbol = st.text_input(
        "Primary Stock Symbol",
        value="AAPL",
        help="Enter the main ticker symbol to analyze (e.g., AAPL, MSFT, GOOGL).",
    )
    symbol = sanitize_symbol(raw_symbol) if raw_symbol else ""

    multi_symbols_raw = st.text_area(
        "Compare Symbols",
        value="MSFT, GOOGL",
        help="Provide additional ticker symbols separated by commas or new lines for multi-symbol analysis.",
    )

    st.subheader("Time Period")
    period_options = {
        "1 Day (Real-time)": "1d",
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
        "Max": "max",
    }
    period = st.selectbox("Select Period", options=list(period_options.keys()), index=3)
    period_value = period_options[period]

    interval_options = {
        "1 Minute (Real-time)": "1m",
        "1 Hour": "1h",
        "1 Day": "1d",
        "1 Week": "1wk",
        "1 Month": "1mo",
    }
    
    # Adjust interval options based on period
    if period_value == "1d":
        interval = st.selectbox("Select Interval", options=["1 Minute (Real-time)", "5 Minutes", "15 Minutes", "30 Minutes", "1 Hour"], index=0)
        interval_map = {
            "1 Minute (Real-time)": "1m",
            "5 Minutes": "5m",
            "15 Minutes": "15m",
            "30 Minutes": "30m",
            "1 Hour": "1h"
        }
        interval_value = interval_map[interval]
    else:
        interval = st.selectbox("Select Interval", options=list(interval_options.keys()), index=2)
        interval_value = interval_options[interval]

    st.subheader("Technical Indicators")

    st.markdown("##### Moving Averages")
    use_sma = st.checkbox("Simple Moving Average (SMA)", value=True)
    sma_short_length = 20
    sma_long_length = 50
    if use_sma:
        sma_col1, sma_col2 = st.columns(2)
        with sma_col1:
            sma_short_length = st.number_input("SMA Short Length", min_value=5, max_value=100, value=20, step=5)
        with sma_col2:
            sma_long_length = st.number_input("SMA Long Length", min_value=50, max_value=200, value=50, step=5)

    use_ema = st.checkbox("Exponential Moving Average (EMA)", value=True)
    ema_short_length = 12
    ema_long_length = 50
    if use_ema:
        ema_col1, ema_col2 = st.columns(2)
        with ema_col1:
            ema_short_length = st.number_input("EMA Short Length", min_value=5, max_value=100, value=12, step=5)
        with ema_col2:
            ema_long_length = st.number_input("EMA Long Length", min_value=50, max_value=200, value=50, step=5)

    st.markdown("##### Oscillators")
    use_rsi = st.checkbox("Relative Strength Index (RSI)", value=True)
    rsi_length = st.slider(
        "RSI Length", min_value=5, max_value=30, value=14, step=1, disabled=not use_rsi
    )

    use_macd = st.checkbox("MACD", value=False)
    macd_fast, macd_slow, macd_signal = 12, 26, 9
    if use_macd:
        macd_col1, macd_col2 = st.columns(2)
        with macd_col1:
            macd_fast = st.number_input(
                "Fast Length", min_value=5, max_value=30, value=12, step=1
            )
            macd_signal = st.number_input(
                "Signal Length", min_value=3, max_value=15, value=9, step=1
            )
        with macd_col2:
            macd_slow = st.number_input(
                "Slow Length", min_value=10, max_value=50, value=26, step=1
            )

    use_bbands = st.checkbox("Bollinger Bands", value=True)
    bb_length, bb_std = 20, 2.0
    if use_bbands:
        bb_col1, bb_col2 = st.columns(2)
        with bb_col1:
            bb_length = st.number_input("Length", min_value=5, max_value=50, value=20, step=1)
        with bb_col2:
            bb_std = st.number_input(
                "Standard Deviation", min_value=1.0, max_value=4.0, value=2.0, step=0.1
            )

    st.markdown("##### Additional Indicators")
    use_stoch = st.checkbox("Stochastic Oscillator", value=False)
    stoch_k, stoch_d = 14, 3
    if use_stoch:
        stoch_col1, stoch_col2 = st.columns(2)
        with stoch_col1:
            stoch_k = st.number_input("K Length", min_value=5, max_value=30, value=14, step=1)
        with stoch_col2:
            stoch_d = st.number_input("D Length", min_value=1, max_value=10, value=3, step=1)

    use_adx = st.checkbox("Average Directional Index (ADX)", value=False)
    adx_length = st.slider(
        "ADX Length", min_value=5, max_value=30, value=14, step=1, disabled=not use_adx
    )

    use_willr = st.checkbox("Williams %R", value=False)
    willr_length = st.slider(
        "Williams %R Length",
        min_value=5,
        max_value=30,
        value=14,
        step=1,
        disabled=not use_willr,
    )

    st.subheader("Visualization Settings")
    use_candlestick = st.checkbox("Use Candlestick Chart", value=True)
    fig_width = st.slider("Figure Width", min_value=8, max_value=20, value=12, step=1)
    fig_height = st.slider("Figure Height (per subplot)", min_value=3, max_value=8, value=4, step=1)
    
    # Real-time update settings
    st.subheader("Real-time Updates")
    enable_real_time = st.checkbox("Enable Real-time Updates", value=False)
    update_interval = 60  # Default value
    if enable_real_time:
        update_interval = st.slider("Update Interval (seconds)", min_value=10, max_value=300, value=60, step=10)

indicator_configs = prepare_indicator_configs(
    use_sma,
    sma_short_length if use_sma else 20,
    sma_long_length if use_sma else 50,
    use_ema,
    ema_short_length if use_ema else 12,
    ema_long_length if use_ema else 26,
    use_rsi,
    rsi_length,
    use_macd,
    macd_fast,
    macd_slow,
    macd_signal,
    use_bbands,
    bb_length,
    bb_std,
    use_stoch,
    stoch_k,
    stoch_d,
    use_adx,
    adx_length,
    use_willr,
    willr_length,
)

# Process symbols
multi_symbols = parse_symbol_list(multi_symbols_raw, symbol)
if symbol:
    comparison_symbols = [symbol] + [s for s in multi_symbols if s != symbol]
else:
    comparison_symbols = multi_symbols

# Analysis button
analyze_button = st.button("Run Analysis", type="primary", key="analyze")

# Real-time update placeholder
real_time_placeholder = st.empty()

if analyze_button or (enable_real_time and 'last_update' not in st.session_state):
    if len(comparison_symbols) < 1:
        st.warning("Please provide at least one ticker symbol to run the analysis.")
    else:
        with st.spinner("Fetching data for selected symbols..."):
            data_fetcher = DataFetcher(use_vectorbt=False)
            data = data_fetcher.fetch_data(
                comparison_symbols, period=period_value, interval=interval_value
            )

        available_data = {
            sym: df
            for sym, df in data.items()
            if df is not None and not df.empty
        }
        missing_symbols = [sym for sym in comparison_symbols if sym not in available_data]

        if not available_data:
            st.error(
                "None of the requested symbols returned data. Please verify the tickers and try again."
            )
        else:
            st.session_state.multi_data = available_data
            st.session_state.multi_symbols = list(available_data.keys())
            st.session_state.multi_companies = {
                sym: data_fetcher.get_company_name(sym) for sym in available_data
            }
            st.session_state.multi_missing = missing_symbols
            st.session_state.last_update = datetime.now()

# Real-time updates
if enable_real_time and 'multi_data' in st.session_state:
    # Check if it's time to update
    if 'last_update' not in st.session_state or \
       (datetime.now() - st.session_state.last_update).seconds > update_interval:
        
        with real_time_placeholder.container():
            st.info("Fetching real-time data...")
            # Update the primary symbol with real-time data
            if symbol in st.session_state.multi_data:
                real_time_data = fetch_real_time_data(symbol, period="1d", interval="1m")
                if not real_time_data.empty:
                    # Combine historical and real-time data
                    combined_data = pd.concat([st.session_state.multi_data[symbol], real_time_data])
                    # Remove duplicates, keeping the latest
                    combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                    st.session_state.multi_data[symbol] = combined_data
                    st.session_state.last_update = datetime.now()
                    st.success("Real-time data updated!")
                else:
                    st.warning("Could not fetch real-time data.")
            else:
                st.warning("Primary symbol not found in data.")

# Display results if data is available
multi_data = st.session_state.get("multi_data", {})

if multi_data:
    if st.session_state.get("multi_missing"):
        missing_display = ", ".join(st.session_state["multi_missing"])
        st.warning(f"Data could not be retrieved for: {missing_display}")

    ta = TechnicalAnalysis()
    visualizer = Visualizer(figsize=(fig_width, fig_height))

    summary_rows = []
    indicator_frames = {}

    for sym, df in multi_data.items():
        df_with_indicators = ta.calculate_indicators(df.copy(), indicator_configs)
        indicator_frames[sym] = df_with_indicators

        last_close, period_return, volatility = summarize_performance(df_with_indicators)
        summary_rows.append(
            {
                "Symbol": sym,
                "Company": st.session_state.multi_companies.get(sym, sym),
                "Last Close": last_close,
                "Period Return (%)": period_return * 100 if not np.isnan(period_return) else np.nan,
                "Annualized Volatility (%)": volatility * 100 if not np.isnan(volatility) else np.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.set_index("Symbol")
    st.markdown("#### Performance Snapshot")
    st.dataframe(
        summary_df.style.format(
            {
                "Last Close": "${:,.2f}",
                "Period Return (%)": "{:.2f}%",
                "Annualized Volatility (%)": "{:.2f}%",
            }
        )
    )

    st.markdown("#### Indicator Views")
    
    # Create tabs for each symbol
    if len(indicator_frames) > 1:
        symbol_tabs = st.tabs([sym for sym in indicator_frames.keys()])
        tab_iter = zip(indicator_frames.items(), symbol_tabs)
    else:
        # If only one symbol, don't use tabs
        tab_iter = [(list(indicator_frames.items())[0], None)]
    
    for (sym, df_with_indicators), symbol_tab in tab_iter:
        with (symbol_tab or st.container()):
            company_name = st.session_state.multi_companies.get(sym, sym)
            st.markdown(f"**{sym} - {company_name}**")

            # Create and display the plot
            with st.spinner("Creating visualization..."):
                fig = visualizer.create_plot_figure(
                    df_with_indicators,
                    sym,
                    indicator_configs,
                    company_name=company_name,
                    use_candlestick=use_candlestick
                )

            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Plot generation failed for this symbol.")

            # Data and download section
            with st.expander("Show indicator data"):
                st.dataframe(df_with_indicators)

            col1, col2 = st.columns(2)
            with col1:
                csv = df_with_indicators.to_csv().encode("utf-8")
                st.download_button(
                    label=f"Download {sym} CSV",
                    data=csv,
                    file_name=f"{sym}_technical_analysis.csv",
                    mime="text/csv",
                    key=f"download_{sym}_csv",
                )
            with col2:
                if fig:
                    # Convert Plotly figure to HTML for download
                    html_string = fig.to_html(include_plotlyjs='cdn')
                    st.download_button(
                        label=f"Download {sym} Chart",
                        data=html_string,
                        file_name=f"{sym}_technical_analysis.html",
                        mime="text/html",
                        key=f"download_{sym}_chart",
                    )
else:
    st.info(
        "Enter stock symbols in the sidebar and click 'Run Analysis' to generate technical insights."
    )

# --- Backtesting ---
with st.expander("Backtest Trading Strategies", expanded=False):
    st.subheader("Backtest Trading Strategies")
    st.caption(
        "Evaluate rule-based strategies using the primary symbol's historical data and review performance metrics."
    )

    if st.session_state.get("multi_data") is None or symbol not in st.session_state.get("multi_data", {}):
        st.info("Run an analysis first to load price history for backtesting.")
    else:
        strategy_type = st.selectbox(
            "Strategy Type",
            ["Moving Average Crossover", "RSI Strategy"],
            key="strategy_type",
        )

        # Initialize variables with default values
        ma_short = 20
        ma_long = 50
        rsi_bt_length = 14
        rsi_oversold = 30
        rsi_overbought = 70

        if strategy_type == "Moving Average Crossover":
            ma_short = st.slider(
                "Short MA Length", min_value=5, max_value=50, value=20, step=1, key="ma_short"
            )
            ma_long = st.slider(
                "Long MA Length", min_value=10, max_value=200, value=50, step=5, key="ma_long"
            )
        else:
            rsi_bt_length = st.slider(
                "RSI Length", min_value=5, max_value=30, value=14, step=1, key="bt_rsi_length"
            )
            rsi_oversold = st.slider(
                "Oversold Level", min_value=10, max_value=40, value=30, step=1, key="rsi_oversold"
            )
            rsi_overbought = st.slider(
                "Overbought Level", min_value=60, max_value=90, value=70, step=1, key="rsi_overbought"
            )

        initial_capital = st.number_input(
            "Initial Capital",
            min_value=1000,
            max_value=1_000_000,
            value=10_000,
            step=1000,
            key="initial_capital",
        )
        commission_pct = st.number_input(
            "Commission (%)",
            min_value=0.0,
            max_value=2.0,
            value=0.1,
            step=0.05,
            key="commission_pct",
        )
        commission = commission_pct / 100

        run_backtest = st.button("Run Backtest", type="primary", key="run_backtest_btn")

        if run_backtest:
            from src.backtesting.engine import BacktestEngine
            from src.backtesting.strategy import (
                MovingAverageCrossoverStrategy,
                RSIStrategy,
            )

            engine = BacktestEngine(use_vectorbt=False)

            if strategy_type == "Moving Average Crossover":
                strategy = MovingAverageCrossoverStrategy(
                    short_window=ma_short,
                    long_window=ma_long,
                    name=f"MA_{ma_short}_{ma_long}",
                )
                strategy_label = f"Moving Average Crossover (Short: {ma_short}, Long: {ma_long})"
            else:
                strategy = RSIStrategy(
                    rsi_period=rsi_bt_length,
                    oversold=rsi_oversold,
                    overbought=rsi_overbought,
                    name=f"RSI_{rsi_bt_length}_{rsi_oversold}_{rsi_overbought}",
                )
                strategy_label = (
                    f"RSI Strategy (Period: {rsi_bt_length}, Oversold: {rsi_oversold}, Overbought: {rsi_overbought})"
                )

            with st.spinner("Running backtest..."):
                result = engine.run_backtest(
                    strategy=strategy,
                    data=st.session_state.multi_data[symbol],
                    initial_capital=initial_capital,
                    commission=commission,
                )

            st.session_state.backtest_result = {
                "result": result,
                "strategy": strategy,
                "strategy_label": strategy_label,
            }

        if st.session_state.get("backtest_result"):
            stored_symbol = symbol
            company_name = st.session_state.multi_companies.get(symbol, symbol) if 'multi_companies' in st.session_state else stored_symbol
            result_payload = st.session_state.backtest_result
            result = result_payload["result"]
            strategy = result_payload["strategy"]
            strategy_label = result_payload["strategy_label"]

            st.markdown(f"### Backtest Results: {stored_symbol} - {company_name}")
            st.write(f"Strategy: {strategy_label}")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{result['total_return']:.2%}")
            with col2:
                st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
            with col3:
                st.metric("Max Drawdown", f"{result['max_drawdown']:.2%}")
            with col4:
                if result["win_rate"] is not None:
                    st.metric("Win Rate", f"{result['win_rate']:.2%}")
                else:
                    st.metric("Win Rate", "N/A")

            fig = result_payload.get("figure")
            if fig is None:
                from src.backtesting.engine import BacktestEngine

                engine = BacktestEngine(use_vectorbt=False)
                fig = engine.visualize_results(result, stored_symbol, strategy.name)
                result_payload["figure"] = fig

            st.pyplot(fig)

            st.subheader("Trade Details")
            trades = result.get("trades")
            if hasattr(trades, "records") and len(trades.records) > 0:
                trades_df = trades.records
                st.dataframe(trades_df)
            else:
                st.info("No trades were executed during the backtest period.")

            st.subheader("Signal Data")
            st.dataframe(result["signals"])

            csv = result["signals"].to_csv().encode("utf-8")
            st.download_button(
                label=f"Download {strategy.name} Signals",
                data=csv,
                file_name=f"{stored_symbol}_{strategy.name}_signals.csv",
                mime="text/csv",
            )

st.markdown("---")
st.markdown(
    """
    <div style=\"text-align: center;\">
        <p>Developed with ❤️ using Streamlit | Data source: Yahoo Finance</p>
    </div>
    """,
    unsafe_allow_html=True,
)
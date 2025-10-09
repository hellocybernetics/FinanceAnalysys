"""
Streamlit app for technical analysis of financial data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from datetime import datetime, timedelta
import os

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
    sma_length,
    use_ema,
    ema_length,
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
        indicators.append({"name": "SMA", "params": {"length": sma_length}})

    if use_ema:
        indicators.append({"name": "EMA", "params": {"length": ema_length}})

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
    unique_symbols = []
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


col1, col2 = st.columns([5, 1])
with col1:
    st.title("📊 Financial Technical Analysis")
with col2:
    theme_icon = "🌙" if st.session_state.theme == "light" else "☀️"
    st.button(theme_icon, on_click=toggle_theme, key="theme_toggle")

st.markdown("Analyze single or multiple stocks, visualize indicators, and backtest trading ideas.")

tab_single, tab_multi, tab_backtest = st.tabs(
    ["Single Symbol Analysis", "Multi-Symbol Dashboard", "Backtest"]
)

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
        "1 Day": "1d",
        "1 Week": "1wk",
        "1 Month": "1mo",
    }
    interval = st.selectbox("Select Interval", options=list(interval_options.keys()), index=0)
    interval_value = interval_options[interval]

    st.subheader("Technical Indicators")

    st.markdown("##### Moving Averages")
    use_sma = st.checkbox("Simple Moving Average (SMA)", value=True)
    sma_length = st.slider(
        "SMA Length", min_value=5, max_value=200, value=20, step=5, disabled=not use_sma
    )

    use_ema = st.checkbox("Exponential Moving Average (EMA)", value=True)
    ema_length = st.slider(
        "EMA Length", min_value=5, max_value=200, value=50, step=5, disabled=not use_ema
    )

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

    use_bbands = st.checkbox("Bollinger Bands", value=False)
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
    fig_width = st.slider("Figure Width", min_value=8, max_value=20, value=12, step=1)
    fig_height = st.slider("Figure Height", min_value=6, max_value=16, value=8, step=1)

indicator_configs = prepare_indicator_configs(
    use_sma,
    sma_length,
    use_ema,
    ema_length,
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

# --- Single Symbol Analysis ---
with tab_single:
    st.subheader("Single Symbol Technical Analysis")
    st.caption("Run a full indicator suite for the primary symbol selected in the sidebar.")

    analyze_single = st.button("Run Single Analysis", type="primary", key="analyze_single")

    if analyze_single:
        if not symbol:
            st.warning("Please provide a valid primary stock symbol before running the analysis.")
        else:
            with st.spinner(f"Fetching data for {symbol}..."):
                data_fetcher = DataFetcher(use_vectorbt=False)
                data = data_fetcher.fetch_data([symbol], period=period_value, interval=interval_value)

            if symbol in data and not data[symbol].empty:
                st.session_state.single_data = data[symbol]
                st.session_state.single_symbol = symbol
                st.session_state.single_company = data_fetcher.get_company_name(symbol)
                st.session_state.pop("backtest_result", None)
            else:
                st.error(
                    f"Failed to fetch data for {symbol}. Please check the symbol and try again."
                )

    if st.session_state.get("single_data") is not None:
        stored_symbol = st.session_state.get("single_symbol")
        company_name = st.session_state.get("single_company", stored_symbol)

        if symbol and stored_symbol and symbol != stored_symbol:
            st.info(
                f"Showing cached results for {stored_symbol}. Update the symbol and rerun the analysis to refresh."
            )

        with st.spinner("Calculating technical indicators..."):
            ta = TechnicalAnalysis()
            df_with_indicators = ta.calculate_indicators(
                st.session_state.single_data.copy(), indicator_configs
            )
            st.session_state.single_df = df_with_indicators

        with st.spinner("Creating visualization..."):
            visualizer = Visualizer(figsize=(fig_width, fig_height))
            fig = visualizer.create_plot_figure(
                df_with_indicators,
                stored_symbol,
                indicator_configs,
                company_name=company_name,
            )

            buf = None
            if fig:
                buf = visualizer.save_figure_to_buffer(fig, format="png")

        st.markdown(f"### {stored_symbol} - {company_name}")

        if buf:
            st.image(buf, use_container_width=True)
        else:
            st.warning(
                "Could not generate the plot. Please review the selected indicators or try again."
            )

        with st.expander("View Indicator Data"):
            st.dataframe(df_with_indicators)

        download_col1, download_col2 = st.columns(2)
        with download_col1:
            csv = df_with_indicators.to_csv().encode("utf-8")
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{stored_symbol}_technical_analysis.csv",
                mime="text/csv",
            )
        with download_col2:
            if buf:
                st.download_button(
                    label="Download Chart",
                    data=buf,
                    file_name=f"{stored_symbol}_technical_analysis.png",
                    mime="image/png",
                )
    else:
        st.info(
            "Enter a primary stock symbol and click 'Run Single Analysis' to generate technical insights."
        )

# --- Multi Symbol Analysis ---
with tab_multi:
    st.subheader("Multi-Symbol Dashboard")
    st.caption(
        "Compare the primary symbol with additional tickers, visualize indicators, and review key metrics side-by-side."
    )

    multi_symbols = parse_symbol_list(multi_symbols_raw, symbol)
    if symbol:
        comparison_symbols = [symbol] + [s for s in multi_symbols if s != symbol]
    else:
        comparison_symbols = multi_symbols

    analyze_multi = st.button("Run Multi-Symbol Analysis", key="analyze_multi")

    if analyze_multi:
        if len(comparison_symbols) < 2:
            st.warning(
                "Provide at least two distinct ticker symbols (including the primary symbol) to run the dashboard."
            )
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
        symbol_tabs = st.tabs([sym for sym in indicator_frames.keys()])

        for (sym, df_with_indicators), symbol_tab in zip(
            indicator_frames.items(), symbol_tabs
        ):
            with symbol_tab:
                company_name = st.session_state.multi_companies.get(sym, sym)
                st.markdown(f"**{sym} - {company_name}**")

                fig = visualizer.create_plot_figure(
                    df_with_indicators,
                    sym,
                    indicator_configs,
                    company_name=company_name,
                )

                buf = None
                if fig:
                    buf = visualizer.save_figure_to_buffer(fig, format="png")

                if buf:
                    st.image(buf, use_container_width=True)
                    st.download_button(
                        label=f"Download {sym} Chart",
                        data=buf,
                        file_name=f"{sym}_technical_analysis.png",
                        mime="image/png",
                        key=f"download_{sym}_chart",
                    )
                else:
                    st.warning("Plot generation failed for this symbol.")

                with st.expander("Show indicator data"):
                    st.dataframe(df_with_indicators)

                csv = df_with_indicators.to_csv().encode("utf-8")
                st.download_button(
                    label=f"Download {sym} CSV",
                    data=csv,
                    file_name=f"{sym}_technical_analysis.csv",
                    mime="text/csv",
                    key=f"download_{sym}_csv",
                )
    else:
        st.info(
            "Add comparison symbols in the sidebar and click 'Run Multi-Symbol Analysis' to build the dashboard."
        )

# --- Backtesting ---
with tab_backtest:
    st.subheader("Backtest Trading Strategies")
    st.caption(
        "Evaluate rule-based strategies using the primary symbol's historical data and review performance metrics."
    )

    if st.session_state.get("single_data") is None:
        st.info("Run a single-symbol analysis first to load price history for backtesting.")
    else:
        strategy_type = st.selectbox(
            "Strategy Type",
            ["Moving Average Crossover", "RSI Strategy"],
            key="strategy_type",
        )

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
                    data=st.session_state.single_data,
                    initial_capital=initial_capital,
                    commission=commission,
                )

            st.session_state.backtest_result = {
                "result": result,
                "strategy": strategy,
                "strategy_label": strategy_label,
            }

        if st.session_state.get("backtest_result"):
            stored_symbol = st.session_state.get("single_symbol", symbol)
            company_name = st.session_state.get("single_company", stored_symbol)
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

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

col1, col2 = st.columns([5, 1])
with col1:
    st.title("📊 Financial Technical Analysis")
with col2:
    theme_icon = "🌙" if st.session_state.theme == "light" else "☀️"
    st.button(theme_icon, on_click=toggle_theme, key="theme_toggle")

st.markdown("Analyze stock price data with technical indicators and visualize the results.")

tab1, tab2 = st.tabs(["Technical Analysis", "Backtest"])

with st.sidebar:
    st.header("Configuration")
    
    symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter a valid stock symbol (e.g., AAPL, MSFT, GOOGL)")
    
    st.subheader("Time Period")
    period_options = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
        "Max": "max"
    }
    period = st.selectbox("Select Period", options=list(period_options.keys()), index=3)
    period_value = period_options[period]
    
    interval_options = {
        "1 Day": "1d",
        "1 Week": "1wk",
        "1 Month": "1mo"
    }
    interval = st.selectbox("Select Interval", options=list(interval_options.keys()), index=0)
    interval_value = interval_options[interval]
    
    st.subheader("Technical Indicators")
    
    st.markdown("##### Moving Averages")
    use_sma = st.checkbox("Simple Moving Average (SMA)", value=True)
    sma_length = st.slider("SMA Length", min_value=5, max_value=200, value=20, step=5, disabled=not use_sma)
    
    use_ema = st.checkbox("Exponential Moving Average (EMA)", value=True)
    ema_length = st.slider("EMA Length", min_value=5, max_value=200, value=50, step=5, disabled=not use_ema)
    
    st.markdown("##### Oscillators")
    use_rsi = st.checkbox("Relative Strength Index (RSI)", value=True)
    rsi_length = st.slider("RSI Length", min_value=5, max_value=30, value=14, step=1, disabled=not use_rsi)
    
    use_macd = st.checkbox("MACD", value=False)
    if use_macd:
        macd_col1, macd_col2 = st.columns(2)
        with macd_col1:
            macd_fast = st.number_input("Fast Length", min_value=5, max_value=30, value=12, step=1)
            macd_signal = st.number_input("Signal Length", min_value=3, max_value=15, value=9, step=1)
        with macd_col2:
            macd_slow = st.number_input("Slow Length", min_value=10, max_value=50, value=26, step=1)
    
    use_bbands = st.checkbox("Bollinger Bands", value=False)
    if use_bbands:
        bb_col1, bb_col2 = st.columns(2)
        with bb_col1:
            bb_length = st.number_input("Length", min_value=5, max_value=50, value=20, step=1)
        with bb_col2:
            bb_std = st.number_input("Standard Deviation", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
    
    st.markdown("##### Additional Indicators")
    use_stoch = st.checkbox("Stochastic Oscillator", value=False)
    if use_stoch:
        stoch_col1, stoch_col2 = st.columns(2)
        with stoch_col1:
            stoch_k = st.number_input("K Length", min_value=5, max_value=30, value=14, step=1)
        with stoch_col2:
            stoch_d = st.number_input("D Length", min_value=1, max_value=10, value=3, step=1)
    
    use_adx = st.checkbox("Average Directional Index (ADX)", value=False)
    adx_length = st.slider("ADX Length", min_value=5, max_value=30, value=14, step=1, disabled=not use_adx)
    
    use_willr = st.checkbox("Williams %R", value=False)
    willr_length = st.slider("Williams %R Length", min_value=5, max_value=30, value=14, step=1, disabled=not use_willr)
    
    st.subheader("Visualization Settings")
    fig_width = st.slider("Figure Width", min_value=8, max_value=20, value=12, step=1)
    fig_height = st.slider("Figure Height", min_value=6, max_value=16, value=8, step=1)
    
    st.subheader("Backtest Settings")
    run_backtest = st.checkbox("Enable Backtest", value=False)
    
    if run_backtest:
        st.markdown("##### Strategy Selection")
        strategy_type = st.selectbox(
            "Strategy Type", 
            ["Moving Average Crossover", "RSI Strategy"],
            index=0
        )
        
        if strategy_type == "Moving Average Crossover":
            ma_short = st.slider("Short MA Length", min_value=5, max_value=50, value=20, step=1)
            ma_long = st.slider("Long MA Length", min_value=10, max_value=200, value=50, step=5)
        elif strategy_type == "RSI Strategy":
            rsi_length = st.slider("RSI Length", min_value=5, max_value=30, value=14, step=1)
            rsi_oversold = st.slider("Oversold Level", min_value=10, max_value=40, value=30, step=1)
            rsi_overbought = st.slider("Overbought Level", min_value=60, max_value=90, value=70, step=1)
        
        st.markdown("##### Backtest Parameters")
        initial_capital = st.number_input("Initial Capital", min_value=1000, max_value=1000000, value=10000, step=1000)
        commission = st.number_input("Commission (%)", min_value=0.0, max_value=2.0, value=0.1, step=0.05) / 100
    
    analyze_button = st.button("Analyze", type="primary", use_container_width=True)

if analyze_button or 'data' in st.session_state:
    with st.spinner(f"Fetching data for {symbol}..."):
        data_fetcher = DataFetcher(use_vectorbt=False)
        
        data = data_fetcher.fetch_data([symbol], period=period_value, interval=interval_value)
        
        if symbol in data and not data[symbol].empty:
            st.session_state.data = data[symbol]
            st.session_state.symbol = symbol
            st.session_state.company_name = data_fetcher.get_company_name(symbol)
        else:
            st.error(f"Failed to fetch data for {symbol}. Please check the symbol and try again.")
            st.stop()
    
    with tab1:
        st.header(f"{st.session_state.symbol} - {st.session_state.company_name}")
        
        indicators = []
    
    if use_sma:
        indicators.append({"name": "SMA", "params": {"length": sma_length}})
    
    if use_ema:
        indicators.append({"name": "EMA", "params": {"length": ema_length}})
    
    if use_rsi:
        indicators.append({"name": "RSI", "params": {"length": rsi_length}})
    
    if use_macd:
        indicators.append({"name": "MACD", "params": {"fast": macd_fast, "slow": macd_slow, "signal": macd_signal}})
    
    if use_bbands:
        indicators.append({"name": "BBands", "params": {"length": bb_length, "std": bb_std}})
    
    if use_stoch:
        indicators.append({"name": "Stochastic", "params": {"k": stoch_k, "d": stoch_d}})
    
    if use_adx:
        indicators.append({"name": "ADX", "params": {"length": adx_length}})
    
    if use_willr:
        indicators.append({"name": "WILLR", "params": {"length": willr_length}})
    
    with st.spinner("Calculating technical indicators..."):
        ta = TechnicalAnalysis()
        df_with_indicators = ta.calculate_indicators(st.session_state.data, indicators)
    
    with st.spinner("Creating visualization..."):
        visualizer = Visualizer(figsize=(fig_width, fig_height))
        
        # Create the figure object
        fig = visualizer.create_plot_figure(
            df_with_indicators, 
            st.session_state.symbol, 
            indicators, 
            company_name=st.session_state.company_name
        )
        
        buf = None
        if fig:
            # Save the figure to a buffer
            buf = visualizer.save_figure_to_buffer(fig, format='png')
        
        if buf:
            st.image(buf, use_container_width=True)
        else:
            st.warning("Could not generate the plot. Please check the logs or selected indicators.")
    
    with st.expander("View Data Table"):
        st.dataframe(df_with_indicators)
    
    col1, col2 = st.columns(2)
    with col1:
        csv = df_with_indicators.to_csv().encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{symbol}_technical_analysis.csv",
            mime="text/csv",
        )
    with col2:
        if buf: # Only show download button if buffer exists
            st.download_button(
                label="Download Chart",
                data=buf,
                file_name=f"{symbol}_technical_analysis.png",
                mime="image/png",
            )

    if run_backtest:
        with tab2:
            st.header(f"Backtest Results: {st.session_state.symbol} - {st.session_state.company_name}")
            
            from src.backtesting.engine import BacktestEngine
            from src.backtesting.strategy import MovingAverageCrossoverStrategy, RSIStrategy
            
            engine = BacktestEngine(use_vectorbt=False)
            
            if strategy_type == "Moving Average Crossover":
                strategy = MovingAverageCrossoverStrategy(
                    short_window=ma_short,
                    long_window=ma_long,
                    name=f"MA_{ma_short}_{ma_long}"
                )
                st.write(f"Strategy: Moving Average Crossover (Short: {ma_short}, Long: {ma_long})")
            else:  # RSI Strategy
                strategy = RSIStrategy(
                    rsi_period=rsi_length,
                    oversold=rsi_oversold,
                    overbought=rsi_overbought,
                    name=f"RSI_{rsi_length}_{rsi_oversold}_{rsi_overbought}"
                )
                st.write(f"Strategy: RSI (Period: {rsi_length}, Oversold: {rsi_oversold}, Overbought: {rsi_overbought})")
            
            with st.spinner("Running backtest..."):
                result = engine.run_backtest(
                    strategy=strategy,
                    data=st.session_state.data,
                    initial_capital=initial_capital,
                    commission=commission
                )
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{result['total_return']:.2%}")
            with col2:
                st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
            with col3:
                st.metric("Max Drawdown", f"{result['max_drawdown']:.2%}")
            with col4:
                if result['win_rate'] is not None:
                    st.metric("Win Rate", f"{result['win_rate']:.2%}")
                else:
                    st.metric("Win Rate", "N/A")
            
            st.subheader("Backtest Chart")
            fig = engine.visualize_results(result, st.session_state.symbol, strategy.name)
            st.pyplot(fig)
            
            st.subheader("Trade Details")
            if hasattr(result['trades'], 'records') and len(result['trades'].records) > 0:
                trades_df = result['trades'].records
                st.dataframe(trades_df)
            else:
                st.info("No trades were executed during the backtest period.")
            
            st.subheader("Signal Data")
            st.dataframe(result['signals'])
            
            csv = result['signals'].to_csv().encode('utf-8')
            st.download_button(
                label="Download Signals CSV",
                data=csv,
                file_name=f"{symbol}_{strategy.name}_signals.csv",
                mime="text/csv",
            )

else:
    st.info("Enter a stock symbol and select technical indicators in the sidebar, then click 'Analyze' to generate the analysis.")
    
    st.image("https://miro.medium.com/max/1400/1*b1nQQH6zQCiAEEMgYwCxQg.png", caption="Sample Technical Analysis Chart")

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p>Developed with ❤️ using Streamlit | Data source: Yahoo Finance</p>
    </div>
    """,
    unsafe_allow_html=True
)

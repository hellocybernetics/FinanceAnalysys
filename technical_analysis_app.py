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
            # Use a different key for the backtest RSI length to avoid conflict with main chart RSI
            backtest_rsi_length_val = st.slider("RSI Length", min_value=5, max_value=30, value=st.session_state.get('backtest_rsi_length_val', 14), step=1, key="backtest_rsi_length")
            rsi_oversold = st.slider("Oversold Level", min_value=10, max_value=40, value=30, step=1)
            rsi_overbought = st.slider("Overbought Level", min_value=60, max_value=90, value=70, step=1)
            # Store for potential use if main chart RSI length differs
            st.session_state.backtest_rsi_length_val = backtest_rsi_length_val
            st.session_state.rsi_oversold_val = rsi_oversold
            st.session_state.rsi_overbought_val = rsi_overbought

        st.markdown("---") # Separator
        st.markdown("##### Build Custom Strategy")
        # Ensure use_custom_strategy is initialized in session_state if not present
        if 'use_custom_strategy' not in st.session_state:
            st.session_state.use_custom_strategy = False
        
        # Update session state based on checkbox
        st.session_state.use_custom_strategy = st.checkbox(
            "Enable Custom Strategy Builder", 
            value=st.session_state.use_custom_strategy, 
            key="use_custom_strategy_cb"
        )

        # Initialize or retrieve custom_strategy_config from session_state
        if 'custom_strategy_config' not in st.session_state:
            st.session_state.custom_strategy_config = {
                "name": "MyCustomStrategy",
                "selected_indicators": [],
                "indicators_params": {},
                "indicators": [],
                "buy_condition": "SMA_20 > Close AND RSI_14 < 30",
                "sell_condition": "SMA_20 < Close OR RSI_14 > 70"
            }

        if st.session_state.use_custom_strategy:
            custom_strategy_name = st.text_input(
                "Custom Strategy Name", 
                value=st.session_state.custom_strategy_config.get("name", "MyCustomStrategy"),
                help="Enter a name for your custom strategy."
            )
            st.session_state.custom_strategy_config['name'] = custom_strategy_name

            st.markdown("###### Indicators for Custom Strategy")
            available_indicators = {
                "SMA": {"length": 20}, "EMA": {"length": 50}, "RSI": {"length": 14},
                "MACD": {"fast": 12, "slow": 26, "signal": 9},
                "BBands": {"length": 20, "std": 2}, "ATR": {"length": 14},
                "Stochastic": {"k": 14, "d": 3}, "ADX": {"length": 14}, "WILLR": {"length": 14}
            }

            selected_indicator_types = st.multiselect(
                "Select Indicators", options=list(available_indicators.keys()),
                default=st.session_state.custom_strategy_config.get("selected_indicators", []),
                help="Choose indicators to use in your strategy.", key="custom_indicator_multiselect"
            )
            st.session_state.custom_strategy_config["selected_indicators"] = selected_indicator_types

            custom_indicators_params = []
            # Ensure indicators_params is a dict in session state
            if 'indicators_params' not in st.session_state.custom_strategy_config or not isinstance(st.session_state.custom_strategy_config['indicators_params'], dict):
                st.session_state.custom_strategy_config['indicators_params'] = {}

            for ind_type in selected_indicator_types:
                st.markdown(f"**Parameters for {ind_type}**")
                # Get current params for this ind_type from session_state, or default if not found
                current_params_for_ind = st.session_state.custom_strategy_config['indicators_params'].get(ind_type, available_indicators[ind_type])
                
                params = {} # To store the updated params from widgets
                if ind_type in ["SMA", "EMA", "RSI", "ATR", "ADX", "WILLR"]:
                    params['length'] = st.slider(f"{ind_type} Length", 1, 200, current_params_for_ind.get('length', available_indicators[ind_type]['length']), key=f"custom_{ind_type}_length")
                elif ind_type == "MACD":
                    params['fast'] = st.slider(f"{ind_type} Fast Length", 1, 50, current_params_for_ind.get('fast',available_indicators[ind_type]['fast']), key=f"custom_{ind_type}_fast")
                    params['slow'] = st.slider(f"{ind_type} Slow Length", 1, 100, current_params_for_ind.get('slow',available_indicators[ind_type]['slow']), key=f"custom_{ind_type}_slow")
                    params['signal'] = st.slider(f"{ind_type} Signal Length", 1, 50, current_params_for_ind.get('signal',available_indicators[ind_type]['signal']), key=f"custom_{ind_type}_signal")
                elif ind_type == "BBands":
                    params['length'] = st.slider(f"{ind_type} BBands Length", 1, 100, current_params_for_ind.get('length',available_indicators[ind_type]['length']), key=f"custom_{ind_type}_bband_length") 
                    params['std'] = st.slider(f"{ind_type} Std Dev", 1.0, 4.0, float(current_params_for_ind.get('std',available_indicators[ind_type]['std'])), step=0.1, key=f"custom_{ind_type}_std")
                elif ind_type == "Stochastic":
                    params['k'] = st.slider(f"{ind_type} %K Length", 1, 50, current_params_for_ind.get('k',available_indicators[ind_type]['k']), key=f"custom_{ind_type}_k")
                    params['d'] = st.slider(f"{ind_type} %D Length", 1, 50, current_params_for_ind.get('d',available_indicators[ind_type]['d']), key=f"custom_{ind_type}_d")
                
                custom_indicators_params.append({'name': ind_type, 'params': params})
                # Store/update individual indicator params in session state
                st.session_state.custom_strategy_config['indicators_params'][ind_type] = params
            
            st.session_state.custom_strategy_config['indicators'] = custom_indicators_params

            st.markdown("###### Custom Strategy Rules")
            buy_condition_help_text = """
Define buy rule using OHLCV data and calculated indicators.
Available OHLCV columns: Open, High, Low, Close, Volume.
Indicator columns are named based on type and parameters, e.g.:
- SMA_20, EMA_50 (replace numbers with your chosen lengths)
- RSI_14
- MACD_12_26 (MACD line), MACD_Signal_9 (Signal line), MACD_Hist_12_26_9
- BBU_20_2 (Upper BB), BBM_20_2 (Middle BB), BBL_20_2 (Lower BB)
- ATR_14, STOCHk_14_3, STOCHd_14_3, ADX_14, WILLR_14

Use standard comparison (>, <, ==, etc.) and logical operators (AND, OR, NOT).
Example: (Close > SMA_50) AND (MACD_12_26 > MACD_Signal_9) AND (RSI_14 < 30)
Ensure indicator parameters in rules match your selections above.
            """
            buy_condition = st.text_area(
                "Buy Condition", 
                value=st.session_state.custom_strategy_config.get("buy_condition", "SMA_20 > Close AND RSI_14 < 30"),
                height=100, 
                help=buy_condition_help_text, 
                key="custom_buy_condition"
            )
            st.session_state.custom_strategy_config['buy_condition'] = buy_condition

            sell_condition_help_text = """
Define sell rule using OHLCV data and calculated indicators.
Available OHLCV columns: Open, High, Low, Close, Volume.
Indicator columns are named based on type and parameters (see examples in Buy Condition help).
Example: (Close < SMA_50) OR (RSI_14 > 70) OR (Close < BBL_20_2)
Use standard comparison (>, <, ==, etc.) and logical operators (AND, OR, NOT).
Ensure indicator parameters in rules match your selections above.
            """
            sell_condition = st.text_area(
                "Sell Condition", 
                value=st.session_state.custom_strategy_config.get("sell_condition", "SMA_20 < Close OR RSI_14 > 70"),
                height=100, 
                help=sell_condition_help_text, 
                key="custom_sell_condition"
            )
            st.session_state.custom_strategy_config['sell_condition'] = sell_condition
        
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
        # Use the main chart RSI length unless backtest RSI strategy is active and has a different value
        current_rsi_length = rsi_length 
        if run_backtest and strategy_type == 'RSI Strategy' and 'backtest_rsi_length_val' in st.session_state:
            current_rsi_length = st.session_state.backtest_rsi_length_val
        indicators.append({"name": "RSI", "params": {"length": current_rsi_length}})
    
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
            from src.backtesting.strategy import MovingAverageCrossoverStrategy, RSIStrategy, GenericUserStrategy # Added GenericUserStrategy
            
            engine = BacktestEngine(use_vectorbt=False)
            strategies_to_run = []

            # Prepare predefined strategy (if selected and backtesting is enabled)
            # Note: strategy_type, ma_short, ma_long are available from the sidebar scope
            if strategy_type == "Moving Average Crossover":
                predefined_strategy = MovingAverageCrossoverStrategy(
                    short_window=ma_short, # Direct variable from sidebar
                    long_window=ma_long,   # Direct variable from sidebar
                    name=f"MA_Crossover_{ma_short}_{ma_long}"
                )
                strategies_to_run.append(predefined_strategy)
            elif strategy_type == "RSI Strategy":
                predefined_strategy = RSIStrategy(
                    rsi_period=st.session_state.backtest_rsi_length_val, # From session_state
                    oversold=st.session_state.rsi_oversold_val,         # From session_state
                    overbought=st.session_state.rsi_overbought_val,     # From session_state
                    name=f"RSI_{st.session_state.backtest_rsi_length_val}_{st.session_state.rsi_oversold_val}_{st.session_state.rsi_overbought_val}"
                )
                strategies_to_run.append(predefined_strategy)

            # Prepare custom strategy (if enabled and configured)
            if st.session_state.get('use_custom_strategy', False):
                custom_config = st.session_state.get('custom_strategy_config', {})
                if custom_config.get('buy_condition') and custom_config.get('sell_condition') and custom_config.get('name') and custom_config.get('indicators') is not None:
                    # 'indicators' should be a list of dicts like [{'name': 'SMA', 'params': {'length': 20}}, ...]
                    # This is already correctly stored in st.session_state.custom_strategy_config['indicators']
                    strategy_details_for_generic = {
                        'indicators': custom_config.get('indicators', []),
                        'buy_condition': custom_config.get('buy_condition'),
                        'sell_condition': custom_config.get('sell_condition')
                    }
                    custom_strategy_obj = GenericUserStrategy(
                        strategy_config=strategy_details_for_generic,
                        name=custom_config['name']
                    )
                    strategies_to_run.append(custom_strategy_obj)
                else:
                    st.warning(f"Custom strategy '{custom_config.get('name', 'Unnamed')}' is enabled but not fully configured (missing rules or indicators). It will not be run.")

            if not strategies_to_run:
                st.warning("No strategies selected or properly configured for backtesting.")
            else:
                for strategy_obj in strategies_to_run:
                    st.subheader(f"Results for: {strategy_obj.name}")
                    if isinstance(strategy_obj, MovingAverageCrossoverStrategy):
                        st.markdown(f"Strategy Type: Moving Average Crossover (Short: {strategy_obj.short_window}, Long: {strategy_obj.long_window})")
                    elif isinstance(strategy_obj, RSIStrategy):
                        st.markdown(f"Strategy Type: RSI (Period: {strategy_obj.rsi_period}, Oversold: {strategy_obj.oversold}, Overbought: {strategy_obj.overbought})")
                    elif isinstance(strategy_obj, GenericUserStrategy):
                        st.markdown(f"Strategy Type: Custom User Strategy")
                        st.markdown(f"Buy Condition: `{strategy_obj.buy_condition_str}`")
                        st.markdown(f"Sell Condition: `{strategy_obj.sell_condition_str}`")
                        if strategy_obj.indicators_config:
                            inds_summary = ", ".join([f"{ind['name']}({', '.join([f'{k}={v}' for k,v in ind.get('params', {}).items()])})" for ind in strategy_obj.indicators_config])
                            st.markdown(f"Indicators: {inds_summary}")
                    
                    # Ensure data is available in session state
                    if 'data' not in st.session_state or st.session_state.data.empty:
                        st.error("No data available for backtesting. Please fetch data first from the 'Technical Analysis' tab or main settings.")
                        continue # Skip to next strategy if data is missing

                    # initial_capital and commission are direct variables from sidebar scope
                    with st.spinner(f"Running backtest for {strategy_obj.name}..."):
                        result = engine.run_backtest(
                            strategy=strategy_obj,
                            data=st.session_state.data,
                            initial_capital=initial_capital, 
                            commission=commission 
                        )
                    
                    # Display evaluation error if any for GenericUserStrategy
                    if isinstance(strategy_obj, GenericUserStrategy) and hasattr(strategy_obj, 'evaluation_error') and strategy_obj.evaluation_error:
                        st.error(f"⚠️ Custom Strategy Rule Evaluation Error(s) for '{strategy_obj.name}':\n{strategy_obj.evaluation_error}")

                    if result:
                        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                        with res_col1:
                            st.metric("Total Return", f"{result.get('total_return', np.nan):.2%}")
                        with res_col2:
                            st.metric("Sharpe Ratio", f"{result.get('sharpe_ratio', np.nan):.2f}")
                        with res_col3:
                            st.metric("Max Drawdown", f"{result.get('max_drawdown', np.nan):.2%}")
                        with res_col4:
                            win_rate = result.get('win_rate', np.nan)
                            st.metric("Win Rate", f"{win_rate:.2%}" if not pd.isna(win_rate) else "N/A")
                        
                        st.subheader("Backtest Chart")
                        backtest_fig = engine.visualize_results(result, st.session_state.symbol, strategy_obj.name)
                        if backtest_fig:
                            st.pyplot(backtest_fig)
                        else:
                            st.warning("Could not generate the backtest plot.")
                        
                        st.subheader("Trade Details")
                        trades_df = result.get('trades_df', pd.DataFrame()) # Assuming engine might return df directly
                        if not trades_df.empty:
                            st.dataframe(trades_df)
                        else:
                            st.info("No trades were executed during the backtest period for this strategy.")
                        
                        st.subheader("Signal Data")
                        signals_df = result.get('signals', pd.DataFrame())
                        if not signals_df.empty:
                            st.dataframe(signals_df)
                            csv_signals = signals_df.to_csv().encode('utf-8')
                            st.download_button(
                                label=f"Download Signals CSV for {strategy_obj.name}",
                                data=csv_signals,
                                file_name=f"{st.session_state.symbol}_{strategy_obj.name}_signals.csv",
                                mime="text/csv",
                                key=f"download_signals_{strategy_obj.name}" # Unique key for download button
                            )
                        else:
                            st.info("No signal data to display for this strategy.")
                    else:
                        st.error(f"Backtest failed to produce results for strategy: {strategy_obj.name}")

                    st.markdown("---") # Separator for next strategy

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

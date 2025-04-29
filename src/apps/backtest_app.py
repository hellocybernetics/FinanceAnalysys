# src/apps/backtest_app.py
import streamlit as st
import pandas as pd
import numpy as np
import vectorbt as vbt
import talib
import seaborn as sns
import matplotlib.pyplot as plt
import traceback

# Import necessary functions from other modules
# from src.core.data.yahoo.fetch_from_yahoo import fetch_stock_data # Removed

# vectorbt global settings
# vbt.settings.set_theme("streamlit") # Use streamlit theme for plots -> Commented out due to KeyError
vbt.settings.plotting['layout']['width'] = 1000 # Adjust plot width
vbt.settings.plotting['layout']['height'] = 600
vbt.settings.array_wrapper['freq'] = 'D'

# --- Helper Functions (Indicators and Signals) ---

# Indicators (using TA-Lib is generally faster than manual calculation)
def chande_momentum_oscillator(close_prices, period=14):
    return talib.CMO(close_prices, timeperiod=period)

def trix_indicator(close_prices, period=14):
    trix = talib.TRIX(close_prices, timeperiod=period)
    trix_signal = talib.EMA(trix, timeperiod=9) # Common practice: 9-period EMA signal for Trix
    return trix, trix_signal

# Reverted parameter names
def generate_signals(df_close, cmo_period, trix_period):
    # Expects df_close to be a pandas Series
    cmo = chande_momentum_oscillator(df_close, period=cmo_period)
    trix, trix_signal = trix_indicator(df_close, period=trix_period)

    # Ensure alignment by using pandas operations before converting to numpy
    entries_pd = (cmo > 0) & (trix > trix_signal) & (trix.shift(1) <= trix_signal.shift(1))
    exits_pd = (cmo < 0) & (trix < trix_signal) & (trix.shift(1) >= trix_signal.shift(1))

    # Align index with original df_close before converting to numpy
    # Using align might be unnecessary if signals are generated on the same df index
    # entries_aligned, exits_aligned = vbt.base.indexing.align(entries_pd, exits_pd, df_close.index)

    return entries_pd.to_numpy(), exits_pd.to_numpy()

# --- Main Backtesting Function --- Reverted to loop-based optimization

def run_backtest_optimization(df_input, symbol, cmo_range, trix_range, initial_capital):
    """Runs the backtest optimization using loops and returns results."""
    if df_input is None or df_input.empty:
        st.error("Input DataFrame is empty or None.")
        return None, None, None, None

    # --- Data Preparation ---
    st.write("Preparing data for backtest...")
    df = df_input.copy()
    try:
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is not None:
                df.index = df.index.tz_convert('UTC').tz_localize(None)
        else:
            df.index = pd.to_datetime(df.index, utc=True).tz_convert('UTC').tz_localize(None)

        if 'Close' not in df.columns:
             st.error("DataFrame must contain a 'Close' column.")
             return None, None, None, None
        df['Close'] = df['Close'].astype(np.float64)

        df = df.resample('D').ffill()
        df.index.freq = df.index.inferred_freq

        df.dropna(subset=['Close'], inplace=True)
    except Exception as e:
        st.error(f"Error preparing data for backtest: {e}")
        st.error(traceback.format_exc())
        return None, None, None, None

    if df.empty:
        st.error("Data became empty after preparation for backtest.")
        return None, None, None, None
    # --- End Data Preparation ---

    st.write(f"Running optimization loop for CMO {cmo_range} and Trix {trix_range}...")

    cmo_periods = np.arange(cmo_range[0], cmo_range[1] + 1)
    trix_periods = np.arange(trix_range[0], trix_range[1] + 1)

    # Initialize storage for results
    total_returns_matrix = np.full((len(cmo_periods), len(trix_periods)), np.nan)
    # Using a dictionary to store portfolios might be easier to access by param tuple
    portfolios_dict = {}

    # --- Optimization Loop ---
    progress_bar = st.progress(0)
    total_combinations = len(cmo_periods) * len(trix_periods)
    current_combination = 0

    try:
        for i, cmo_p in enumerate(cmo_periods):
            for j, trix_p in enumerate(trix_periods):
                current_combination += 1
                # Update progress bar
                progress_bar.progress(current_combination / total_combinations,
                                      text=f"Testing CMO={cmo_p}, Trix={trix_p} ({current_combination}/{total_combinations})")

                # Generate signals for this combination
                entries, exits = generate_signals(df['Close'], cmo_p, trix_p)

                # Skip if no entry signals are generated
                if not np.any(entries):
                    total_returns_matrix[i, j] = -1 # Indicate no trades or poor performance
                    continue

                # Run portfolio simulation for this combination
                pf = vbt.Portfolio.from_signals(
                    close=df['Close'],
                    entries=entries,
                    exits=exits,
                    freq=df.index.freq,
                    init_cash=initial_capital,
                    fees=0.001,
                    sl_stop=0.05
                    # group_by=False, use_col_idxs=False needed here? No, single run.
                )

                # Store results
                total_returns_matrix[i, j] = pf.total_return()
                portfolios_dict[(cmo_p, trix_p)] = pf # Store portfolio keyed by params

        progress_bar.empty() # Remove progress bar after completion

        # --- Process Results ---
        # Find best parameters based on the highest total return
        if np.isnan(total_returns_matrix).all():
             st.warning("Backtest optimization resulted in no valid returns.")
             return None, None, None, None

        best_cmo_idx, best_trix_idx = np.unravel_index(np.nanargmax(total_returns_matrix), total_returns_matrix.shape)
        best_cmo_period = cmo_periods[best_cmo_idx]
        best_trix_period = trix_periods[best_trix_idx]
        best_return = total_returns_matrix[best_cmo_idx, best_trix_idx]

        best_params = {'cmo_period': best_cmo_period, 'trix_period': best_trix_period}
        st.success(f"Optimization complete. Best params: CMO={best_cmo_period}, Trix={best_trix_period} (Return: {best_return:.2%})")

        # Prepare heatmap data (using the matrix directly)
        heatmap_df = pd.DataFrame(total_returns_matrix, index=cmo_periods, columns=trix_periods)
        heatmap_df.index.name = 'CMO Period'
        heatmap_df.columns.name = 'Trix Period'

        # Generate heatmap
        st.write("Generating heatmap...")
        fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 8))
        sns.heatmap(heatmap_df, annot=False, cmap="viridis", fmt=".2%", ax=ax_heatmap)
        ax_heatmap.set_title('Total Return Heatmap')
        plt.tight_layout()

        # Get the portfolio for the best parameters
        best_pf = portfolios_dict.get((best_cmo_period, best_trix_period))
        if best_pf is None:
            st.error("Could not retrieve the best performing portfolio. This should not happen.")
            return fig_heatmap, None, None, best_params

        stats = best_pf.stats()

        # Generate portfolio plot for the best parameters
        st.write("Generating best portfolio plot...")
        fig_pf = None
        try:
            fig_pf = best_pf.plot(subplots=[
                'orders',
                'trade_pnl',
                'cum_returns'
            ])
        except Exception as plot_err:
             st.warning(f"Could not generate portfolio plot: {plot_err}")

    except Exception as e:
        st.error(f"Error during backtest optimization loop: {e}")
        st.error(traceback.format_exc())
        progress_bar.empty() # Ensure progress bar is removed on error
        return None, None, None, None

    return fig_heatmap, fig_pf, stats, best_params 
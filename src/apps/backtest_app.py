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
    """Generates long and short entry/exit signals."""
    # Expects df_close to be a pandas Series
    cmo = chande_momentum_oscillator(df_close, period=cmo_period)
    trix, trix_signal = trix_indicator(df_close, period=trix_period)

    # Long signals
    entries_pd = (cmo > 0) & (trix > trix_signal) & (trix.shift(1) <= trix_signal.shift(1))
    exits_pd = (cmo < 0) & (trix < trix_signal) & (trix.shift(1) >= trix_signal.shift(1))

    # Short signals (symmetric)
    short_entries_pd = exits_pd.copy() # Enter short when long exits
    short_exits_pd = entries_pd.copy() # Exit short when long enters

    # Convert to numpy arrays
    entries = entries_pd.to_numpy()
    exits = exits_pd.to_numpy()
    short_entries = short_entries_pd.to_numpy()
    short_exits = short_exits_pd.to_numpy()

    return entries, exits, short_entries, short_exits # Return all four

# --- Main Backtesting Function ---

# Updated signature to include new parameters for Fixed Risk sizing
def run_backtest_optimization(
    df_input, symbol,
    # Signal params (used in signal mode)
    cmo_range, trix_range,
    # Sizing params (used in sizing mode)
    atr_p_range=None,
    target_vol_range=None,
    osc_scale_range=None,
    osc_base_size_pct=0.1,
    osc_clip_value=50,
    bb_std_range=None,
    bb_width_scale_range=None,
    bb_len=20,
    bb_base_size_pct=0.1,
    risk_pct=0.02, # Fixed risk per trade % (default 2%)
    sl_pct_range=None, # Stop loss % range for optimization
    # Common params
    initial_capital=100000,
    allow_shorting=False, use_trailing_stop=False, tsl_pct=0.0, 
    # Mode control
    optimization_mode="シグナルパラメータ最適化",
    sizing_algorithm=None,
    # Fixed signal params from Phase 1
    fixed_signal_params: dict | None = None
):
    """Runs backtest optimization based on the selected mode."""
    # --- Initial Checks ---
    if df_input is None or df_input.empty:
        st.error("Input DataFrame is empty or None.")
        return None, None, None, None, None

    # --- Data Preparation ---
    # st.write("Preparing data for backtest...") # Less verbose
    df = df_input.copy()
    try:
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is not None:
                df.index = df.index.tz_convert('UTC').tz_localize(None)
        else:
            df.index = pd.to_datetime(df.index, utc=True).tz_convert('UTC').tz_localize(None)

        if 'Close' not in df.columns:
             st.error("DataFrame must contain a 'Close' column.")
             return None, None, None, None, None
        # Ensure High, Low columns exist if needed for ATR
        if optimization_mode == "取引量決定最適化" and sizing_algorithm == "ボラティリティ基準":
             if 'High' not in df.columns or 'Low' not in df.columns:
                  st.error("ボラティリティ基準の取引量計算には 'High' と 'Low' 列が必要です。")
                  return None, None, None, None, None
             df['High'] = df['High'].astype(np.float64)
             df['Low'] = df['Low'].astype(np.float64)

        df['Close'] = df['Close'].astype(np.float64)
        df = df.resample('D').ffill() # Use ffill for daily data
        df.index.freq = df.index.inferred_freq
        df.dropna(subset=['Close'], inplace=True) # Drop days where close is NaN after ffill
        # Drop NaNs for H/L too if ATR is needed
        if optimization_mode == "取引量決定最適化" and sizing_algorithm == "ボラティリティ基準":
             df.dropna(subset=['High', 'Low'], inplace=True)

    except Exception as e:
        st.error(f"Error preparing data for backtest: {e}")
        st.error(traceback.format_exc())
        return None, None, None, None, None

    if df.empty:
        st.error("Data became empty after preparation for backtest.")
        return None, None, None, None, None
    # --- End Data Preparation ---

    # ==========================================================================
    # --- Optimization Mode Branching ---
    # ==========================================================================
    fig_results_plot = None
    fig_heatmap = None
    best_pf = None
    best_params = None
    stats = None
    accurate_position_size = None

    if optimization_mode == "シグナルパラメータ最適化":
        st.write(f"シグナルパラメータ最適化を実行中 (CMO {cmo_range}, Trix {trix_range})...")
        # --- Existing Signal Parameter Optimization Logic ---
        cmo_periods = np.arange(cmo_range[0], cmo_range[1] + 1)
        trix_periods = np.arange(trix_range[0], trix_range[1] + 1)
        total_returns_matrix = np.full((len(cmo_periods), len(trix_periods)), np.nan)
        portfolios_dict = {}
        progress_bar = st.progress(0)
        total_combinations = len(cmo_periods) * len(trix_periods)
        current_combination = 0

        try:
            for i, cmo_p in enumerate(cmo_periods):
                for j, trix_p in enumerate(trix_periods):
                    current_combination += 1
                    progress_bar.progress(current_combination / total_combinations,
                                          text=f"Testing CMO={cmo_p}, Trix={trix_p} ({current_combination}/{total_combinations})")
                    entries, exits, short_entries, short_exits = generate_signals(df['Close'], cmo_p, trix_p)
                    if not np.any(entries) and (not allow_shorting or not np.any(short_entries)):
                        total_returns_matrix[i, j] = np.nan # Use NaN for consistency
                        continue
                    portfolio_kwargs = {
                        'close': df['Close'], 'entries': entries, 'exits': exits,
                        'freq': df.index.freq, 'init_cash': initial_capital, 'fees': 0.001
                    }
                    if allow_shorting:
                        portfolio_kwargs['short_entries'] = short_entries
                        portfolio_kwargs['short_exits'] = short_exits
                    if use_trailing_stop and tsl_pct > 0:
                        portfolio_kwargs['sl_stop'] = tsl_pct
                    pf = vbt.Portfolio.from_signals(**portfolio_kwargs)
                    total_returns_matrix[i, j] = pf.total_return()
                    portfolios_dict[(cmo_p, trix_p)] = pf

            progress_bar.empty()
            if np.isnan(total_returns_matrix).all():
                 st.warning("Backtest optimization resulted in no valid returns.")
                 return None, None, None, None, None
            best_cmo_idx, best_trix_idx = np.unravel_index(np.nanargmax(total_returns_matrix), total_returns_matrix.shape)
            best_cmo_period = cmo_periods[best_cmo_idx]
            best_trix_period = trix_periods[best_trix_idx]
            best_params = {'cmo_period': best_cmo_period, 'trix_period': best_trix_period}
            best_return = total_returns_matrix[best_cmo_idx, best_trix_idx]
            st.success(f"シグナル最適化完了. Best params: CMO={best_cmo_period}, Trix={best_trix_period} (Return: {best_return:.2f}%)")
            heatmap_df = pd.DataFrame(total_returns_matrix * 100, index=cmo_periods, columns=trix_periods) # Show returns in %
            heatmap_df.index.name = 'CMO Period'
            heatmap_df.columns.name = 'Trix Period'
            fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 8))
            sns.heatmap(heatmap_df, annot=False, cmap="viridis", fmt=".2f", ax=ax_heatmap, cbar_kws={'format': '%.0f%%'}) # Format cbar
            ax_heatmap.set_title('Total Return (%) Heatmap (Signal Params)')
            plt.tight_layout()
            best_pf = portfolios_dict.get((best_cmo_period, best_trix_period))

        except Exception as e:
            st.error(f"Error during signal parameter optimization loop: {e}")
            st.error(traceback.format_exc())
            if 'progress_bar' in locals(): progress_bar.empty()
            return None, None, None, None, None
        # --- End of Signal Parameter Optimization Logic ---

    elif optimization_mode == "取引量決定最適化":
        # --- Determine fixed signal parameters to use ---
        default_cmo_p = 14
        default_trix_p = 14
        if fixed_signal_params and isinstance(fixed_signal_params, dict):
            fixed_cmo_p = fixed_signal_params.get('cmo_period', default_cmo_p)
            fixed_trix_p = fixed_signal_params.get('trix_period', default_trix_p)
            st.info(f"セッション状態から取得した固定シグナルパラメータを使用: CMO={fixed_cmo_p}, Trix={fixed_trix_p}")
        else:
            st.warning(f"固定シグナルパラメータが見つかりません。デフォルト値を使用: CMO={default_cmo_p}, Trix={default_trix_p}")
            fixed_cmo_p = default_cmo_p
            fixed_trix_p = default_trix_p
        
        # --- Generate base signals once with fixed parameters ---
        entries, exits, short_entries, short_exits = generate_signals(df['Close'], fixed_cmo_p, fixed_trix_p)
        
        if not np.any(entries) and (not allow_shorting or not np.any(short_entries)):
            st.warning("ベースとなる売買シグナルが生成されませんでした。取引量最適化を実行できません。")
            return None, None, None, None, None
            
        # --- Sizing Algorithm Branching ---
        if sizing_algorithm == "ボラティリティ基準":
            st.write(f"取引量最適化（ボラティリティ基準）を実行中 (ATR {atr_p_range}, Vol Target {target_vol_range})...")

            # Validate ranges
            if atr_p_range is None or target_vol_range is None:
                 st.error("ATR期間範囲と目標ボラティリティ範囲が必要です。")
                 return None, None, None, None, None

            # Parameter ranges for sizing
            atr_periods = np.arange(atr_p_range[0], atr_p_range[1] + 1)
            # Target volatility range is percentage, convert to decimal
            target_vols = np.linspace(target_vol_range[0] / 100.0, target_vol_range[1] / 100.0, 10) # Example: 10 steps

            total_returns_matrix = np.full((len(atr_periods), len(target_vols)), np.nan)
            portfolios_dict = {}
            progress_bar = st.progress(0)
            total_combinations = len(atr_periods) * len(target_vols)
            current_combination = 0

            # Common portfolio args (exclude size initially)
            portfolio_common_kwargs = {
                'close': df['Close'],
                'entries': entries, # Keep signals for timing
                'exits': exits,
                'short_entries': short_entries if allow_shorting else None, # Pass short signals if allowed
                'short_exits': short_exits if allow_shorting else None,
                'freq': df.index.freq, 
                'init_cash': initial_capital, 
                'fees': 0.001
                # sl_stop is handled below IF needed
            }
            if use_trailing_stop and tsl_pct > 0:
                 # This might conflict with size calculation logic, needs careful testing
                 # Let's add it back, but be aware it might override some size logic implicitly
                 portfolio_common_kwargs['sl_stop'] = tsl_pct
            
            # Add allow_shorting flag explicitly if needed by underlying functions
            # portfolio_common_kwargs['allow_shorting'] = allow_shorting

            try:
                for i, atr_p in enumerate(atr_periods):
                    # Calculate ATR
                    atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=atr_p)
                    # Handle ATR NaNs and potential zeros
                    atr_safe = atr.fillna(method='ffill').fillna(method='bfill').fillna(1e-6).where(atr != 0, 1e-6)

                    for j, target_vol in enumerate(target_vols):
                        current_combination += 1
                        progress_bar.progress(current_combination / total_combinations,
                                              text=f"Testing ATR Period={atr_p}, Target Vol={target_vol:.2%} ({current_combination}/{total_combinations})")

                        # --- Calculate Order Size in Shares --- 
                        order_size_shares = np.zeros_like(df['Close'], dtype=float) # Initialize order size array
                        risk_per_trade = initial_capital * target_vol # Simplified risk amount
                        
                        # Calculate size for long entries
                        entry_points = np.where(entries)[0]
                        if len(entry_points) > 0:
                            atr_at_entry = atr_safe.iloc[entry_points]
                            size_at_entry = (risk_per_trade / atr_at_entry).fillna(0).astype(int)
                            order_size_shares[entry_points] += size_at_entry # Add positive size
                        
                        # Calculate size for short entries (if allowed)
                        if allow_shorting:
                            short_entry_points = np.where(short_entries)[0]
                            if len(short_entry_points) > 0:
                                atr_at_short_entry = atr_safe.iloc[short_entry_points]
                                size_at_short_entry = (risk_per_trade / atr_at_short_entry).fillna(0).astype(int)
                                # Ensure size array contains only POSITIVE values
                                # Direction is handled by short_entries signal
                                order_size_shares[short_entry_points] += size_at_short_entry # CORRECTED: ADD positive size
                        
                        # Check if any non-zero orders were generated
                        if np.all(order_size_shares == 0):
                            total_returns_matrix[i, j] = np.nan # Indicate no trades
                            continue 

                        # Run portfolio simulation with absolute order size
                        pf = vbt.Portfolio.from_signals(
                            size=order_size_shares,      # Pass calculated order sizes (shares)
                            # size_type removed
                            **portfolio_common_kwargs  # Add common arguments (includes signals for timing)
                        )

                        total_returns_matrix[i, j] = pf.total_return()
                        portfolios_dict[(atr_p, target_vol)] = pf

                progress_bar.empty()
                if np.isnan(total_returns_matrix).all():
                     st.warning("取引量最適化では有効なリターンが得られませんでした。")
                     return None, None, None, None, None

                # Find best sizing parameters
                best_atr_idx, best_vol_idx = np.unravel_index(np.nanargmax(total_returns_matrix), total_returns_matrix.shape)
                best_atr_period = atr_periods[best_atr_idx]
                best_target_vol = target_vols[best_vol_idx]
                best_params = {'atr_period': best_atr_period, 'target_vol_pct': best_target_vol * 100} # Store vol as %
                best_return = total_returns_matrix[best_atr_idx, best_vol_idx]
                st.success(f"取引量最適化完了. Best params: ATR Period={best_atr_period}, Target Vol={best_target_vol:.2%} (Return: {best_return:.2%})")

                # Prepare heatmap for sizing parameters
                heatmap_df = pd.DataFrame(total_returns_matrix * 100, index=atr_periods, columns=[f"{v:.1%}" for v in target_vols]) # Show returns in %, format vol axis
                heatmap_df.index.name = 'ATR Period'
                heatmap_df.columns.name = 'Target Volatility (%)'
                fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 8))
                sns.heatmap(heatmap_df, annot=False, cmap="viridis", fmt=".2f", ax=ax_heatmap, cbar_kws={'format': '%.0f%%'}) # Format cbar
                ax_heatmap.set_title('Total Return (%) Heatmap (Sizing Params)')
                plt.tight_layout()
                best_pf = portfolios_dict.get((best_atr_period, best_target_vol))

            except Exception as e:
                st.error(f"Error during volatility sizing optimization loop: {e}")
                st.error(traceback.format_exc())
                if 'progress_bar' in locals(): progress_bar.empty()
                return None, None, None, None, None
            # --- End of Volatility Sizing Logic ---

        elif sizing_algorithm == "オシレータ基準":
            st.write(f"取引量最適化（オシレータ基準）を実行中 (Scale Factor {osc_scale_range})...")

            # Validate ranges/params
            if osc_scale_range is None:
                 st.error("オシレータスケール係数範囲が必要です。")
                 return None, None, None, None, None

            # Calculate CMO values needed for this algorithm
            cmo_values = chande_momentum_oscillator(df['Close'], period=fixed_cmo_p) 

            if not np.any(entries) and (not allow_shorting or not np.any(short_entries)):
                st.warning("ベースとなる売買シグナルが生成されませんでした。取引量最適化を実行できません。")
                return None, None, None, None, None

            # Parameter range for optimization
            osc_scale_factors = np.linspace(osc_scale_range[0], osc_scale_range[1], 10) # Example: 10 steps

            # Storage for 1D results
            returns_list = []
            portfolios_dict_1d = {}

            progress_bar = st.progress(0)
            total_combinations = len(osc_scale_factors)
            current_combination = 0
            
            # Common portfolio args (exclude size initially)
            portfolio_common_kwargs = {
                'close': df['Close'], 'entries': entries, 'exits': exits,
                'short_entries': short_entries if allow_shorting else None,
                'short_exits': short_exits if allow_shorting else None,
                'freq': df.index.freq, 'init_cash': initial_capital, 'fees': 0.001
            }
            if use_trailing_stop and tsl_pct > 0:
                 portfolio_common_kwargs['sl_stop'] = tsl_pct

            try:
                for scale_factor in osc_scale_factors:
                    current_combination += 1
                    progress_bar.progress(current_combination / total_combinations,
                                          text=f"Testing Osc Scale Factor={scale_factor:.2f} ({current_combination}/{total_combinations})")

                    # --- Calculate Order Size in Shares based on Oscillator --- 
                    base_value = initial_capital * osc_base_size_pct
                    # Clip absolute CMO, handle NaNs, calculate scaling factor
                    clipped_osc = cmo_values.abs().clip(0, osc_clip_value).fillna(0)
                    # Prevent division by zero if osc_clip_value is 0
                    clip_divisor = osc_clip_value if osc_clip_value > 0 else 1.0
                    scaling_effect = scale_factor * (clipped_osc / clip_divisor)
                    target_invest_value = base_value * (1 + scaling_effect)
                    
                    # Calculate target shares (ensure positive) at entry points
                    order_size_shares = np.zeros_like(df['Close'], dtype=float)
                    entry_points = np.where(entries)[0]
                    if len(entry_points) > 0:
                        entry_prices = df['Close'].iloc[entry_points].where(lambda x: x!= 0, 1e-9) # Avoid zero price
                        target_shares_long = (target_invest_value.iloc[entry_points] / entry_prices).fillna(0).astype(int)
                        order_size_shares[entry_points] += target_shares_long

                    if allow_shorting:
                         short_entry_points = np.where(short_entries)[0]
                         if len(short_entry_points) > 0:
                              entry_prices_short = df['Close'].iloc[short_entry_points].where(lambda x: x!= 0, 1e-9)
                              target_shares_short = (target_invest_value.iloc[short_entry_points] / entry_prices_short).fillna(0).astype(int)
                              order_size_shares[short_entry_points] += target_shares_short # Still ADD positive size

                    if np.all(order_size_shares == 0):
                        returns_list.append(np.nan)
                        continue

                    # Run portfolio simulation with absolute order size
                    pf = vbt.Portfolio.from_signals(
                        size=order_size_shares, # Absolute shares
                        # size_type removed
                        **portfolio_common_kwargs
                    )
                    
                    current_return = pf.total_return()
                    returns_list.append(current_return)
                    portfolios_dict_1d[scale_factor] = pf

                progress_bar.empty()
                returns_array = np.array(returns_list)
                if np.isnan(returns_array).all():
                     st.warning("オシレータ基準最適化では有効なリターンが得られませんでした。")
                     return None, None, None, None, None

                # Find best scale factor
                best_idx = np.nanargmax(returns_array)
                best_osc_scale_factor = osc_scale_factors[best_idx]
                best_params = {'osc_scale_factor': best_osc_scale_factor} # Store best factor
                best_return = returns_array[best_idx]
                st.success(f"オシレータ基準最適化完了. Best Scale Factor={best_osc_scale_factor:.2f} (Return: {best_return:.2%})")

                # Generate 1D results plot
                fig_results_plot, ax_results = plt.subplots(figsize=(10, 6))
                ax_results.plot(osc_scale_factors, returns_array * 100) # Show return in %
                ax_results.set_xlabel('Oscillator Scale Factor')
                ax_results.set_ylabel('Total Return (%)')
                ax_results.set_title('Total Return vs Oscillator Scale Factor')
                ax_results.grid(True)
                plt.tight_layout()
                
                best_pf = portfolios_dict_1d.get(best_osc_scale_factor)
                fig_heatmap = None # No heatmap for 1D optimization

            except Exception as e:
                st.error(f"Error during oscillator sizing optimization loop: {e}")
                st.error(traceback.format_exc())
                if 'progress_bar' in locals(): progress_bar.empty()
                return None, None, None, None, None
            # --- End of Oscillator Sizing Logic ---

        elif sizing_algorithm == "ボリンジャーバンド幅基準":
            # ... (BB Width Sizing Logic - sets fig_heatmap, best_pf, best_params)
            pass 
            
        elif sizing_algorithm == "資金管理基準 (固定リスク率)":
            st.write(f"取引量最適化（資金管理基準）を実行中 (SL % {sl_pct_range}, Risk={risk_pct:.1%})")
            
            # Validate ranges/params
            if sl_pct_range is None:
                 st.error("ストップロス率範囲が必要です。")
                 return None, None, None, None, None
            
            # Fixed Signal Parameters already determined above
            st.info(f"固定パラメータを使用: Signal(CMO={fixed_cmo_p}, Trix={fixed_trix_p}), Risk Per Trade={risk_pct:.1%}")

            # Parameter range for optimization (SL % -> decimal)
            sl_pcts = np.linspace(sl_pct_range[0] / 100.0, sl_pct_range[1] / 100.0, 10) # 10 steps

            # Storage for 1D results
            returns_list = []
            portfolios_dict_1d = {}

            progress_bar = st.progress(0)
            total_combinations = len(sl_pcts)
            current_combination = 0
            
            # Common portfolio args (exclude size and sl_stop initially)
            # NOTE: We pass sl_stop dynamically INSIDE the loop based on the current sl_pct
            portfolio_common_kwargs = {
                'close': df['Close'], 'entries': entries, 'exits': exits,
                'short_entries': short_entries if allow_shorting else None,
                'short_exits': short_exits if allow_shorting else None,
                'freq': df.index.freq, 'init_cash': initial_capital, 'fees': 0.001
                # Removed use_trailing_stop flag from here, as sl_stop is set manually
            }
            # if use_trailing_stop and tsl_pct > 0: -> We are optimizing SL%, so TSL flag is ignored here
            #      portfolio_common_kwargs['sl_stop'] = tsl_pct

            try:
                for sl_pct in sl_pcts:
                    current_combination += 1
                    progress_bar.progress(current_combination / total_combinations,
                                          text=f"Testing SL={sl_pct:.2%} ({current_combination}/{total_combinations})")

                    # --- Calculate Order Size in Shares based on Fixed Risk --- 
                    order_size_shares = np.zeros_like(df['Close'], dtype=float)
                    amount_to_risk = initial_capital * risk_pct
                    
                    # Calculate size for long entries
                    entry_points = np.where(entries)[0]
                    if len(entry_points) > 0:
                        entry_prices = df['Close'].iloc[entry_points]
                        stop_loss_prices_long = entry_prices * (1 - sl_pct)
                        points_at_risk_long = entry_prices - stop_loss_prices_long
                        # Avoid division by zero or negative risk
                        safe_points_at_risk_long = points_at_risk_long.where(points_at_risk_long > 1e-9, 1e-9)
                        size_at_entry_long = (amount_to_risk / safe_points_at_risk_long).fillna(0).astype(int)
                        order_size_shares[entry_points] += size_at_entry_long
                    
                    # Calculate size for short entries (if allowed)
                    if allow_shorting:
                        short_entry_points = np.where(short_entries)[0]
                        if len(short_entry_points) > 0:
                            entry_prices_short = df['Close'].iloc[short_entry_points]
                            stop_loss_prices_short = entry_prices_short * (1 + sl_pct)
                            points_at_risk_short = entry_prices_short - stop_loss_prices_short # Should be negative
                            # Avoid division by zero or positive risk
                            safe_points_at_risk_short = points_at_risk_short.where(points_at_risk_short < -1e-9, -1e-9)
                            size_at_entry_short = abs(amount_to_risk / safe_points_at_risk_short).fillna(0).astype(int)
                            order_size_shares[short_entry_points] += size_at_entry_short # Add positive size

                    if np.all(order_size_shares == 0):
                        returns_list.append(np.nan)
                        continue

                    # Run portfolio simulation with calculated size and the current sl_pct for stop-loss
                    pf = vbt.Portfolio.from_signals(
                        size=order_size_shares,
                        sl_stop=sl_pct, # Pass the SL% for this iteration
                        **portfolio_common_kwargs
                    )
                    
                    current_return = pf.total_return()
                    returns_list.append(current_return)
                    portfolios_dict_1d[sl_pct] = pf

                progress_bar.empty()
                returns_array = np.array(returns_list)
                if np.isnan(returns_array).all():
                     st.warning("資金管理基準最適化では有効なリターンが得られませんでした。")
                     return None, None, None, None, None

                # Find best stop loss percentage
                best_idx = np.nanargmax(returns_array)
                best_sl_pct = sl_pcts[best_idx]
                best_params = {'sl_pct': best_sl_pct * 100} # Store SL as %
                best_return = returns_array[best_idx]
                st.success(f"資金管理基準最適化完了. Best SL={best_sl_pct:.2%} (Return: {best_return:.2%})")

                # Generate 1D results plot
                fig_results_plot, ax_results = plt.subplots(figsize=(10, 6))
                ax_results.plot(sl_pcts * 100, returns_array * 100) # Show SL % and return %
                ax_results.set_xlabel('Stop Loss Percent (%)')
                ax_results.set_ylabel('Total Return (%)')
                ax_results.set_title('Total Return vs Stop Loss Percent')
                ax_results.grid(True)
                plt.tight_layout()
                
                best_pf = portfolios_dict_1d.get(best_sl_pct)
                fig_heatmap = None # No heatmap for 1D

            except Exception as e:
                st.error(f"Error during Fixed Risk sizing optimization loop: {e}")
                st.error(traceback.format_exc())
                if 'progress_bar' in locals(): progress_bar.empty()
                return None, None, None, None, None
            # --- End of Fixed Risk Sizing Logic ---
            
        else:
            st.warning(f"取引量決定アルゴリズム「{sizing_algorithm}」は未実装です。")
            return None, None, None, None, None
        # --- End of Position Sizing Optimization Logic ---

    else:
        st.error(f"未定義の最適化モードです: {optimization_mode}")
        return None, None, None, None, None
    # ==========================================================================
    # --- Common Post-processing (Get Stats, Position Size, Plot Portfolio) ---
    # ==========================================================================
    if best_pf is None:
         st.error("最適化後に有効なポートフォリオが見つかりませんでした。")
         # Pass fig_results_plot if it exists, otherwise fig_heatmap or None
         plot_fig = fig_results_plot if fig_results_plot is not None else (fig_heatmap if 'fig_heatmap' in locals() else None)
         return plot_fig, None, None, best_params if 'best_params' in locals() else None, None

    try:
        stats = best_pf.stats()

        # --- Accurate Position Size Calculation --- (Keep this common logic)
        # st.write("Calculating accurate position size from trades...") # Less verbose
        accurate_position_size = pd.Series(0.0, index=df.index)
        records_df = best_pf.trades.records
        if not records_df.empty:
            try:
                entry_indices = df.index[records_df['entry_idx'].astype(int)]
                exit_indices = df.index[records_df['exit_idx'].astype(int)]
                entry_adj_size = records_df['size'] * np.where(records_df['direction'] == 0, 1, -1)
                entry_changes = pd.Series(entry_adj_size.values, index=entry_indices)
                closed_trades = records_df[records_df['status'] == 1]
                if not closed_trades.empty:
                    closed_exit_indices = df.index[closed_trades['exit_idx'].astype(int)]
                    exit_adj_size_closed = -closed_trades['size'][closed_trades.index] * np.where(closed_trades['direction'] == 0, 1, -1) # Index alignment needed
                    exit_changes = pd.Series(exit_adj_size_closed.values, index=closed_exit_indices)
                else:
                    exit_changes = pd.Series(dtype=float)
                all_changes = pd.concat([entry_changes, exit_changes])
                net_changes = all_changes.groupby(all_changes.index).sum()
                aligned_changes, _ = net_changes.align(accurate_position_size, fill_value=0.0)
                accurate_position_size = aligned_changes.cumsum()
            except Exception as calc_err:
                st.warning(f"ポジションサイズの正確な計算中にエラー: {calc_err}")
                accurate_position_size = None
        else:
            # st.info("No trades found in records, position size remains 0.") # Already handled inside mode
            pass # accurate_position_size remains Series of 0.0

        # --- Generate portfolio plot --- (Keep this common logic)
        # st.write("Generating best portfolio plot...") # Less verbose
        fig_pf = None
        try:
            fig_pf = best_pf.plot(subplots=[
                'orders','trade_pnl','cum_returns'
            ])
        except Exception as plot_err:
             st.warning(f"Could not generate portfolio plot: {plot_err}")

    except Exception as post_proc_err:
         st.error(f"Error during post-processing (stats/position/plot): {post_proc_err}")
         st.error(traceback.format_exc())
         # Pass fig_results_plot if it exists, otherwise fig_heatmap or None
         plot_fig = fig_results_plot if fig_results_plot is not None else (fig_heatmap if 'fig_heatmap' in locals() else None)
         return plot_fig, None, None, best_params if 'best_params' in locals() else None, None

    # --- Final Return ---
    plot_fig = fig_results_plot if fig_results_plot is not None else fig_heatmap # Return the correct plot figure
    return plot_fig, fig_pf, stats, best_params, accurate_position_size 
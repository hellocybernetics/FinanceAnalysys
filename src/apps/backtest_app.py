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

# Updated signature to include new parameters for Fixed Risk sizing and IS/OOS split
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
    fixed_signal_params: dict | None = None,
    # IS/OOS split control
    optimization_split_pct: float = 0.7 # Default to 70% for optimization
):
    """Runs backtest optimization with In-Sample optimization and Out-of-Sample validation."""
    # --- Initial Checks ---
    if df_input is None or df_input.empty:
        st.error("Input DataFrame is empty or None.")
        return None, None, None, None, None
    if not 0.1 <= optimization_split_pct <= 0.9:
         st.error("最適化データ期間 (%) は 10% から 90% の間でなければなりません。")
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

    # --- In-Sample / Out-of-Sample Split ---
    min_required_is_length = 30 # Minimum required data points for In-Sample
    min_required_oos_length = 20 # Minimum required data points for Out-of-Sample
    split_index = int(len(df) * optimization_split_pct)

    if split_index < min_required_is_length or len(df) - split_index < min_required_oos_length:
        st.error(f"データが短すぎるか、分割割合 ({optimization_split_pct:.0%}) が不適切で、In-Sample (最低{min_required_is_length}期間) または Out-of-Sample (最低{min_required_oos_length}期間) を確保できません。")
        return None, None, None, None, None # Or appropriate error return

    df_is = df.iloc[:split_index].copy() # Use copy to avoid SettingWithCopyWarning
    df_oos = df.iloc[split_index:].copy()
    st.info(f"データを分割: In-Sample ({len(df_is)}期間: {df_is.index.min().date()} - {df_is.index.max().date()}), Out-of-Sample ({len(df_oos)}期間: {df_oos.index.min().date()} - {df_oos.index.max().date()})")

    # ==========================================================================
    # --- In-Sample Optimization ---
    # ==========================================================================
    fig_optimization_plot = None # Renamed from fig_results_plot/fig_heatmap
    best_pf_is = None # Portfolio from In-Sample optimization
    best_params = None
    # stats_is = None # Stats from IS are usually not the main output

    if optimization_mode == "シグナルパラメータ最適化":
        st.write(f"In-Sample シグナルパラメータ最適化を実行中 (CMO {cmo_range}, Trix {trix_range})...")
        # --- Signal Parameter Optimization Logic (using df_is) ---
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
                                          text=f"IS Testing CMO={cmo_p}, Trix={trix_p} ({current_combination}/{total_combinations})")
                    # Use df_is for signal generation
                    entries, exits, short_entries, short_exits = generate_signals(df_is['Close'], cmo_p, trix_p)
                    if not np.any(entries) and (not allow_shorting or not np.any(short_entries)):
                        total_returns_matrix[i, j] = np.nan # Use NaN for consistency
                        continue
                    portfolio_kwargs = {
                        'close': df_is['Close'], 'entries': entries, 'exits': exits, # Use df_is
                        'freq': df_is.index.freq, 'init_cash': initial_capital, 'fees': 0.001 # Use df_is
                    }
                    if allow_shorting:
                        portfolio_kwargs['short_entries'] = short_entries
                        portfolio_kwargs['short_exits'] = short_exits
                    if use_trailing_stop and tsl_pct > 0:
                        portfolio_kwargs['sl_stop'] = tsl_pct
                    # Run portfolio on df_is
                    pf_is = vbt.Portfolio.from_signals(**portfolio_kwargs)
                    total_returns_matrix[i, j] = pf_is.total_return()
                    portfolios_dict[(cmo_p, trix_p)] = pf_is # Store IS portfolio

            progress_bar.empty()
            if np.isnan(total_returns_matrix).all():
                 st.warning("In-Sample シグナル最適化では有効なリターンが得られませんでした。")
                 return None, None, None, None, None # Cannot proceed without params
            best_cmo_idx, best_trix_idx = np.unravel_index(np.nanargmax(total_returns_matrix), total_returns_matrix.shape)
            best_cmo_period = cmo_periods[best_cmo_idx]
            best_trix_period = trix_periods[best_trix_idx]
            best_params = {'cmo_period': best_cmo_period, 'trix_period': best_trix_period}
            best_return_is = total_returns_matrix[best_cmo_idx, best_trix_idx]
            st.success(f"In-Sample シグナル最適化完了. Best params: CMO={best_cmo_period}, Trix={best_trix_period} (IS Return: {best_return_is:.2%})")

            # Generate IS heatmap
            heatmap_df = pd.DataFrame(total_returns_matrix * 100, index=cmo_periods, columns=trix_periods)
            heatmap_df.index.name = 'CMO Period'
            heatmap_df.columns.name = 'Trix Period'
            fig_heatmap_is, ax_heatmap = plt.subplots(figsize=(10, 8))
            sns.heatmap(heatmap_df, annot=False, cmap="viridis", fmt=".2f", ax=ax_heatmap, cbar_kws={'format': '%.0f%%'})
            ax_heatmap.set_title('In-Sample Total Return (%) Heatmap (Signal Params)')
            plt.tight_layout()
            fig_optimization_plot = fig_heatmap_is # Assign IS plot to return variable
            best_pf_is = portfolios_dict.get((best_cmo_period, best_trix_period)) # Get the best IS portfolio

        except Exception as e:
            st.error(f"Error during In-Sample signal parameter optimization loop: {e}")
            st.error(traceback.format_exc())
            if 'progress_bar' in locals(): progress_bar.empty()
            return None, None, None, None, None
        # --- End of Signal Parameter Optimization Logic (IS) ---

    elif optimization_mode == "取引量決定最適化":
        # --- Determine fixed signal parameters to use (Same logic as before) ---
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

        # --- Generate base signals ONCE for BOTH IS and OOS using fixed parameters ---
        # NOTE: We need signals for the entire period IF size calculation depends on signals (like entry points)
        # But the portfolio simulation for optimization will only use IS data.
        # Let's generate for the whole df, then filter later if needed, or use df_is directly for portfolio runs.
        # It's simpler to generate signals based on the period they are run on.
        entries_is, exits_is, short_entries_is, short_exits_is = generate_signals(df_is['Close'], fixed_cmo_p, fixed_trix_p)

        if not np.any(entries_is) and (not allow_shorting or not np.any(short_entries_is)):
            st.warning("In-Sample 期間でベースとなる売買シグナルが生成されませんでした。取引量最適化を実行できません。")
            return None, None, None, None, None

        # --- Sizing Algorithm Branching (using df_is) ---
        if sizing_algorithm == "ボラティリティ基準":
            st.write(f"In-Sample 取引量最適化（ボラティリティ基準）を実行中 (ATR {atr_p_range}, Vol Target {target_vol_range})...")

            # Validate ranges (Same as before)
            if atr_p_range is None or target_vol_range is None:
                 st.error("ATR期間範囲と目標ボラティリティ範囲が必要です。")
                 return None, None, None, None, None

            atr_periods = np.arange(atr_p_range[0], atr_p_range[1] + 1)
            target_vols = np.linspace(target_vol_range[0] / 100.0, target_vol_range[1] / 100.0, 10)

            total_returns_matrix = np.full((len(atr_periods), len(target_vols)), np.nan)
            portfolios_dict = {}
            progress_bar = st.progress(0)
            total_combinations = len(atr_periods) * len(target_vols)
            current_combination = 0

            # Common portfolio args for IS (exclude size initially)
            portfolio_common_kwargs_is = {
                'close': df_is['Close'],
                'entries': entries_is, # Use IS signals
                'exits': exits_is,
                'short_entries': short_entries_is if allow_shorting else None,
                'short_exits': short_exits_is if allow_shorting else None,
                'freq': df_is.index.freq,
                'init_cash': initial_capital,
                'fees': 0.001
            }
            if use_trailing_stop and tsl_pct > 0:
                 portfolio_common_kwargs_is['sl_stop'] = tsl_pct

            try:
                for i, atr_p in enumerate(atr_periods):
                    # Calculate ATR on df_is
                    atr = talib.ATR(df_is['High'], df_is['Low'], df_is['Close'], timeperiod=atr_p)
                    atr_safe = atr.fillna(method='ffill').fillna(method='bfill').fillna(1e-6).where(atr != 0, 1e-6)

                    for j, target_vol in enumerate(target_vols):
                        current_combination += 1
                        progress_bar.progress(current_combination / total_combinations,
                                              text=f"IS Testing ATR Period={atr_p}, Target Vol={target_vol:.2%} ({current_combination}/{total_combinations})")

                        # Calculate Order Size in Shares based on df_is data
                        order_size_shares_is = np.zeros_like(df_is['Close'], dtype=float)
                        risk_per_trade = initial_capital * target_vol

                        entry_points_is = np.where(entries_is)[0]
                        if len(entry_points_is) > 0:
                            atr_at_entry = atr_safe.iloc[entry_points_is]
                            size_at_entry = (risk_per_trade / atr_at_entry).fillna(0).astype(int)
                            order_size_shares_is[entry_points_is] += size_at_entry

                        if allow_shorting:
                            short_entry_points_is = np.where(short_entries_is)[0]
                            if len(short_entry_points_is) > 0:
                                atr_at_short_entry = atr_safe.iloc[short_entry_points_is]
                                size_at_short_entry = (risk_per_trade / atr_at_short_entry).fillna(0).astype(int)
                                order_size_shares_is[short_entry_points_is] += size_at_short_entry

                        if np.all(order_size_shares_is == 0):
                            total_returns_matrix[i, j] = np.nan
                            continue

                        # Run IS portfolio simulation
                        pf_is = vbt.Portfolio.from_signals(
                            size=order_size_shares_is, # IS order sizes
                            **portfolio_common_kwargs_is # IS common args
                        )

                        total_returns_matrix[i, j] = pf_is.total_return()
                        portfolios_dict[(atr_p, target_vol)] = pf_is # Store IS portfolio

                progress_bar.empty()
                if np.isnan(total_returns_matrix).all():
                     st.warning("In-Sample 取引量最適化（ボラティリティ基準）では有効なリターンが得られませんでした。")
                     return None, None, None, None, None

                best_atr_idx, best_vol_idx = np.unravel_index(np.nanargmax(total_returns_matrix), total_returns_matrix.shape)
                best_atr_period = atr_periods[best_atr_idx]
                best_target_vol = target_vols[best_vol_idx]
                best_params = {'atr_period': best_atr_period, 'target_vol_pct': best_target_vol * 100}
                best_return_is = total_returns_matrix[best_atr_idx, best_vol_idx]
                st.success(f"In-Sample 取引量最適化完了. Best params: ATR Period={best_atr_period}, Target Vol={best_target_vol:.2%} (IS Return: {best_return_is:.2%})")

                # Prepare IS heatmap
                heatmap_df = pd.DataFrame(total_returns_matrix * 100, index=atr_periods, columns=[f"{v:.1%}" for v in target_vols])
                heatmap_df.index.name = 'ATR Period'
                heatmap_df.columns.name = 'Target Volatility (%)'
                fig_heatmap_is, ax_heatmap = plt.subplots(figsize=(10, 8))
                sns.heatmap(heatmap_df, annot=False, cmap="viridis", fmt=".2f", ax=ax_heatmap, cbar_kws={'format': '%.0f%%'})
                ax_heatmap.set_title('In-Sample Total Return (%) Heatmap (Vol Sizing Params)')
                plt.tight_layout()
                fig_optimization_plot = fig_heatmap_is # Assign IS plot
                best_pf_is = portfolios_dict.get((best_atr_period, best_target_vol)) # Get best IS portfolio

            except Exception as e:
                st.error(f"Error during In-Sample volatility sizing optimization loop: {e}")
                st.error(traceback.format_exc())
                if 'progress_bar' in locals(): progress_bar.empty()
                return None, None, None, None, None
            # --- End of Volatility Sizing Logic (IS) ---

        elif sizing_algorithm == "オシレータ基準":
            st.write(f"In-Sample 取引量最適化（オシレータ基準）を実行中 (Scale Factor {osc_scale_range})...")

            if osc_scale_range is None:
                 st.error("オシレータスケール係数範囲が必要です。")
                 return None, None, None, None, None

            # Calculate CMO values on df_is
            cmo_values_is = chande_momentum_oscillator(df_is['Close'], period=fixed_cmo_p)

            osc_scale_factors = np.linspace(osc_scale_range[0], osc_scale_range[1], 10)

            returns_list = []
            portfolios_dict_1d = {}

            progress_bar = st.progress(0)
            total_combinations = len(osc_scale_factors)
            current_combination = 0

            # Common portfolio args for IS
            portfolio_common_kwargs_is = {
                'close': df_is['Close'], 'entries': entries_is, 'exits': exits_is, # Use IS signals
                'short_entries': short_entries_is if allow_shorting else None,
                'short_exits': short_exits_is if allow_shorting else None,
                'freq': df_is.index.freq, 'init_cash': initial_capital, 'fees': 0.001
            }
            if use_trailing_stop and tsl_pct > 0:
                 portfolio_common_kwargs_is['sl_stop'] = tsl_pct

            try:
                for scale_factor in osc_scale_factors:
                    current_combination += 1
                    progress_bar.progress(current_combination / total_combinations,
                                          text=f"IS Testing Osc Scale Factor={scale_factor:.2f} ({current_combination}/{total_combinations})")

                    # Calculate Order Size based on df_is data
                    base_value = initial_capital * osc_base_size_pct
                    clipped_osc = cmo_values_is.abs().clip(0, osc_clip_value).fillna(0)
                    clip_divisor = osc_clip_value if osc_clip_value > 0 else 1.0
                    scaling_effect = scale_factor * (clipped_osc / clip_divisor)
                    target_invest_value = base_value * (1 + scaling_effect)

                    order_size_shares_is = np.zeros_like(df_is['Close'], dtype=float)
                    entry_points_is = np.where(entries_is)[0]
                    if len(entry_points_is) > 0:
                        entry_prices = df_is['Close'].iloc[entry_points_is].where(lambda x: x!= 0, 1e-9)
                        target_shares_long = (target_invest_value.iloc[entry_points_is] / entry_prices).fillna(0).astype(int)
                        order_size_shares_is[entry_points_is] += target_shares_long

                    if allow_shorting:
                         short_entry_points_is = np.where(short_entries_is)[0]
                         if len(short_entry_points_is) > 0:
                              entry_prices_short = df_is['Close'].iloc[short_entry_points_is].where(lambda x: x!= 0, 1e-9)
                              target_shares_short = (target_invest_value.iloc[short_entry_points_is] / entry_prices_short).fillna(0).astype(int)
                              order_size_shares_is[short_entry_points_is] += target_shares_short

                    if np.all(order_size_shares_is == 0):
                        returns_list.append(np.nan)
                        continue

                    # Run IS portfolio simulation
                    pf_is = vbt.Portfolio.from_signals(
                        size=order_size_shares_is, # IS order sizes
                        **portfolio_common_kwargs_is # IS common args
                    )

                    current_return = pf_is.total_return()
                    returns_list.append(current_return)
                    portfolios_dict_1d[scale_factor] = pf_is # Store IS portfolio

                progress_bar.empty()
                returns_array = np.array(returns_list)
                if np.isnan(returns_array).all():
                     st.warning("In-Sample オシレータ基準最適化では有効なリターンが得られませんでした。")
                     return None, None, None, None, None

                best_idx = np.nanargmax(returns_array)
                best_osc_scale_factor = osc_scale_factors[best_idx]
                best_params = {'osc_scale_factor': best_osc_scale_factor}
                best_return_is = returns_array[best_idx]
                st.success(f"In-Sample オシレータ基準最適化完了. Best Scale Factor={best_osc_scale_factor:.2f} (IS Return: {best_return_is:.2%})")

                # Generate IS 1D results plot
                fig_results_plot_is, ax_results = plt.subplots(figsize=(10, 6))
                ax_results.plot(osc_scale_factors, returns_array * 100)
                ax_results.set_xlabel('Oscillator Scale Factor')
                ax_results.set_ylabel('Total Return (%)')
                ax_results.set_title('In-Sample Total Return vs Oscillator Scale Factor')
                ax_results.grid(True)
                plt.tight_layout()
                fig_optimization_plot = fig_results_plot_is # Assign IS plot
                best_pf_is = portfolios_dict_1d.get(best_osc_scale_factor) # Get best IS portfolio

            except Exception as e:
                st.error(f"Error during In-Sample oscillator sizing optimization loop: {e}")
                st.error(traceback.format_exc())
                if 'progress_bar' in locals(): progress_bar.empty()
                return None, None, None, None, None
            # --- End of Oscillator Sizing Logic (IS) ---

        elif sizing_algorithm == "ボリンジャーバンド幅基準":
            st.warning(f"取引量決定アルゴリズム「{sizing_algorithm}」は未実装です。")
            return None, None, None, None, None

        elif sizing_algorithm == "資金管理基準 (固定リスク率)":
             st.warning(f"取引量決定アルゴリズム「{sizing_algorithm}」は未実装です。")
             return None, None, None, None, None

        else:
            st.warning(f"取引量決定アルゴリズム「{sizing_algorithm}」は未実装です。")
            return None, None, None, None, None
        # --- End of Sizing Algorithm Branching (IS) ---
    # --- End of Trading Volume Optimization Logic (IS) ---

    else:
        st.error(f"未定義の最適化モードです: {optimization_mode}")
        return None, None, None, None, None
    # ==========================================================================
    # --- Check if Optimization was Successful ---
    # ==========================================================================
    if best_params is None:
         st.error("In-Sample 最適化で有効なパラメータが見つかりませんでした。Out-of-Sample 検証に進めません。")
         # Return only the IS optimization plot if it exists
         return fig_optimization_plot, None, None, None, None

    if best_pf_is is None:
         st.warning("In-Sample 最適化で最良ポートフォリオが見つかりませんでした。")
         # We have best_params, but no best_pf_is. Can still proceed to OOS validation.
         # Might indicate the best params resulted in no trades in IS.

    # ==========================================================================
    # --- Out-of-Sample Validation ---
    # ==========================================================================
    st.write("---")
    st.write(f"最適パラメータ {best_params} を使用して Out-of-Sample 検証を実行中...")

    pf_oos = None
    stats_oos = None
    fig_pf_oos = None
    accurate_position_size_oos = None

    try:
        # 1. Generate Signals/Indicators for OOS using best_params
        if optimization_mode == "シグナルパラメータ最適化":
            entries_oos, exits_oos, short_entries_oos, short_exits_oos = generate_signals(
                df_oos['Close'], best_params['cmo_period'], best_params['trix_period']
            )
            order_size_oos = None # No specific sizing in this mode (vectorbt defaults)
            size_kwarg_oos = {}

        elif optimization_mode == "取引量決定最適化":
             # Use fixed signal params for OOS signals as well
             entries_oos, exits_oos, short_entries_oos, short_exits_oos = generate_signals(
                 df_oos['Close'], fixed_cmo_p, fixed_trix_p
             )

             # Calculate Order Size for OOS using best_params and df_oos data
             if sizing_algorithm == "ボラティリティ基準":
                  atr_oos = talib.ATR(df_oos['High'], df_oos['Low'], df_oos['Close'], timeperiod=best_params['atr_period'])
                  atr_safe_oos = atr_oos.fillna(method='ffill').fillna(method='bfill').fillna(1e-6).where(atr_oos != 0, 1e-6)
                  risk_per_trade_oos = initial_capital * (best_params['target_vol_pct'] / 100.0) # Use best target vol

                  order_size_oos = np.zeros_like(df_oos['Close'], dtype=float)
                  entry_points_oos = np.where(entries_oos)[0]
                  if len(entry_points_oos) > 0:
                       atr_at_entry_oos = atr_safe_oos.iloc[entry_points_oos]
                       size_at_entry_oos = (risk_per_trade_oos / atr_at_entry_oos).fillna(0).astype(int)
                       order_size_oos[entry_points_oos] += size_at_entry_oos
                  if allow_shorting:
                       short_entry_points_oos = np.where(short_entries_oos)[0]
                       if len(short_entry_points_oos) > 0:
                            atr_at_short_entry_oos = atr_safe_oos.iloc[short_entry_points_oos]
                            size_at_short_entry_oos = (risk_per_trade_oos / atr_at_short_entry_oos).fillna(0).astype(int)
                            order_size_oos[short_entry_points_oos] += size_at_short_entry_oos
                  size_kwarg_oos = {'size': order_size_oos} # Use absolute size

             elif sizing_algorithm == "オシレータ基準":
                  cmo_values_oos = chande_momentum_oscillator(df_oos['Close'], period=fixed_cmo_p) # Fixed CMO period
                  base_value_oos = initial_capital * osc_base_size_pct
                  clipped_osc_oos = cmo_values_oos.abs().clip(0, osc_clip_value).fillna(0)
                  clip_divisor = osc_clip_value if osc_clip_value > 0 else 1.0
                  scaling_effect_oos = best_params['osc_scale_factor'] * (clipped_osc_oos / clip_divisor) # Use best scale factor
                  target_invest_value_oos = base_value_oos * (1 + scaling_effect_oos)

                  order_size_oos = np.zeros_like(df_oos['Close'], dtype=float)
                  entry_points_oos = np.where(entries_oos)[0]
                  if len(entry_points_oos) > 0:
                      entry_prices_oos = df_oos['Close'].iloc[entry_points_oos].where(lambda x: x!= 0, 1e-9)
                      target_shares_long_oos = (target_invest_value_oos.iloc[entry_points_oos] / entry_prices_oos).fillna(0).astype(int)
                      order_size_oos[entry_points_oos] += target_shares_long_oos
                  if allow_shorting:
                      short_entry_points_oos = np.where(short_entries_oos)[0]
                      if len(short_entry_points_oos) > 0:
                          entry_prices_short_oos = df_oos['Close'].iloc[short_entry_points_oos].where(lambda x: x!= 0, 1e-9)
                          target_shares_short_oos = (target_invest_value_oos.iloc[short_entry_points_oos] / entry_prices_short_oos).fillna(0).astype(int)
                          order_size_oos[short_entry_points_oos] += target_shares_short_oos
                  size_kwarg_oos = {'size': order_size_oos} # Use absolute size
             else:
                 # Handle unimplemented sizing algorithms if needed, or assume default sizing
                 st.warning(f"Out-of-Sample 検証のための取引量計算は {sizing_algorithm} では未実装です。デフォルトの取引量を使用します。")
                 order_size_oos = None
                 size_kwarg_oos = {}
        else:
             # Should not happen due to initial mode check
             entries_oos, exits_oos, short_entries_oos, short_exits_oos = (None, None, None, None)
             order_size_oos = None
             size_kwarg_oos = {}


        # 2. Check if any signals exist for OOS period
        if not np.any(entries_oos) and (not allow_shorting or not np.any(short_entries_oos)):
            st.warning("Out-of-Sample 期間で有効なエントリーシグナルが生成されませんでした。")
            # Still return IS results, but OOS results will be None
            return fig_optimization_plot, None, None, best_params, None

        # 3. Run Portfolio Simulation for OOS
        portfolio_kwargs_oos = {
            'close': df_oos['Close'],
            'entries': entries_oos,
            'exits': exits_oos,
            'freq': df_oos.index.freq,
            'init_cash': initial_capital,
            'fees': 0.001,
            **size_kwarg_oos # Add size argument if calculated
        }
        if allow_shorting:
            portfolio_kwargs_oos['short_entries'] = short_entries_oos
            portfolio_kwargs_oos['short_exits'] = short_exits_oos
        if use_trailing_stop and tsl_pct > 0:
            portfolio_kwargs_oos['sl_stop'] = tsl_pct

        pf_oos = vbt.Portfolio.from_signals(**portfolio_kwargs_oos)

        # 4. Calculate OOS Statistics
        if pf_oos.trades.count() > 0:
            stats_oos = pf_oos.stats()
            st.success("Out-of-Sample 検証完了.")
        else:
            st.info("Out-of-Sample 検証期間中にトレードは実行されませんでした。")
            stats_oos = None # No stats if no trades

        # 5. Calculate Accurate Position Size for OOS
        accurate_position_size_oos = pd.Series(0.0, index=df_oos.index) # Initialize for OOS index
        if pf_oos is not None and not pf_oos.trades.records.empty:
            try:
                records_df_oos = pf_oos.trades.records
                # Use df_oos.index for mapping indices
                entry_indices_oos = df_oos.index[records_df_oos['entry_idx'].astype(int)]
                exit_indices_oos = df_oos.index[records_df_oos['exit_idx'].astype(int)] # Potential issue if index not aligned?
                entry_adj_size_oos = records_df_oos['size'] * np.where(records_df_oos['direction'] == 0, 1, -1)
                entry_changes_oos = pd.Series(entry_adj_size_oos.values, index=entry_indices_oos)

                closed_trades_oos = records_df_oos[records_df_oos['status'] == 1]
                if not closed_trades_oos.empty:
                    closed_exit_indices_oos = df_oos.index[closed_trades_oos['exit_idx'].astype(int)]
                    # Ensure index alignment for subtraction
                    exit_adj_size_closed_oos = -closed_trades_oos['size'].reindex(closed_exit_indices_oos.unique()).fillna(0) * np.where(closed_trades_oos['direction'].reindex(closed_exit_indices_oos.unique()).fillna(0) == 0, 1, -1)
                    exit_changes_oos = pd.Series(exit_adj_size_closed_oos.values, index=closed_exit_indices_oos)
                else:
                    exit_changes_oos = pd.Series(dtype=float)

                all_changes_oos = pd.concat([entry_changes_oos, exit_changes_oos])
                if not all_changes_oos.empty:
                    net_changes_oos = all_changes_oos.groupby(all_changes_oos.index).sum()
                    aligned_changes_oos, _ = net_changes_oos.align(accurate_position_size_oos, fill_value=0.0)
                    accurate_position_size_oos = aligned_changes_oos.cumsum()
                else:
                     accurate_position_size_oos = pd.Series(0.0, index=df_oos.index) # Ensure it covers OOS period

            except Exception as calc_err_oos:
                st.warning(f"Out-of-Sample ポジションサイズの計算中にエラー: {calc_err_oos}")
                accurate_position_size_oos = None
        else:
             accurate_position_size_oos = pd.Series(0.0, index=df_oos.index) # Ensure it covers OOS period if no trades

        # 6. Generate OOS Portfolio Plot
        if pf_oos is not None and pf_oos.trades.count() > 0:
            try:
                fig_pf_oos = pf_oos.plot(subplots=['orders', 'trade_pnl', 'cum_returns'])
            except Exception as plot_err_oos:
                 st.warning(f"Out-of-Sample ポートフォリオプロットを生成できませんでした: {plot_err_oos}")
                 fig_pf_oos = None
        else:
            fig_pf_oos = None # No plot if no trades

    except Exception as oos_err:
         st.error(f"Error during Out-of-Sample validation: {oos_err}")
         st.error(traceback.format_exc())
         # Return IS results, OOS results will be None
         return fig_optimization_plot, None, None, best_params, None

    # --- Final Return ---
    return fig_optimization_plot, fig_pf_oos, stats_oos, best_params, accurate_position_size_oos 
import streamlit as st
import pandas as pd # Placeholder import
import numpy as np

# Import app modules
from src.apps import basic_analysis_app # Import the specific module
from src.apps import prophet_app # Import prophet_app
from src.apps import backtest_app # Import backtest_app
# Import data fetching function
from src.core.data.yahoo.fetch_from_yahoo import fetch_stock_data

# --- Page Configuration ---
st.set_page_config(
    page_title="Finance Analysis Dashboard",
    page_icon="💹",
    layout="wide"
)

# --- Sidebar ---
st.sidebar.title("分析設定")

# Analysis selection
analysis_choice = st.sidebar.radio(
    "分析の種類を選択してください:",
    ("基本分析", "Prophet予測", "バックテスト")
)

# Common parameters
st.sidebar.header("共通パラメータ")
symbol = st.sidebar.text_input("銘柄コード", "AAPL")
period = st.sidebar.selectbox(
    "データ取得期間",
    ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max'],
    index=6 # Default '2y'
)
interval = st.sidebar.selectbox(
    "データ間隔",
    ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'],
    index=8 # Default '1d'
)

# --- Data Loading Function ---
@st.cache_data # Cache the fetched data based on inputs
def load_data(symbol, period, interval):
    """Fetches and caches stock and fundamental data."""
    try:
        df, fundamental_data = fetch_stock_data(symbol, period=period, interval=interval)
        if df is None or df.empty:
            st.error(f"Failed to fetch stock data for {symbol}. Check symbol or try again later.")
            return None, None
        return df.copy(), fundamental_data # Return copy to avoid cache mutation issues
    except Exception as e:
        st.error(f"An error occurred during data fetching: {e}")
        return None, None

# --- Main Area ---
st.title(f"{symbol} - {analysis_choice}")

# Load data once
st.subheader("データ読み込み")
with st.spinner(f'{symbol}のデータを読み込み中 ({period}, {interval})...'):
    df_loaded, fundamental_data_loaded = load_data(symbol, period, interval)

# Proceed only if data is loaded successfully
if df_loaded is not None:
    st.success("データ読み込み完了")
    st.dataframe(df_loaded.head()) # Show a preview of the loaded data

    # --- Analysis Execution based on Choice ---
    if analysis_choice == "基本分析":
        st.header("基本分析結果")
        with st.spinner('基本分析を実行中...'):
            # Pass loaded data to the function
            fig_price, fig_tech, fundamental_dict_processed = basic_analysis_app.run_basic_analysis(
                df_loaded, fundamental_data_loaded, symbol # Pass df, fundamental_data, symbol
            )
        # Display results (Keep existing display logic)
        if fig_price:
            st.subheader("価格と出来高")
            st.pyplot(fig_price)
        else:
            st.warning("価格チャートを生成できませんでした。")

        if fig_tech:
            st.subheader("テクニカル指標")
            st.pyplot(fig_tech)
        else:
            st.warning("テクニカル指標チャートを生成できませんでした。")

        if fundamental_dict_processed: # Use the processed dict from the function
            st.subheader("ファンダメンタル指標")
            # Display fundamental data
            if 'meta' in fundamental_dict_processed:
                st.write(f"**企業名:** {fundamental_dict_processed['meta'].get('company_name', 'N/A')}")
                st.write(f"**市場:** {fundamental_dict_processed['meta'].get('market', 'N/A')}")
                st.write(f"**通貨:** {fundamental_dict_processed['meta'].get('currency', 'N/A')}")

            if 'latest' in fundamental_dict_processed:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**バリュエーション**")
                    valuation_df = pd.DataFrame.from_dict(fundamental_dict_processed['latest'].get('valuation', {}), orient='index')
                    if not valuation_df.empty:
                        st.table(valuation_df[['value', 'unit']])
                    else:
                        st.write("データなし")

                    st.write("**財務指標**")
                    financial_df = pd.DataFrame.from_dict(fundamental_dict_processed['latest'].get('financial', {}), orient='index')
                    if not financial_df.empty:
                         financial_df['value_str'] = financial_df['value'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else 'N/A')
                         st.table(financial_df[['value_str', 'unit']].rename(columns={'value_str': 'Value'}))
                    else:
                        st.write("データなし")
                with col2:
                    st.write("**経営指標**")
                    indicators_df = pd.DataFrame.from_dict(fundamental_dict_processed['latest'].get('indicators', {}), orient='index')
                    if not indicators_df.empty:
                        st.table(indicators_df[['value', 'unit']])
                    else:
                        st.write("データなし")

                    st.write("**成長性指標**")
                    growth_df = pd.DataFrame.from_dict(fundamental_dict_processed['latest'].get('growth', {}), orient='index')
                    if not growth_df.empty:
                        st.table(growth_df[['value', 'unit']])
                    else:
                        st.write("データなし")

            if 'quarterly' in fundamental_dict_processed and fundamental_dict_processed['quarterly']:
                 st.write("**四半期データ**")
                 if 'dates' in fundamental_dict_processed['quarterly']:
                     for col, data in fundamental_dict_processed['quarterly'].items():
                         if col != 'dates' and isinstance(data, dict) and 'values' in data:
                            if len(fundamental_dict_processed['quarterly']['dates']) == len(data['values']):
                                 metric_df = pd.DataFrame({
                                     'Value': data['values']
                                 }, index=fundamental_dict_processed['quarterly']['dates'])
                                 metric_df['Unit'] = data.get('unit', '')
                                 try:
                                    numeric_values = pd.to_numeric(metric_df['Value'], errors='coerce')
                                    formatted_values = numeric_values.apply(lambda x: f"{x:,.0f}" if pd.notna(x) else 'N/A')
                                    if not formatted_values.isna().all():
                                        metric_df['Value'] = formatted_values
                                 except Exception:
                                     pass
                                 st.write(f"*{col}*")
                                 st.dataframe(metric_df)
                            else:
                                st.warning(f"Skipping quarterly data for '{col}' due to mismatched lengths.")
                 else:
                    st.warning("Quarterly data is missing 'dates' information.")
        elif fundamental_data_loaded is None and fig_price is not None: # Check loaded fundamental data
            st.info("ファンダメンタルデータを取得できませんでした。")
        elif fundamental_dict_processed is None and fundamental_data_loaded is not None:
             st.warning("取得したファンダメンタルデータの処理中にエラーが発生しました。")

    elif analysis_choice == "Prophet予測":
        st.header("Prophet予測結果")
        st.sidebar.header("Prophet予測パラメータ")
        forecast_days = st.sidebar.number_input("予測日数", min_value=30, max_value=365*5, value=365, step=30)

        with st.spinner(f'Prophet予測を実行中 ({forecast_days}日間)...'):
            # Pass loaded df to the function
            fig_test, fig_future, metrics = prophet_app.run_prophet_forecast(
                df_loaded, symbol, forecast_days # Pass df, symbol, forecast_days
            )

        if metrics:
            st.subheader("テストデータ評価メトリクス")
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{metrics.get('Test RMSE', 'N/A'):.4f}")
            col2.metric("MAE", f"{metrics.get('Test MAE', 'N/A'):.4f}")
            col3.metric("R2 Score", f"{metrics.get('Test R2', 'N/A'):.4f}")

        if fig_test:
            st.subheader("テストデータ期間の予測 vs 実績")
            st.pyplot(fig_test)
        else:
            st.warning("テストデータの予測プロットを生成できませんでした。")

        if fig_future:
            st.subheader(f"将来 {forecast_days} 日間の予測")
            st.pyplot(fig_future)
        else:
            st.warning("将来予測プロットを生成できませんでした。")

    elif analysis_choice == "バックテスト":
        st.header("バックテスト結果")
        
        # --- Backtest Sidebar ---
        st.sidebar.header("バックテスト設定") # New header

        optimization_mode = st.sidebar.radio(
            "最適化モード:",
            ("シグナルパラメータ最適化", "取引量決定最適化"),
            key="optim_mode",
            help="「シグナル」はCMO/Trixの期間を最適化します。「取引量決定」は（将来的に）取引サイズ決定ロジックを最適化します。"
        )

        # Parameters for Signal Optimization
        if optimization_mode == "シグナルパラメータ最適化":
            st.sidebar.subheader("シグナルパラメータ")
            cmo_range = st.sidebar.slider("CMO期間範囲", 2, 50, (5, 20), key="cmo_signal")
            trix_range = st.sidebar.slider("Trix期間範囲", 2, 50, (5, 20), key="trix_signal")
            # Set defaults for sizing mode parameters
            sizing_algorithm = None
            atr_p_range = None
            target_vol_range = None
            # Add defaults for oscillator params
            osc_scale_range = None
            osc_base_size_pct = 0.1 # Match default in function signature
            osc_clip_value = 50   # Match default in function signature
        
        # Parameters for Sizing Optimization
        elif optimization_mode == "取引量決定最適化":
            st.sidebar.subheader("取引量決定パラメータ")
            sizing_algorithm = st.sidebar.selectbox(
                "取引量決定アルゴリズム:",
                ("ボラティリティ基準", "オシレータ基準", "資金管理基準 (固定リスク率)"), 
                key="sizing_algo"
            )
            
            # Reset ranges initially
            atr_p_range = None
            target_vol_range = None
            osc_scale_range = None
            # Set defaults for signal mode parameters
            cmo_range = (14,14) # Default fixed range
            trix_range = (14,14) # Default fixed range
            
            if sizing_algorithm == "ボラティリティ基準":
                st.sidebar.write("ボラティリティ基準パラメータ:")
                atr_p_range = st.sidebar.slider("ATR期間範囲", 5, 100, (10, 30), key="atr_p_range")
                target_vol_range_pct = st.sidebar.slider("目標ボラティリティ範囲 (%) Daily", 0.1, 10.0, (0.5, 2.0), step=0.1, format="%.1f%%", key="target_vol_range")
                target_vol_range = (target_vol_range_pct[0], target_vol_range_pct[1])
                # Set unused params to defaults expected by function signature if needed (or None)
                osc_base_size_pct = 0.1 
                osc_clip_value = 50
            elif sizing_algorithm == "オシレータ基準":
                st.sidebar.write("オシレータ基準パラメータ:")
                osc_scale_range = st.sidebar.slider("スケール係数範囲", 0.0, 5.0, (0.5, 2.0), step=0.1, key="osc_scale_range")
                osc_base_size_pct = st.sidebar.number_input("ベースサイズ (%)", 1.0, 100.0, 10.0, step=1.0, key="osc_base_pct") / 100.0 # Convert to decimal
                osc_clip_value = st.sidebar.number_input("CMOクリップ値 (絶対値)", 1.0, 100.0, 50.0, step=1.0, key="osc_clip")
                # Set unused params to None or defaults
                # atr_p_range = None
                # target_vol_range = None
            else:
                 st.sidebar.warning(f"アルゴリズム「{sizing_algorithm}」のパラメータ設定は未実装です。")
                 # Ensure all sizing specific ranges are None
                 atr_p_range = None
                 target_vol_range = None
                 osc_scale_range = None
                 # Use default fixed values for base/clip if needed, or handle in function
                 osc_base_size_pct = 0.1
                 osc_clip_value = 50

        # Common Parameters for both modes (for now)
        st.sidebar.subheader("共通パラメータ")
        initial_capital = st.sidebar.number_input("初期資金", min_value=1000, value=100000, step=1000, key="init_capital")

        # Strategy Options common to both modes
        st.sidebar.subheader("戦略オプション")
        allow_shorting = st.sidebar.checkbox("空売りを許可する", value=False, key="allow_short")
        use_trailing_stop = st.sidebar.checkbox("トレーリングストップを使用する", value=False, key="use_tsl")
        tsl_pct_input = 0.0 # Default value if not used
        if use_trailing_stop:
            tsl_pct_input = st.sidebar.number_input(
                "トレーリングストップ (%)",
                min_value=0.1,
                max_value=50.0,
                value=5.0, # Default 5%
                step=0.1,
                format="%.1f",
                key="tsl_pct"
            ) / 100.0 # Convert percentage to decimal

        # --- Backtest Execution ---
        if st.button("バックテスト実行"):
            # Update spinner message for oscillator mode
            if optimization_mode == "シグナルパラメータ最適化":
                spinner_message = f'シグナルパラメータ最適化を実行中 (CMO: {cmo_range}, Trix: {trix_range})...'
            elif sizing_algorithm == "ボラティリティ基準":
                spinner_message = f'ボラティリティ基準 取引量最適化を実行中 (ATR: {atr_p_range}, Vol: {target_vol_range})...'
            elif sizing_algorithm == "オシレータ基準":
                spinner_message = f'オシレータ基準 取引量最適化を実行中 (Scale: {osc_scale_range}, Base: {osc_base_size_pct:.1%}, Clip: {osc_clip_value:.0f})...'
            else:
                spinner_message = "バックテスト実行中 (未実装アルゴリズム選択)..."

            with st.spinner(spinner_message):
                # Retrieve fixed signal params from session state if in sizing mode
                fixed_signal_params_to_pass = None
                if optimization_mode == "取引量決定最適化":
                    fixed_signal_params_to_pass = st.session_state.get('best_signal_params', None)
                    if fixed_signal_params_to_pass is None:
                        # Display warning only if trying to run sizing without prior signal optim
                        st.warning("シグナル最適化が未実行か失敗したため、デフォルトのシグナルパラメータ（CMO=14, Trix=14）を使用します。", icon="⚠️")
                    else:
                        # Display info if params are loaded successfully
                         st.info(f"保存された最適シグナルパラメータを使用: CMO={fixed_signal_params_to_pass.get('cmo_period', 'N/A')}, Trix={fixed_signal_params_to_pass.get('trix_period', 'N/A')}", icon="ℹ️")
                        
                # Call the backtest function
                results_plot, fig_pf, stats, best_params_from_run, position_size = backtest_app.run_backtest_optimization(
                    df_loaded, symbol,
                    cmo_range=cmo_range, 
                    trix_range=trix_range,
                    atr_p_range=atr_p_range,
                    target_vol_range=target_vol_range,
                    osc_scale_range=osc_scale_range,
                    osc_base_size_pct=osc_base_size_pct,
                    osc_clip_value=osc_clip_value,
                    initial_capital=initial_capital,
                    allow_shorting=allow_shorting,
                    use_trailing_stop=use_trailing_stop,
                    tsl_pct=tsl_pct_input,
                    optimization_mode=optimization_mode,
                    sizing_algorithm=sizing_algorithm,
                    fixed_signal_params=fixed_signal_params_to_pass # Pass retrieved or None params
                )
            
            # Store best signal params in session state IF signal optim was successful
            if optimization_mode == "シグナルパラメータ最適化" and best_params_from_run:
                 st.session_state['best_signal_params'] = best_params_from_run
                 st.success("最適シグナルパラメータをセッションに保存しました。")
                 # Clear potentially stored params from other modes
                 # (Optional, but good practice if modes change params keys)
                 # if 'best_sizing_params' in st.session_state: del st.session_state['best_sizing_params'] 
            elif optimization_mode == "取引量決定最適化" and best_params_from_run:
                 # Store best sizing params (optional)
                 # st.session_state['best_sizing_params'] = best_params_from_run
                 pass # Or clear signal params if sizing was run?

            # --- Display results ---
            # Rename the variable for clarity in display logic
            best_params = best_params_from_run 
            if best_params:
                st.subheader("最適化結果")
                if optimization_mode == "シグナルパラメータ最適化":
                    st.write(f"**最適シグナルパラメータ:** CMO={best_params['cmo_period']}, Trix={best_params['trix_period']}")
                elif optimization_mode == "取引量決定最適化":
                    st.write(f"**最適取引量パラメータ ({sizing_algorithm}):**")
                    if sizing_algorithm == "ボラティリティ基準":
                        st.write(f"  ATR Period = {best_params.get('atr_period', 'N/A')}")
                        st.write(f"  Target Volatility = {best_params.get('target_vol_pct', 'N/A'):.2f}%")
                    elif sizing_algorithm == "オシレータ基準":
                         st.write(f"  Scale Factor = {best_params.get('osc_scale_factor', 'N/A'):.2f}")
                         # Display fixed params used
                         st.caption(f" (使用した固定値: Base Size={osc_base_size_pct:.1%}, Clip Value={osc_clip_value:.0f})")
                    else:
                         st.write("未実装アルゴリズムのパラメータ表示")

            # Display the appropriate plot (heatmap or 1D plot)
            if results_plot: # Renamed variable to handle both plot types
                if optimization_mode == "シグナルパラメータ最適化" or sizing_algorithm == "ボラティリティ基準":
                    heatmap_title = "シグナルパラメータ別リターン (ヒートマップ)" if optimization_mode == "シグナルパラメータ最適化" else f"取引量パラメータ別リターン ({sizing_algorithm} - ヒートマップ)"
                    st.subheader(heatmap_title)
                elif sizing_algorithm == "オシレータ基準":
                    st.subheader("リターン vs オシレータスケール係数")
                # Display the plot using st.pyplot
                st.pyplot(results_plot)
            else:
                 if best_params:
                     st.warning("結果プロットを生成できませんでした。")

            if stats is not None:
                st.subheader("最適パラメータでのバックテスト統計")
                stats_df = stats.to_frame(name='Value')
                
                # Define columns that represent percentages and should end with %
                pct_indices = [
                    'Total Return [%]', 'Benchmark Return [%]', 'Max Drawdown [%]', 
                    'Best Trade [%]', 'Worst Trade [%]', 'Avg Winning Trade [%]', 
                    'Avg Losing Trade [%]', 'Win Rate [%]'
                ]
                # Define columns that are ratios/factors and need specific decimal places
                ratio_indices = [
                    'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Omega Ratio', 
                    'Profit Factor', 'Beta', 'Alpha' 
                ]
                
                # Loop through index to apply specific formatting
                for idx in stats_df.index:
                    value = stats_df.loc[idx, 'Value']
                    try:
                        # Check if value is numeric before formatting
                        if isinstance(value, (int, float, np.number)):
                            if idx in pct_indices:
                                # Format as float with 2 decimals and add %
                                stats_df.loc[idx, 'Value'] = f"{value:.2f}%"
                            elif idx in ratio_indices:
                                # Format as float with 3 decimals
                                stats_df.loc[idx, 'Value'] = f"{value:.3f}"
                            elif 'Value' in idx or 'PnL' in idx or 'Paid' in idx:
                                 stats_df.loc[idx, 'Value'] = f"{value:,.2f}"
                                 
                        elif 'Date' in idx or 'Period' in idx or 'Duration' in idx:
                             if isinstance(value, pd.Timestamp):
                                 stats_df.loc[idx, 'Value'] = value.strftime('%Y-%m-%d')
                             elif isinstance(value, pd.Timedelta):
                                 stats_df.loc[idx, 'Value'] = str(value)
                        
                    except (TypeError, ValueError, AttributeError) as fmt_err:
                         pass 
                
                stats_df['Value'] = stats_df['Value'].astype(str) 
                st.dataframe(stats_df, height=600)
            else:
                 if best_params:
                     st.warning("バックテスト統計を生成できませんでした。")

            if fig_pf:
                st.subheader("最適パラメータでのポートフォリオ詳細")
                st.plotly_chart(fig_pf, use_container_width=True)
            else:
                if best_params:
                    st.warning("ポートフォリオプロットを生成できませんでした。")
            
            # Add Position Size Plot
            if position_size is not None and not position_size.empty:
                st.subheader("正確なポジションサイズ推移")
                st.line_chart(position_size)
                st.caption("ポジションサイズは取引履歴 (trades.records) から正確に計算されています。")
            else:
                # Only show warning if backtest ran successfully but position couldn't be plotted
                if best_params:
                    st.info("ポジションサイズの推移は表示できませんでした。")

        else:
            st.info("サイドバーでパラメータを設定し、「バックテスト実行」ボタンを押してください。")

    else:
        st.error("無効な分析タイプが選択されました。")
else:
     st.error("データの読み込みに失敗しました。分析を実行できません。")

# Add footer or other common elements if needed
# st.write("---")
# st.caption("データソース: Yahoo Finance") 
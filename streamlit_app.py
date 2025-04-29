import streamlit as st
import pandas as pd # Placeholder import

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
        st.sidebar.header("バックテストパラメータ")
        cmo_range = st.sidebar.slider("CMO期間範囲", 2, 50, (5, 20))
        trix_range = st.sidebar.slider("Trix期間範囲", 2, 50, (5, 20))
        initial_capital = st.sidebar.number_input("初期資金", min_value=1000, value=100000, step=1000)

        if st.button("バックテスト実行"):
            with st.spinner(f'バックテスト最適化を実行中 (CMO: {cmo_range}, Trix: {trix_range})...'):
                # Pass loaded df to the function
                fig_heatmap, fig_pf, stats, best_params = backtest_app.run_backtest_optimization(
                    df_loaded, symbol, cmo_range, trix_range, initial_capital # Pass df, symbol, ranges, capital
                )

            if best_params:
                st.subheader("最適化結果")
                st.write(f"**最適パラメータ:** CMO={best_params['cmo_period']}, Trix={best_params['trix_period']}")

            if fig_heatmap:
                st.subheader("パラメータ別リターン (ヒートマップ)")
                st.pyplot(fig_heatmap)
            else:
                st.warning("リターンヒートマップを生成できませんでした。")

            if stats is not None:
                st.subheader("最適パラメータでのバックテスト統計")
                stats_df = stats.to_frame(name='Value')
                for idx in stats_df.index:
                    if 'Return' in idx or 'Sharpe' in idx or 'Sortino' in idx or 'Calmar' in idx or 'Max Drawdown' in idx or 'Profit Factor' in idx:
                        try:
                             if isinstance(stats_df.loc[idx, 'Value'], (int, float)):
                                 stats_df.loc[idx, 'Value'] = f"{stats_df.loc[idx, 'Value']:.2%}" if 'Return' in idx or 'Drawdown' in idx else f"{stats_df.loc[idx, 'Value']:.3f}"
                        except (TypeError, ValueError):
                            pass
                    elif 'Date' in idx or 'Duration' in idx:
                         try:
                             if isinstance(stats_df.loc[idx, 'Value'], pd.Timestamp):
                                 stats_df.loc[idx, 'Value'] = stats_df.loc[idx, 'Value'].strftime('%Y-%m-%d')
                             elif isinstance(stats_df.loc[idx, 'Value'], pd.Timedelta):
                                 stats_df.loc[idx, 'Value'] = str(stats_df.loc[idx, 'Value'])
                         except AttributeError:
                             pass
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
        else:
            st.info("サイドバーでパラメータを設定し、「バックテスト実行」ボタンを押してください。")

    else:
        st.error("無効な分析タイプが選択されました。")
else:
     st.error("データの読み込みに失敗しました。分析を実行できません。")

# Add footer or other common elements if needed
# st.write("---")
# st.caption("データソース: Yahoo Finance") 
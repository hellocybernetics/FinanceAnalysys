"""
Comprehensive Financial Analysis Dashboard
統合財務分析ダッシュボード - Technical + Fundamental 分析の統合ビュー
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.services.technical_service import TechnicalAnalysisService
from src.services.fundamental_service import FundamentalAnalysisService
from src.core.models import (
    TechnicalAnalysisRequest,
    FundamentalAnalysisRequest,
    IndicatorConfig
)
from src.visualization.visualizer import Visualizer
from src.visualization.export_handler import ExportHandler
from src.utils.price_change import PriceChangeCalculator
from src.utils.timeframes import get_intervals_for_period

# ==================== プリセット銘柄セット ====================
PRESET_DATASETS = {
    "米国: Magnificent 7 + ETF": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "MAGS"  # 'MAGS' = Roundhill Magnificent Seven ETF
    ],
    "日本: 主要企業": [
        "7203.T",  # Toyota Motor
        "9984.T",  # SoftBank Group
        "8306.T",  # Mitsubishi UFJ Financial Group
        "6758.T",  # Sony Group
        "6501.T",  # Hitachi
        "6861.T",  # Keyence
        "8058.T",  # Mitsubishi Corporation
    ],
}
DEFAULT_PRESET = "米国: Magnificent 7 + ETF"


def apply_symbol_preset():
    """セッション状態の銘柄入力をプリセットに合わせて更新する。"""
    preset = st.session_state.get("symbol_preset")
    if preset in PRESET_DATASETS:
        st.session_state.symbols_input = ", ".join(PRESET_DATASETS[preset])
# ==================== ユーティリティ ====================
SIGNAL_COLOR_MAP = {
    'positive': '#059669',  # bullish green (matching candlestick up)
    'negative': '#dc2626',  # bearish red (matching candlestick down)
    'neutral': '#64748b',   # slate gray
}


def classify_signal_text(signal_text: str) -> tuple[str, str]:
    """Return (arrow, style) for a given textual signal."""
    if not signal_text:
        return '→', 'neutral'

    lower = signal_text.lower()
    bullish_keywords = ['bullish', 'buy', 'oversold', 'support', 'uptrend', 'long']
    bearish_keywords = ['bearish', 'sell', 'overbought', 'resistance', 'downtrend', 'short']

    if any(keyword in lower for keyword in bullish_keywords):
        return '↑', 'positive'
    if any(keyword in lower for keyword in bearish_keywords):
        return '↓', 'negative'
    if 'neutral' in lower or 'sideways' in lower or 'range' in lower:
        return '→', 'neutral'
    return '→', 'neutral'


def render_signal_line(name: str, description: str) -> None:
    arrow, style = classify_signal_text(description)
    color = SIGNAL_COLOR_MAP.get(style, '#546e7a')
    safe_desc = description or 'シグナル情報なし'
    st.markdown(
        f"""
        <div style="padding:4px 0;">
            <span style="color:{color}; font-weight:600;">{arrow} {name}</span><br>
            <span style="color:#334155; font-size:0.9em;">{safe_desc}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


def evaluate_pe_ratio(pe: float | None) -> tuple[str, str, str] | None:
    if pe is None or pe <= 0:
        return None
    if pe < 12:
        return '↑', 'positive', '割安 (PER < 12)'
    if pe <= 25:
        return '→', 'neutral', '適正レンジ (12-25)'
    return '↓', 'negative', '割高 (PER > 25)'


def evaluate_pb_ratio(pb: float | None) -> tuple[str, str, str] | None:
    if pb is None or pb <= 0:
        return None
    if pb < 1:
        return '↑', 'positive', '割安 (PBR < 1)'
    if pb <= 3:
        return '→', 'neutral', '適正レンジ (1-3)'
    return '↓', 'negative', '割高 (PBR > 3)'


def evaluate_dividend_yield(div_yield: float | None) -> tuple[str, str, str] | None:
    if div_yield is None or div_yield < 0:
        return None
    # div_yield is already in percentage format (0.4 = 0.4%)
    if div_yield >= 4:
        return '↑', 'positive', '高配当 (>4%)'
    if div_yield >= 1:
        return '→', 'neutral', '平均的 (1-4%)'
    return '↓', 'negative', '低配当 (<1%)'


def render_valuation_line(label: str, value_text: str, evaluation: tuple[str, str, str] | None) -> None:
    if evaluation:
        arrow, style, note = evaluation
        color = SIGNAL_COLOR_MAP.get(style, '#546e7a')
        st.markdown(
            f"""
            <div style="padding:4px 0;">
                <span style="color:{color}; font-weight:600;">{arrow} {label}: {value_text}</span><br>
                <span style="color:{color}; font-size:0.85em;">{note}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='padding:4px 0;'><span style='font-weight:600;'>{label}: {value_text}</span></div>",
            unsafe_allow_html=True
        )
# ==================== ページ設定 ====================
st.set_page_config(
    page_title="統合財務分析ダッシュボード",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== サービス初期化 ====================
@st.cache_resource
def get_services():
    """サービスインスタンスを取得（キャッシュ）"""
    return {
        'technical': TechnicalAnalysisService(),
        'fundamental': FundamentalAnalysisService(),
        'visualizer': Visualizer(),
        'export': ExportHandler()
    }

services = get_services()

# ==================== セッション状態初期化 ====================
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = None
if 'run_requested' not in st.session_state:
    st.session_state.run_requested = False
if 'indicator_configs' not in st.session_state:
    st.session_state.indicator_configs = [
        IndicatorConfig(name='SMA', params={'length': 20}),
        IndicatorConfig(name='RSI', params={'length': 14}),
        IndicatorConfig(name='MACD', params={'fast': 12, 'slow': 26, 'signal': 9}),
    ]
if 'price_change_period' not in st.session_state:
    st.session_state.price_change_period = None
if 'selected_interval' not in st.session_state:
    st.session_state.selected_interval = '1d'
if 'symbol_preset' not in st.session_state:
    st.session_state.symbol_preset = DEFAULT_PRESET
if 'symbols_input' not in st.session_state:
    st.session_state.symbols_input = ", ".join(PRESET_DATASETS[DEFAULT_PRESET])

# ==================== カスタムCSS ====================
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .positive {
        color: #00c853;
        font-weight: bold;
    }
    .negative {
        color: #ff1744;
        font-weight: bold;
    }
    .section-header {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        margin: 20px 0 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== サイドバー ====================
st.sidebar.title("📊 統合財務分析")
st.sidebar.markdown("---")

# シンボル入力
preset_options = ["カスタム"] + list(PRESET_DATASETS.keys())
current_preset = st.session_state.get("symbol_preset", DEFAULT_PRESET)
initial_index = preset_options.index(current_preset) if current_preset in preset_options else 0
st.sidebar.selectbox(
    "📚 プリセット銘柄セット",
    preset_options,
    index=initial_index,
    key="symbol_preset",
    on_change=apply_symbol_preset,
    help="プリセットを選ぶと下の入力欄が自動で更新されます。カスタムを選んだまま編集すれば自由に追加できます。"
)

symbols_input = st.sidebar.text_input(
    "📈 分析銘柄（カンマ区切り）",
    key="symbols_input",
    help="複数の銘柄をカンマで区切って入力してください（例: NVDA, 7203.T）。"
)
symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]

# 分析モード選択
st.sidebar.markdown("### 🎯 分析モード")
analysis_mode = st.sidebar.radio(
    "選択してください",
    ["🔍 統合分析", "⚖️ 銘柄比較"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# テクニカル分析パラメータ
if analysis_mode in ["🔍 統合分析", "⚖️ 銘柄比較"]:
    with st.sidebar.expander("📊 テクニカル分析設定", expanded=True):
        period_options = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"]
        default_period_index = period_options.index("1y")
        period = st.selectbox(
            "期間",
            period_options,
            index=default_period_index,
            key="selected_period"
        )

        available_intervals = get_intervals_for_period(period)
        stored_interval = st.session_state.get('selected_interval')
        if stored_interval not in available_intervals:
            stored_interval = available_intervals[0]
            st.session_state.selected_interval = stored_interval

        interval = st.selectbox(
            "間隔",
            available_intervals,
            index=available_intervals.index(stored_interval),
            key="selected_interval"
        )

# ファンダメンタル分析パラメータ
if analysis_mode in ["🔍 統合分析", "⚖️ 銘柄比較"]:
    with st.sidebar.expander("💼 ファンダメンタル分析設定", expanded=True):
        include_financials = st.checkbox("財務諸表を含む", value=True)
        include_ratios = st.checkbox("財務比率を計算", value=True)

st.sidebar.markdown("---")

# 実行ボタン
run_analysis = st.sidebar.button("🚀 分析実行", type="primary", use_container_width=True)
if run_analysis:
    st.session_state.analysis_results = None  # 前回の結果をクリア
    st.session_state.selected_symbol = symbols[0] if symbols else None
    st.session_state.run_requested = True

# ==================== ヘッダー ====================
st.title("📊 統合財務分析ダッシュボード")
st.markdown(f"**分析モード**: {analysis_mode} | **対象銘柄**: {', '.join(symbols) if symbols else '未選択'}")
st.markdown("---")

# ==================== メインコンテンツ ====================

if analysis_mode == "🔍 統合分析":
    st.markdown('<div class="section-header">🔍 統合分析 - Technical & Fundamental</div>', unsafe_allow_html=True)

    if not symbols:
        st.warning("⚠️ 分析する銘柄を入力してください")
    elif st.session_state.run_requested:
        with st.spinner("分析中..."):
            try:
                # 指標設定
                indicator_configs = [
                    IndicatorConfig(name='SMA', params={'length': 20}),
                    IndicatorConfig(name='RSI', params={'length': 14}),
                    IndicatorConfig(name='MACD', params={'fast': 12, 'slow': 26, 'signal': 9}),
                ]

                # Technical分析
                tech_request = TechnicalAnalysisRequest(
                    symbols=symbols,
                    period=period,
                    interval=interval,
                    indicators=indicator_configs,
                    use_cache=True
                )
                tech_results = services['technical'].analyze(tech_request)

                # Fundamental分析
                fund_request = FundamentalAnalysisRequest(
                    symbols=symbols,
                    include_financials=include_financials,
                    include_ratios=include_ratios
                )
                fund_results = services['fundamental'].analyze(fund_request)

                # 結果を保存
                st.session_state.analysis_results = {
                    'technical': {r.symbol: r for r in tech_results},
                    'fundamental': {r.symbol: r for r in fund_results}
                }
                st.session_state.indicator_configs = indicator_configs  # 指標設定も保存
                st.session_state.run_requested = False  # フラグをリセット

            except Exception as e:
                st.error(f"❌ 分析中にエラーが発生しました: {str(e)}")
                st.session_state.run_requested = False
    else:
        st.info("👈 サイドバーから「🚀 分析実行」ボタンを押してください")

    # 結果表示
    if st.session_state.analysis_results:
        # 銘柄選択タブ
        if len(symbols) > 1:
            symbol_tabs = st.tabs([f"📈 {sym}" for sym in symbols])
        else:
            symbol_tabs = [st.container()]

        for idx, symbol in enumerate(symbols):
            with symbol_tabs[idx]:
                tech_result = st.session_state.analysis_results['technical'].get(symbol)
                fund_result = st.session_state.analysis_results['fundamental'].get(symbol)

                if not tech_result or not fund_result:
                    st.error(f"❌ {symbol} のデータ取得に失敗しました")
                    continue

                # ========== 企業情報ヘッダー ==========
                # Row 1: 企業名とセクター
                st.markdown(f"## {symbol} - {fund_result.company_info.name}")
                if fund_result.company_info.sector:
                    st.caption(f"🏢 {fund_result.company_info.sector} | {fund_result.company_info.industry or 'N/A'}")

                # Row 2: 比較期間、変動、現在価格、時価総額
                header_col1, header_col2, header_col3, header_col4 = st.columns([1, 1, 1, 1])

                # 利用可能な変動期間を取得（価格変動計算のため事前に計算）
                available_periods = PriceChangeCalculator.get_available_periods(period, interval)

                # デフォルト値の設定
                if st.session_state.price_change_period is None:
                    st.session_state.price_change_period = PriceChangeCalculator.get_default_change_period(period, interval)

                # 変動期間選択ドロップダウン
                period_labels = [p["label"] for p in available_periods]
                period_values = [p["value"] for p in available_periods]

                # 現在の選択肢が利用可能なリストにない場合はデフォルトに戻す
                if st.session_state.price_change_period not in period_values:
                    st.session_state.price_change_period = period_values[0]

                current_index = period_values.index(st.session_state.price_change_period)

                with header_col1:
                    selected_label = st.selectbox(
                        "比較期間",
                        period_labels,
                        index=current_index,
                        key=f"price_change_{symbol}"
                    )

                    # 選択された値を取得
                    selected_period = period_values[period_labels.index(selected_label)]
                    st.session_state.price_change_period = selected_period

                # 動的に価格変動を計算
                price_change, price_change_pct = PriceChangeCalculator.calculate_price_change(
                    tech_result.data,
                    selected_period,
                    interval
                )

                # 価格変動がNoneの場合はデフォルト値を使用
                if price_change is None or price_change_pct is None:
                    price_change = tech_result.summary.price_change
                    price_change_pct = tech_result.summary.price_change_pct

                with header_col2:
                    # 変動額（変動率）を表示
                    change_sign = "+" if price_change >= 0 else ""
                    st.metric(
                        "変動",
                        f"{change_sign}{price_change:.2f}",
                        f"{price_change_pct:+.2f}%"
                    )

                with header_col3:
                    st.metric("現在価格", f"${tech_result.summary.latest_price:.2f}")

                with header_col4:
                    if fund_result.company_info.market_cap:
                        st.metric("時価総額", f"${fund_result.company_info.market_cap/1e9:.2f}B")

                st.markdown("---")

                # ========== メインコンテンツ：3カラム ==========
                col_left, col_center, col_right = st.columns([1, 2, 1])

                # 左カラムと、中央・右カラムを使ってチャートと詳細指標を配置

                # --- 左カラム：主要指標 ---
                with col_left:
                    st.markdown("### 📊 主要指標")

                    # テクニカルシグナル
                    if tech_result.summary.signals:
                        st.markdown("**📈 テクニカルシグナル**")
                        for name, signal in tech_result.summary.signals.items():
                            render_signal_line(name, signal)

                    st.markdown("---")

                    # ファンダメンタル指標
                    if fund_result.ratios:
                        st.markdown("**💼 バリュエーション**")
                        val = fund_result.ratios.valuation if fund_result.ratios else None
                        if val and val.pe_ratio:
                            render_valuation_line("PER", f"{val.pe_ratio:.2f}", evaluate_pe_ratio(val.pe_ratio))
                        if val and val.pb_ratio:
                            render_valuation_line("PBR", f"{val.pb_ratio:.2f}", evaluate_pb_ratio(val.pb_ratio))
                        if val and val.dividend_yield:
                            render_valuation_line(
                                "配当利回り",
                                f"{val.dividend_yield:.2f}%",
                                evaluate_dividend_yield(val.dividend_yield)
                            )

                        st.markdown("**📈 収益性**")
                        prof = fund_result.ratios.profitability
                        if prof.roe:
                            st.metric("ROE", f"{prof.roe * 100:.2f}%")
                        if prof.net_margin:
                            st.metric("純利益率", f"{prof.net_margin * 100:.2f}%")

                # --- 中央カラム：チャート ---
                with col_center:
                    st.markdown("### 📈 価格チャート")

                    # Plotlyチャート作成
                    fig = services['visualizer'].create_plot_figure(
                        df=tech_result.data,
                        symbol=symbol,
                        indicators=[{'name': ind.name, 'params': ind.params, 'plot': ind.plot} for ind in st.session_state.indicator_configs],
                        company_name=fund_result.company_info.name
                    )

                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                    # ダウンロードボタン
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        csv = tech_result.data.to_csv()
                        st.download_button(
                            label="📥 データ(CSV)",
                            data=csv,
                            file_name=f"{symbol}_data_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    with col_dl2:
                        if fig:
                            html_bytes = services['export'].export_chart(fig, format='html')
                            if html_bytes:
                                st.download_button(
                                    label="📥 チャート(HTML)",
                                    data=html_bytes,
                                    file_name=f"{symbol}_chart_{datetime.now().strftime('%Y%m%d')}.html",
                                    mime="text/html",
                                    use_container_width=True
                                )

                # --- 右カラム：詳細指標 ---
                with col_right:
                    st.markdown("### 📋 詳細指標")

                    # テクニカル指標値
                    if tech_result.summary.indicators:
                        with st.expander("📊 テクニカル指標", expanded=True):
                            for name, value in list(tech_result.summary.indicators.items())[:5]:
                                st.metric(name, f"{value:.2f}")

                    # 財務比率
                    if fund_result.ratios:
                        with st.expander("💼 財務比率", expanded=True):
                            liq = fund_result.ratios.liquidity
                            lev = fund_result.ratios.leverage

                            if liq.current_ratio:
                                st.metric("流動比率", f"{liq.current_ratio:.2f}倍")
                            if lev.debt_to_equity:
                                st.metric("負債資本比率", f"{lev.debt_to_equity:.2f}%")

                # ========== 詳細タブ ==========
                st.markdown("---")
                detail_tabs = st.tabs(["📊 テクニカル詳細", "💼 ファンダメンタル詳細", "📄 財務諸表"])

                with detail_tabs[0]:
                    st.markdown("#### テクニカル指標一覧")
                    if tech_result.summary.indicators:
                        # 指標を表形式で表示
                        indicators_df = pd.DataFrame([
                            {"指標": k, "値": f"{v:.4f}"}
                            for k, v in tech_result.summary.indicators.items()
                        ])
                        st.dataframe(indicators_df, use_container_width=True, hide_index=True)

                with detail_tabs[1]:
                    if fund_result.ratios:
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("##### 💰 バリュエーション")
                            val = fund_result.ratios.valuation
                            ratios_data = []
                            if val.pe_ratio: ratios_data.append({"指標": "PER", "値": f"{val.pe_ratio:.2f}"})
                            if val.pb_ratio: ratios_data.append({"指標": "PBR", "値": f"{val.pb_ratio:.2f}"})
                            if val.ps_ratio: ratios_data.append({"指標": "PSR", "値": f"{val.ps_ratio:.2f}"})
                            if val.dividend_yield: ratios_data.append({"指標": "配当利回り", "値": f"{val.dividend_yield:.2f}%"})
                            if ratios_data:
                                st.dataframe(pd.DataFrame(ratios_data), use_container_width=True, hide_index=True)

                            st.markdown("##### 💧 流動性")
                            liq = fund_result.ratios.liquidity
                            liq_data = []
                            if liq.current_ratio: liq_data.append({"指標": "流動比率", "値": f"{liq.current_ratio:.2f}倍"})
                            if liq.quick_ratio: liq_data.append({"指標": "当座比率", "値": f"{liq.quick_ratio:.2f}倍"})
                            if liq_data:
                                st.dataframe(pd.DataFrame(liq_data), use_container_width=True, hide_index=True)

                        with col2:
                            st.markdown("##### 📈 収益性")
                            prof = fund_result.ratios.profitability
                            prof_data = []
                            if prof.roe: prof_data.append({"指標": "ROE", "値": f"{prof.roe * 100:.2f}%"})
                            if prof.roa: prof_data.append({"指標": "ROA", "値": f"{prof.roa * 100:.2f}%"})
                            if prof.gross_margin: prof_data.append({"指標": "売上総利益率", "値": f"{prof.gross_margin * 100:.2f}%"})
                            if prof.net_margin: prof_data.append({"指標": "純利益率", "値": f"{prof.net_margin * 100:.2f}%"})
                            if prof_data:
                                st.dataframe(pd.DataFrame(prof_data), use_container_width=True, hide_index=True)

                            st.markdown("##### ⚖️ レバレッジ")
                            lev = fund_result.ratios.leverage
                            lev_data = []
                            if lev.debt_to_equity: lev_data.append({"指標": "負債資本比率 (D/E)", "値": f"{lev.debt_to_equity:.2f}%"})
                            if lev.debt_to_assets: lev_data.append({"指標": "負債比率 (D/A)", "値": f"{lev.debt_to_assets * 100:.2f}%"})
                            if lev_data:
                                st.dataframe(pd.DataFrame(lev_data), use_container_width=True, hide_index=True)

                with detail_tabs[2]:
                    if include_financials:
                        fs_tabs = st.tabs(["損益計算書", "貸借対照表", "キャッシュフロー"])

                        with fs_tabs[0]:
                            if fund_result.financials is not None and not fund_result.financials.empty:
                                st.dataframe(fund_result.financials, use_container_width=True)
                            else:
                                st.info("データなし")

                        with fs_tabs[1]:
                            if fund_result.balance_sheet is not None and not fund_result.balance_sheet.empty:
                                st.dataframe(fund_result.balance_sheet, use_container_width=True)
                            else:
                                st.info("データなし")

                        with fs_tabs[2]:
                            if fund_result.cash_flow is not None and not fund_result.cash_flow.empty:
                                st.dataframe(fund_result.cash_flow, use_container_width=True)
                            else:
                                st.info("データなし")
                    else:
                        st.info("財務諸表を表示するには、サイドバーで「財務諸表を含む」を有効にしてください")

elif analysis_mode == "⚖️ 銘柄比較":
    st.markdown('<div class="section-header">⚖️ 銘柄比較モード</div>', unsafe_allow_html=True)

    if len(symbols) < 2:
        st.warning("⚠️ 比較するには2つ以上の銘柄を入力してください")
    elif st.session_state.run_requested:
        with st.spinner("分析中..."):
            try:
                # 分析実行
                indicator_configs = [
                    IndicatorConfig(name='SMA', params={'length': 20}),
                    IndicatorConfig(name='RSI', params={'length': 14}),
                ]

                tech_request = TechnicalAnalysisRequest(
                    symbols=symbols,
                    period=period,
                    interval=interval,
                    indicators=indicator_configs,
                    use_cache=True
                )
                tech_results = services['technical'].analyze(tech_request)

                fund_request = FundamentalAnalysisRequest(
                    symbols=symbols,
                    include_financials=False,
                    include_ratios=True
                )
                fund_results = services['fundamental'].analyze(fund_request)

                st.session_state.analysis_results = {
                    'technical': {r.symbol: r for r in tech_results},
                    'fundamental': {r.symbol: r for r in fund_results}
                }
                st.session_state.run_requested = False

            except Exception as e:
                st.error(f"❌ 分析中にエラーが発生しました: {str(e)}")
                st.session_state.run_requested = False
    else:
        st.info("👈 サイドバーから「🚀 分析実行」ボタンを押してください")

    # 比較表示
    if st.session_state.analysis_results:
        # 比較表作成
        comparison_data = []
        company_names: dict[str, str] = {}
        for symbol in symbols:
            tech_result = st.session_state.analysis_results['technical'].get(symbol)
            fund_result = st.session_state.analysis_results['fundamental'].get(symbol)

            company_name = None
            if fund_result and getattr(fund_result, "company_info", None):
                company_name = fund_result.company_info.name

            if company_name:
                company_names[symbol] = company_name
            else:
                company_names.setdefault(symbol, symbol)

            if tech_result and fund_result:
                row = {
                    '銘柄': symbol,
                    '企業名': company_names[symbol],
                    '株価': f"${tech_result.summary.latest_price:.2f}",
                    '変化率': f"{tech_result.summary.price_change_pct:+.2f}%",
                }

                if fund_result.ratios:
                    if fund_result.ratios.valuation.pe_ratio:
                        row['PER'] = f"{fund_result.ratios.valuation.pe_ratio:.2f}"
                    if fund_result.ratios.valuation.pb_ratio:
                        row['PBR'] = f"{fund_result.ratios.valuation.pb_ratio:.2f}"
                    if fund_result.ratios.profitability.roe:
                        row['ROE'] = f"{fund_result.ratios.profitability.roe * 100:.2f}%"
                    if fund_result.company_info.market_cap:
                        row['時価総額'] = f"${fund_result.company_info.market_cap/1e9:.2f}B"

                comparison_data.append(row)

        if comparison_data:
            st.markdown("### 📊 銘柄比較表")
            st.dataframe(
                pd.DataFrame(comparison_data),
                use_container_width=True,
                hide_index=True
            )

            # 価格データの正規化と可視化
            normalized_series: dict[str, pd.Series] = {}
            for symbol in symbols:
                tech_result = st.session_state.analysis_results['technical'].get(symbol)
                if not tech_result or 'Close' not in tech_result.data:
                    continue

                close = tech_result.data['Close'].dropna()
                if close.empty:
                    continue

                normalized_series[symbol] = (close / close.iloc[0]) * 100

            if normalized_series:
                normalized_df = pd.DataFrame(normalized_series)

                st.markdown("### 📈 正規化価格パフォーマンス")
                st.caption("各銘柄の初期値を100に正規化した推移です。")

                perf_fig = go.Figure()
                for symbol in symbols:
                    if symbol not in normalized_series:
                        continue

                    series = normalized_df[symbol].dropna()
                    if series.empty:
                        continue

                    trace_label = company_names.get(symbol, symbol)
                    perf_fig.add_trace(go.Scatter(
                        x=series.index,
                        y=series.values,
                        mode='lines',
                        name=trace_label,
                        line=dict(width=2)
                    ))

                perf_fig.update_layout(
                    title="正規化パフォーマンス（初期値=100）",
                    xaxis_title="日付",
                    yaxis_title="正規化価格",
                    height=480,
                    hovermode='x unified'
                )

                st.plotly_chart(perf_fig, use_container_width=True)

                if len(normalized_series) > 1:
                    st.markdown("### 🔍 ピア平均との差分")
                    st.caption("選択した銘柄と、それ以外の平均との乖離を表示します。正値でアウトパフォーム、負値でアンダーパフォームを示します。")

                    focus_options = list(normalized_series.keys())
                    default_focus = symbols[0] if symbols[0] in focus_options else focus_options[0]
                    focus_symbol = st.selectbox(
                        "比較対象銘柄",
                        options=focus_options,
                        index=focus_options.index(default_focus),
                        format_func=lambda sym: company_names.get(sym, sym)
                    )
                    focus_label = company_names.get(focus_symbol, focus_symbol)

                    peer_symbols = [s for s in normalized_series.keys() if s != focus_symbol]

                    if peer_symbols:
                        comparison_df = normalized_df[[focus_symbol] + peer_symbols].dropna()

                        if not comparison_df.empty:
                            peer_mean = comparison_df[peer_symbols].mean(axis=1)
                            diff_series = comparison_df[focus_symbol] - peer_mean

                            diff_fig = go.Figure()
                            diff_fig.add_trace(go.Scatter(
                                x=diff_series.index,
                                y=diff_series.values,
                                mode='lines',
                                name=f"{focus_label} - ピア平均",
                                line=dict(width=2, color="#1f77b4"),
                                fill='tozeroy',
                                fillcolor='rgba(31, 119, 180, 0.25)'
                            ))
                            diff_fig.add_hline(y=0, line_dash="dash", line_color="gray")

                            diff_fig.update_layout(
                                title=f"{focus_label} vs ピア平均（正規化価格差）",
                                xaxis_title="日付",
                                yaxis_title="差分",
                                height=320,
                                hovermode='x unified'
                            )

                            st.plotly_chart(diff_fig, use_container_width=True)

# ==================== フッター ====================
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 使い方

**🔍 統合分析**
- Technical + Fundamental を統合表示
- 各銘柄の包括的な分析が可能

**⚖️ 銘柄比較**
- 複数銘柄を並べて比較
- 正規化パフォーマンスとピア平均との差分を確認

---
*Powered by yfinance, Plotly, Streamlit*
""")

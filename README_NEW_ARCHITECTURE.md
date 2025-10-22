# Financial Analysis System - New Architecture

本番環境対応の財務分析システムです。Technical分析とFundamental分析を完全に分離した、拡張可能なサービス層アーキテクチャを採用しています。

## 🎯 主な特徴

- ✅ **Technical/Fundamental完全分離** - 独立したサービス層による明確な責務分離
- ✅ **統一API** - WebGUI/MCP/CLI共通のインターフェース
- ✅ **型安全** - Pydanticによる厳密なデータモデル
- ✅ **財務諸表対応** - yfinanceから財務データ取得+指標計算
- ✅ **マルチフォーマット出力** - PNG/HTML/JSON出力+自動フォールバック
- ✅ **スコープ付きキャッシュ** - 予測可能なキャッシング動作
- ✅ **MCP統合準備完了** - サービス層がそのままMCPツールとして利用可能

## 📁 プロジェクト構造

```
FinanceAnalysys/
├── src/
│   ├── core/                    # コアモデルと例外
│   │   ├── models.py           # Pydanticモデル（DTO）
│   │   └── exceptions.py       # カスタム例外
│   │
│   ├── services/                # ビジネスロジック層
│   │   ├── technical_service.py    # テクニカル分析統合サービス
│   │   ├── fundamental_service.py  # ファンダメンタル分析統合サービス
│   │   └── cache_manager.py        # スコープ付きキャッシュ管理
│   │
│   ├── data/                    # データ取得層
│   │   ├── data_fetcher.py     # 価格データ取得
│   │   └── fundamentals.py     # 財務諸表データ取得
│   │
│   ├── analysis/                # 分析層
│   │   ├── technical_indicators.py  # テクニカル指標計算
│   │   └── fundamental_metrics.py   # 財務指標計算
│   │
│   ├── visualization/           # 可視化層
│   │   ├── visualizer.py       # Plotly可視化
│   │   └── export_handler.py   # マルチフォーマット出力
│   │
│   └── backtesting/            # バックテスト層
│       ├── engine.py           # バックテストエンジン
│       ├── strategy.py         # 戦略基底クラス
│       └── performance.py      # パフォーマンス計算
│
├── scripts/                     # CLIスクリプト
│   ├── run_analysis.py         # テクニカル分析CLI
│   └── run_backtest.py         # バックテストCLI
│
├── technical_analysis_app.py   # Streamlitアプリ
├── mcp_server_example.py       # MCPサーバーサンプル
└── pyproject.toml              # uv依存関係管理
```

## 🚀 セットアップ

### 1. 依存関係のインストール

```bash
# uvを使用（推奨）
uv sync

# または pip
pip install -e .
```

### 2. 必要なパッケージ

主要な追加パッケージ：
- `pydantic>=2.0` - 型安全なデータモデル
- `kaleido>=0.2.1` - 静的画像エクスポート（PNG/JPEG/SVG）
- `yfinance>=0.2.55` - 株価・財務データ取得

#### TA-Lib（オプショナル）

TA-Libは**オプショナル依存関係**として設定されています。現在の実装ではvectorbtを使用しているため、TA-Libなしでも動作します。

必要に応じてインストールする場合：

> 📝 **TA-Lib について**
>
> デフォルト構成では TA-Lib を使用していません。高度な一部の指標で TA-Lib を有効にしたい場合のみ、
> 各自の環境に合わせて `pip install ta-lib` などで追加してください。

## 💻 使用方法

### CLIからの使用

#### テクニカル分析

```bash
# デフォルト設定で実行
python scripts/run_analysis.py

# カスタム設定ファイルを使用
python scripts/run_analysis.py --config config/my_config.yaml

# 出力ディレクトリを指定
python scripts/run_analysis.py --output-dir ./results
```

#### バックテスト

```bash
# デフォルト設定で実行（AAPL、MA Crossover + RSI戦略）
python scripts/run_backtest.py

# カスタム設定で実行
python scripts/run_backtest.py --config config/backtest_config.yaml
```

### Streamlitアプリからの使用

```bash
streamlit run technical_analysis_app.py
```

アプリの機能：
- **Technical Analysis** - 価格チャート + テクニカル指標
- **Fundamental Analysis** - 企業情報 + 財務諸表 + 財務比率
- **Backtest** - トレーディング戦略のバックテスト

### Pythonコードからの使用

#### テクニカル分析

```python
from src.services.technical_service import TechnicalAnalysisService
from src.core.models import TechnicalAnalysisRequest, IndicatorConfig

# サービス初期化
service = TechnicalAnalysisService()

# リクエスト作成
request = TechnicalAnalysisRequest(
    symbols=['AAPL', 'GOOGL'],
    period='1y',
    interval='1d',
    indicators=[
        IndicatorConfig(name='SMA', params={'length': 20}),
        IndicatorConfig(name='RSI', params={'length': 14}),
        IndicatorConfig(name='MACD', params={'fast': 12, 'slow': 26, 'signal': 9})
    ]
)

# 分析実行
results = service.analyze(request)

# 結果の取得
for result in results:
    print(f"{result.symbol}: ${result.summary.latest_price:.2f}")
    print(f"Signals: {result.summary.signals}")
```

#### ファンダメンタル分析

```python
from src.services.fundamental_service import FundamentalAnalysisService
from src.core.models import FundamentalAnalysisRequest

# サービス初期化
service = FundamentalAnalysisService()

# リクエスト作成
request = FundamentalAnalysisRequest(
    symbols=['AAPL', 'MSFT'],
    include_financials=True,
    include_ratios=True
)

# 分析実行
results = service.analyze(request)

# 結果の取得
for result in results:
    print(f"{result.symbol} - {result.company_info.name}")
    if result.ratios:
        print(f"P/E Ratio: {result.ratios.valuation.pe_ratio}")
        print(f"ROE: {result.ratios.profitability.roe}")
```

### MCP統合

MCPサーバーとして使用する例（`mcp_server_example.py`を参照）：

```python
from fastmcp import FastMCP
from src.services.technical_service import TechnicalAnalysisService

mcp = FastMCP("Financial Analysis")
service = TechnicalAnalysisService()

@mcp.tool()
def analyze_stock(symbol: str, period: str = "1y"):
    request = TechnicalAnalysisRequest(
        symbols=[symbol],
        period=period,
        indicators=[
            IndicatorConfig(name='SMA', params={'length': 20}),
            IndicatorConfig(name='RSI', params={'length': 14})
        ]
    )
    results = service.analyze(request)
    return results[0].model_dump()

if __name__ == "__main__":
    mcp.run()
```

## 📊 データモデル

### TechnicalAnalysisRequest

```python
TechnicalAnalysisRequest(
    symbols: List[str]              # 分析対象のシンボル
    period: str = "1y"              # 期間（1d, 5d, 1mo, 3mo, 6mo, 1y, etc.）
    interval: str = "1d"            # 間隔（1m, 5m, 1h, 1d, etc.）
    indicators: List[IndicatorConfig]  # 指標設定
    use_cache: bool = True          # キャッシュ使用
    cache_max_age: int = 30         # キャッシュ有効期限（分）
)
```

### TechnicalAnalysisResult

```python
TechnicalAnalysisResult(
    symbol: str                     # シンボル
    company_name: str               # 企業名
    data: pd.DataFrame              # 価格データ+指標
    summary: TechnicalSummary       # サマリー（価格、指標値、シグナル）
    chart_spec: Dict                # チャート仕様
    timestamp: datetime             # 分析実行時刻
)
```

### FundamentalAnalysisRequest

```python
FundamentalAnalysisRequest(
    symbols: List[str]              # 分析対象のシンボル
    include_financials: bool = True  # 財務諸表を含む
    include_ratios: bool = True     # 財務比率を計算
    include_info: bool = True       # 企業情報を含む
)
```

### FundamentalAnalysisResult

```python
FundamentalAnalysisResult(
    symbol: str                     # シンボル
    company_info: CompanyInfo       # 企業情報
    financials: pd.DataFrame        # 損益計算書
    balance_sheet: pd.DataFrame     # 貸借対照表
    cash_flow: pd.DataFrame         # キャッシュフロー計算書
    ratios: FinancialRatios         # 財務比率
    timestamp: datetime             # 分析実行時刻
)
```

## 🔧 設定ファイル

`config/analysis_config.yaml` の例：

```yaml
data:
  symbols:
    - AAPL
    - GOOGL
    - MSFT
  period: 1y
  interval: 1d

indicators:
  - name: SMA
    params:
      length: 20
    plot: true

  - name: RSI
    params:
      length: 14
    plot: true

  - name: MACD
    params:
      fast: 12
      slow: 26
      signal: 9
    plot: true

visualization:
  output_dir: output
  style: seaborn
  figsize: [12, 8]
  dpi: 300
```

## 🎨 出力フォーマット

### サポートする出力形式

1. **PNG/JPEG/SVG** - 静的画像（Kaleidoが必要）
2. **HTML** - インタラクティブチャート（Kaleidoなしでも動作）
3. **JSON** - チャート仕様データ
4. **CSV** - データエクスポート

### 出力例

```python
from src.visualization.export_handler import ExportHandler

handler = ExportHandler()

# PNG出力（Kaleidoが利用可能な場合）
handler.export_chart(fig, format='png', output_path='chart.png')

# HTMLフォールバック（Kaleidoが利用不可の場合は自動的に）
handler.export_chart(fig, format='html', output_path='chart.html')

# バイナリデータとして取得
png_bytes = handler.export_chart(fig, format='png')
```

## 🧪 テスト

```bash
# 全テストを実行
pytest

# 特定のモジュールをテスト
pytest tests/test_services.py

# カバレッジ付き
pytest --cov=src tests/
```

## 📚 利用可能な指標

### Technical Indicators

- **SMA** - Simple Moving Average
- **EMA** - Exponential Moving Average
- **RSI** - Relative Strength Index
- **MACD** - Moving Average Convergence Divergence
- **BBands** - Bollinger Bands
- **Stochastic** - Stochastic Oscillator
- **ADX** - Average Directional Index
- **WILLR** - Williams %R

### Fundamental Metrics

#### Valuation Ratios
- P/E Ratio (Price-to-Earnings)
- P/B Ratio (Price-to-Book)
- P/S Ratio (Price-to-Sales)
- PEG Ratio
- EV/EBITDA
- Dividend Yield

#### Profitability Ratios
- ROE (Return on Equity)
- ROA (Return on Assets)
- Gross Margin
- Operating Margin
- Net Margin
- EBITDA Margin

#### Liquidity Ratios
- Current Ratio
- Quick Ratio
- Cash Ratio

#### Leverage Ratios
- Debt-to-Equity
- Debt-to-Assets
- Interest Coverage

## 🔍 トラブルシューティング

### Kaleido関連のエラー

Kaleidoがインストールされていない場合、PNG/JPEG/SVG出力は自動的にHTML形式にフォールバックします。

静的画像エクスポートが必要な場合：

```bash
uv add kaleido
```

### yfinanceのデータ取得エラー

一部のシンボルでデータ取得に失敗する場合があります。失敗したシンボルはログに記録され、他のシンボルの処理は継続されます。

### キャッシュの問題

キャッシュをクリアする場合：

```python
from src.services.technical_service import TechnicalAnalysisService

service = TechnicalAnalysisService()
service.cache_manager.invalidate()  # すべてのキャッシュをクリア
```

## 🚧 今後の拡張

### MCP統合の完成

`mcp_server_example.py`を参考にして、本格的なMCPサーバーを構築できます：

1. FastMCPをインストール: `uv add fastmcp`
2. MCPツールを定義（サンプル参照）
3. Claude Codeに登録

### WebGUI統合

FastAPIやFlaskを使用したREST APIサーバーを構築できます：

```python
from fastapi import FastAPI
from src.services.technical_service import TechnicalAnalysisService

app = FastAPI()
service = TechnicalAnalysisService()

@app.post("/api/technical")
async def analyze_technical(request: TechnicalAnalysisRequest):
    results = service.analyze(request)
    return [result.model_dump() for result in results]
```

## 📄 ライセンス

本プロジェクトは教育・研究目的で作成されています。

## 🤝 貢献

バグ報告や機能要望はIssueで受け付けています。

---

**Built with** Python 3.13 | yfinance | Plotly | Streamlit | Pydantic

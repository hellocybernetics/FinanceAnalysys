# 株式分析ウェブアプリ

統合財務分析ダッシュボード - テクニカル分析とファンダメンタル分析を統合した包括的な株式分析アプリケーションです。

## 特徴

- **データ取得**: yfinanceによる株価データと財務データの取得
- **テクニカル分析**: vectorbt + TA-Libによる多様なテクニカル指標の計算とシグナル生成
- **ファンダメンタル分析**: 財務諸表、財務比率、企業情報の取得と分析
- **可視化**: Plotlyによるインタラクティブなローソク足チャートと指標表示
- **バックテスト**: 複数の戦略（移動平均クロスオーバー、RSI等）による高速バックテスト
- **Streamlit UI**: ウェブブラウザから利用できる統合分析ダッシュボード
  - 統合分析モード: Technical + Fundamental の包括的分析
  - 銘柄比較モード: 複数銘柄の正規化パフォーマンス比較
- **キャッシュ機能**: 30分以内の再実行時にデータを再利用して高速化
- **データエクスポート**: CSV、HTML、PNG形式でのデータ・チャート出力

## 実装済み機能

### 1. データ取得レイヤー

**DataFetcher** (`src/data/data_fetcher.py`)
- yfinanceによる株価データ取得（OHLCV）
- 複数銘柄の一括取得（日本株、米国株、暗号通貨対応）
- キャッシュ機能（30分以内の再実行時に高速化）
- 企業名情報の取得

**FundamentalsFetcher** (`src/data/fundamentals.py`)
- 企業情報（セクター、業種、時価総額等）
- 財務諸表（損益計算書、貸借対照表、キャッシュフロー計算書）
- 主要統計データ（PER、PBR、配当利回り等）

### 2. 分析サービスレイヤー

**TechnicalAnalysisService** (`src/services/technical_service.py`)
- vectorbt + TA-Libによるテクニカル指標計算
  - トレンド系: SMA, EMA
  - オシレーター系: RSI, Stochastic, WILLR
  - モメンタム系: MACD, ADX
  - ボラティリティ系: Bollinger Bands
- シグナル生成（買い/売り/中立）
- サマリー統計の自動生成

**FundamentalAnalysisService** (`src/services/fundamental_service.py`)
- 財務比率の自動計算
  - バリュエーション指標（PER、PBR、PSR、配当利回り）
  - 収益性指標（ROE、ROA、利益率）
  - 流動性指標（流動比率、当座比率）
  - レバレッジ指標（負債比率）

**CacheManager** (`src/services/cache_manager.py`)
- ファイルベースのキャッシュ管理
- 30分間のTTL（Time To Live）
- 自動的な古いキャッシュの無効化

### 3. バックテストエンジン

**BacktestEngine** (`src/backtesting/engine.py`)
- 複数の戦略によるバックテスト実行
- パフォーマンス指標の自動計算
  - トータルリターン
  - シャープレシオ
  - 最大ドローダウン
  - 勝率、プロフィットファクター
- 結果の可視化（エクイティカーブ、ドローダウンチャート）

**実装済み戦略** (`src/backtesting/strategy.py`)
- MovingAverageCrossoverStrategy（移動平均クロスオーバー）
- RSIStrategy（RSI逆張り戦略）

### 4. 可視化システム

**Visualizer** (`src/visualization/visualizer.py`)
- Plotlyによるインタラクティブなローソク足チャート
- 複数のテクニカル指標を重ね合わせ表示
- サブプロットによる指標の分離表示
- レスポンシブデザイン対応

**ExportHandler** (`src/visualization/export_handler.py`)
- HTML形式でのチャート出力
- PNG形式での画像出力（予定）
- データのCSV出力

### 5. Streamlit統合ダッシュボード

**technical_analysis_app.py**
- **統合分析モード**
  - Technical + Fundamental の同時表示
  - 企業情報ヘッダー（時価総額、価格変動）
  - 主要指標パネル（テクニカルシグナル、バリュエーション）
  - インタラクティブチャート
  - 詳細タブ（指標一覧、財務諸表）

- **銘柄比較モード**
  - 複数銘柄の比較表
  - 正規化パフォーマンスチャート（初期値=100）
  - ピア平均との差分分析

- **プリセット銘柄セット**
  - 米国: Magnificent 7 + ETF
  - 日本: 主要企業

- **動的価格変動計算**
  - 選択可能な比較期間
  - 期間とインターバルに応じた自動調整

## 今後の拡張予定

1. **分析機能の拡充**
   - より多くのテクニカル指標の追加（Ichimoku、Fibonacci等）
   - パターン認識機能の実装（チャートパターン検出）
   - セクター分析とマクロ経済指標の統合

2. **機械学習統合**
   - 予測モデルの統合（Prophet、LSTM等）
   - バックテスト結果の機械学習による最適化
   - 異常検出アルゴリズムの実装
   - 関連ライブラリ準備済み: scikit-learn, scipy, torch, mlflow

3. **バックテスト機能の強化**
   - より多くの戦略の実装
   - ポートフォリオバックテスト
   - リスク管理機能の追加
   - vectorbt、lean環境の活用

4. **外部連携機能**
   - Twitter API連携による分析結果の自動投稿（ライブラリ準備済み）
   - APIサーバー化（FastAPI、uvicorn準備済み）
   - スケジュール実行機能（schedule準備済み）

## プロジェクト構成

```
FinanceAnalysys/
├── config/                          # 設定ファイル
│   ├── analysis_config.yaml         # テクニカル分析設定
│   └── backtest_config.yaml         # バックテスト設定
├── src/                             # ソースコード
│   ├── core/                        # コアモデルと例外定義
│   │   ├── models.py                # データモデル（Pydantic）
│   │   └── exceptions.py            # カスタム例外
│   ├── data/                        # データ取得レイヤー
│   │   ├── data_fetcher.py          # 株価データ取得
│   │   └── fundamentals.py          # 財務データ取得
│   ├── services/                    # ビジネスロジックレイヤー
│   │   ├── cache_manager.py         # キャッシュ管理
│   │   ├── technical_service.py     # テクニカル分析サービス
│   │   └── fundamental_service.py   # ファンダメンタル分析サービス
│   ├── analysis/                    # 分析モジュール
│   │   ├── technical_indicators.py  # テクニカル指標計算
│   │   └── fundamental_metrics.py   # 財務比率計算
│   ├── backtesting/                 # バックテストエンジン
│   │   ├── engine.py                # バックテスト実行エンジン
│   │   ├── strategy.py              # 取引戦略定義
│   │   └── performance.py           # パフォーマンス計算
│   ├── visualization/               # 可視化モジュール
│   │   ├── visualizer.py            # チャート生成
│   │   └── export_handler.py        # エクスポート機能
│   └── utils/                       # ユーティリティ
│       ├── config_loader.py         # 設定ファイル読み込み
│       ├── price_change.py          # 価格変動計算
│       └── timeframes.py            # 時間軸ユーティリティ
├── scripts/                         # 実行スクリプト
│   ├── run_analysis.py              # CLI分析ツール
│   ├── run_backtest.py              # CLIバックテストツール
│   └── run_streamlit_uv.ps1         # Streamlit起動スクリプト（Windows）
├── tests/                           # テストコード
│   ├── test_technical_indicators.py
│   ├── test_strategies.py
│   ├── test_backtest_engine.py
│   └── test_performance.py
├── output/                          # 出力ディレクトリ
│   ├── cache/                       # キャッシュデータ
│   ├── images/                      # チャート画像
│   └── backtest_results/            # バックテスト結果
├── technical_analysis_app.py        # Streamlitダッシュボード（メインUI）
├── pyproject.toml                   # プロジェクト設定（uv管理）
└── README.md                        # このファイル
```

## 使用方法

### 環境設定

```bash
# 依存関係のインストール
uv sync
```

> 💡 **Windows + uv の注意:** PowerShell で `uv` を利用する際に `.venv\lib64` に対するアクセス拒否エラーが出る場合は、
> `set UV_LINK_MODE=copy` を実行してから `uv sync` や `uv run` を実行してください。リポジトリにはこの設定を自動で行う
> `scripts/run_streamlit_uv.ps1` も用意しています。

### Streamlit統合ダッシュボード（推奨）

```bash
# Streamlitダッシュボードを起動
uv run streamlit run technical_analysis_app.py

# または Windows PowerShell
pwsh -File scripts/run_streamlit_uv.ps1
```

ブラウザで http://localhost:8501 にアクセスして以下の機能を利用できます：
- **統合分析**: Technical + Fundamental の包括的分析
- **銘柄比較**: 複数銘柄の正規化パフォーマンス比較
- インタラクティブなチャート操作
- CSV/HTMLでのデータエクスポート

### CLI分析ツール

```bash
# デフォルト設定で分析実行
python scripts/run_analysis.py

# カスタム設定ファイルを指定
python scripts/run_analysis.py --config config/analysis_config.yaml

# 出力ディレクトリを指定
python scripts/run_analysis.py --output-dir output/custom
```

### CLIバックテストツール

```bash
# デフォルト設定でバックテスト実行
python scripts/run_backtest.py

# カスタム設定ファイルを指定
python scripts/run_backtest.py --config config/backtest_config.yaml

# 出力ディレクトリを指定
python scripts/run_backtest.py --output-dir output/backtest
```

### 設定ファイルの例

**config/analysis_config.yaml** - テクニカル分析設定
```yaml
data:
  symbols:
    - AAPL     # Apple Inc.
    - MSFT     # Microsoft Corporation
    - 7203.T   # Toyota Motor Corporation (Japan)
    - 9984.T   # SoftBank Group Corp. (Japan)
    - BTC-USD  # Bitcoin
  period: 1y   # Options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
  interval: 1d # Options: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo

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
  output_dir: "output/images"
  style: "seaborn"
  figsize: [12, 8]
  dpi: 300
  show_plots: false
```

**config/backtest_config.yaml** - バックテスト設定
```yaml
data:
  symbols: ['AAPL', 'MSFT']
  period: 1y
  interval: 1d

strategies:
  - name: MA_Crossover
    params:
      short_length: 20
      long_length: 50

  - name: RSI
    params:
      rsi_length: 14
      oversold: 30
      overbought: 70

backtest:
  initial_capital: 10000
  commission: 0.001

visualization:
  output_dir: "output"
```

## 技術スタック

### コア技術
- **言語**: Python 3.13
- **パッケージ管理**: uv
- **データモデル**: Pydantic 2.0+

### データ取得・分析
- **株価データ**: yfinance 0.2.55+
- **テクニカル分析**: vectorbt 0.27.2, TA-Lib（vectorbt経由）
- **財務分析**: financepy 0.370
- **統計解析**: scipy 1.15.2, statsmodels 0.14.4+

### バックテスト
- **エンジン**: 自作BacktestEngine
- **高度なバックテスト**: vectorbt 0.27.2（準備済み）
- **アルゴリズムトレード**: lean（準備済み）

### 機械学習（準備済み）
- **フレームワーク**: scikit-learn 1.6.1, PyTorch 2.7.0+
- **時系列予測**: Prophet 1.1.6+
- **実験管理**: MLflow 2.22.0+

### 可視化
- **インタラクティブ**: Plotly 5.24.1
- **静的チャート**: matplotlib 3.10.1, seaborn 0.13.2
- **金融チャート**: mplfinance 0.12.10b0+
- **画像出力**: kaleido 0.2.1+

### UI・Web
- **ダッシュボード**: Streamlit 1.45.0+
- **APIサーバー**: FastAPI 0.115.12+, uvicorn 0.34.0+（準備済み）

### データ処理
- **データフレーム**: pandas 2.2.3, numpy 2.1.3
- **高速処理**: pyarrow 19.0.1+, numba 0.61.0

### その他
- **設定管理**: PyYAML 6.0.2+, python-dotenv 1.1.0+
- **スケジューリング**: schedule 1.2.2（準備済み）

## デプロイ

### Streamlit Community Cloud（推奨）

1. GitHubにリポジトリをプッシュ
2. [Streamlit Community Cloud](https://streamlit.io/cloud)にアクセス
3. "New app"をクリック
4. リポジトリと`technical_analysis_app.py`を選択
5. デプロイ

### ローカル実行

```bash
# 環境構築
uv venv
uv sync

# アプリケーション起動
uv run streamlit run technical_analysis_app.py
```

### 必要な環境
- Python 3.13
- uvパッケージマネージャー
- 依存関係は`pyproject.toml`で管理

## 開発アーキテクチャ

### レイヤー構成
```
UI Layer (Streamlit)
    ↓
Service Layer (TechnicalAnalysisService, FundamentalAnalysisService)
    ↓
Analysis Layer (TechnicalIndicators, FundamentalMetrics)
    ↓
Data Layer (DataFetcher, FundamentalsFetcher)
    ↓
External APIs (yfinance)
```

### 設計原則
- **レイヤー分離**: UI、サービス、データ層の明確な分離
- **依存性注入**: サービス間の疎結合
- **型安全性**: Pydanticによる厳格な型定義
- **キャッシュ戦略**: ファイルベースキャッシュによる高速化
- **エラーハンドリング**: カスタム例外による明確なエラー処理

### テスト
- **ユニットテスト**: pytest 8.3.5+
- **テスト対象**:
  - テクニカル指標計算（`tests/test_technical_indicators.py`）
  - バックテスト戦略（`tests/test_strategies.py`）
  - バックテストエンジン（`tests/test_backtest_engine.py`）
  - パフォーマンス計算（`tests/test_performance.py`）

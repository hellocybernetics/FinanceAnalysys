# 株式分析ウェブアプリ

株の分析とTwitterへの投稿機能を備えたアプリケーションです。

## 特徴

- 株価データの取得(yfinance,openbb)
- 株価データのテクニカル分析(Ta-Lib)
- 株価データの機械学習アルゴリズム開発(scikit-learn, scipy)
- 株価データの金融統計解析(financepy)
- 分析結果の表示(matplitlib, seabron)
- 高速バックテスト(vectortb)
- docker上でのバックテストとトレード(lean)
- Twitter APIを使った分析結果の自動投稿(tweepy, twitter)
- lean上でのアルゴリズム開発を柔軟に実施するためのライブラリ

## 現在の作業状況

以下の機能を実装しました：

1. **データ取得モジュール**
   - vectorbtとyfinanceを使用した株価データの取得
   - 複数の銘柄（日本株、米国株、ビットコイン）に対応
   - 取得したデータをCSVとして保存
   - 30分以内の再実行時にはキャッシュデータを使用（効率化）
   - 会社名情報の取得と表示
   - データ取得失敗時のTODOリスト自動生成

2. **テクニカル分析モジュール**
   - vectorbtを通じてtalibを使用した各種テクニカル指標の計算
   - SMA、EMA、RSI、MACD、ボリンジャーバンドなどの指標をサポート
   - 設定ファイルによる指標パラメータのカスタマイズ

3. **可視化モジュール**
   - 株価とテクニカル指標の可視化
   - 複数の指標を含む総合チャートの生成
   - 結果をPNG形式で保存

4. **設定システム**
   - YAMLまたはJSONによる分析設定
   - 銘柄、期間、間隔、指標などの設定が可能
   - 可視化設定のカスタマイズ

## 当面の目標

1. **分析機能の拡充**
   - より多くのテクニカル指標の追加
   - ファンダメンタル分析の強化
   - パターン認識機能の実装

2. **機械学習統合**
   - 予測モデルとテクニカル分析の統合
   - バックテスト結果の機械学習による最適化
   - 異常検出アルゴリズムの実装

3. **ユーザーインターフェース改善**
   - Streamlitを使用したウェブインターフェースの開発
   - インタラクティブな分析ダッシュボード
   - カスタマイズ可能なレポート生成

4. **パフォーマンス最適化**
   - 大量データ処理の効率化
   - 並列処理によるバックテスト高速化
   - メモリ使用量の最適化
   - データキャッシュによる重複取得の防止（30分以内）

## プロジェクト構成

```
FinanceAnalysisPy/
├── config/                 # 設定ファイル (例: analysis_config.yaml)
├── credentials/            # 認証情報 (例: twitter_config.json)
├── lib/                    # 外部ライブラリ (ta-lib)
├── output/                 # データ保存、分析結果
├── projects/               # バックテスト/トレードのDockerプロジェクト
├── scripts/                # 実行スクリプト (例: run_analysis.py)
├── src/                    # ソースコード
├── pyproject.toml          # プロジェクト設定 (uv/Poetry)
├── README.md               # このファイル
└── requirements.txt        # 依存関係 (pip)
```

### 新規追加ファイル

- **config/analysis_config.yaml**: 分析設定ファイル
- **src/data/data_fetcher.py**: データ取得モジュール
- **src/analysis/technical_indicators.py**: テクニカル分析モジュール
- **src/visualization/visualizer.py**: 可視化モジュール
- **scripts/run_analysis.py**: 分析実行スクリプト

## 使用方法

### 環境設定

```bash
# 依存関係のインストール
uv sync
```

### 分析実行

```bash
# デフォルト設定で実行
python scripts/run_analysis.py

# カスタム設定ファイルを指定
python scripts/run_analysis.py --config path/to/config.yaml

# vectorbtを使用
python scripts/run_analysis.py --use-vectorbt

# 出力ディレクトリを指定
python scripts/run_analysis.py --output-dir path/to/output
```

### 設定ファイルの例

```yaml
# Data Sources
data:
  symbols:
    - AAPL     # Apple Inc.
    - MSFT     # Microsoft Corporation
    - 7203.T   # Toyota Motor Corporation (Japan)
    - 9984.T   # SoftBank Group Corp. (Japan)
    - BTC-USD  # Bitcoin
  period: 1y
  interval: 1d

# Technical Indicators
indicators:
  - name: SMA
    params:
      length: 20
    plot: true
  
  - name: RSI
    params:
      length: 14
    plot: true

# Visualization settings
visualization:
  output_dir: "output/images"
  style: "seaborn"
  figsize: [12, 8]
  dpi: 300
  show_plots: false
```

## 開発方針

1. **モジュール性**
   - 各機能は独立したモジュールとして実装
   - 依存関係の最小化
   - 再利用性の重視

2. **テストと品質**
   - ユニットテストの充実
   - 継続的インテグレーション
   - コード品質の維持

3. **パフォーマンス**
   - 効率的なデータ処理
   - キャッシュの活用
   - 非同期処理の導入

4. **セキュリティ**
   - APIキーの安全な管理
   - データの暗号化
   - アクセス制御の実装

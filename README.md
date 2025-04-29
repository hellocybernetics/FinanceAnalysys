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

## プロジェクト構成

```
FinanceAnalysisPy/
├── src/                    # ソースコード
│   ├── core/              # コア機能
│   │   ├── data/         # データ取得・管理
│   │   └── utils/        # ユーティリティ関数
│   ├── analysis/         # 分析機能
│   │   └── fundamental/  # ファンダメンタル分析
│   ├── visualization/    # 可視化機能
│   └── social/          # SNS連携
│       └── twitter/     # Twitter API連携
├── credentials/          # 認証情報
│   └── twitter_config.json
├── data/                # データ保存
├── result/             # 分析結果
├── examples/           # 使用例
├── projects/           # バックテスト/トレードのDockerプロジェクト
├── lib/               # 外部ライブラリ
├── basic_data_fetch.py  # メインスクリプト
├── pyproject.toml      # プロジェクト設定
├── requirements.txt    # 依存関係
└── README.md          # ドキュメント
```

### 各ディレクトリの役割

- **src/**: アプリケーションのソースコード
  - core/: コア機能（データ取得、ユーティリティ）
  - analysis/: 分析機能
  - visualization/: 可視化機能
  - social/: SNS連携機能

- **credentials/**: 認証情報
  - Twitter APIの設定ファイル
  - その他のAPI認証情報

- **data/**: データ保存
  - キャッシュされたデータ
  - 一時ファイル

- **result/**: 分析結果
  - 生成されたチャート
  - 分析レポート
  - 出力ファイル

- **examples/**: 使用例
  - サンプルコード
  - チュートリアル

- **projects/**: プロジェクト固有のコード
  - 個別の分析プロジェクト
  - バックテスト設定

- **lib/**: 外部ライブラリ
  - カスタムライブラリ
  - 依存関係

### 主要ファイル

- **basic_data_fetch.py**: メインスクリプト
  - データ取得
  - 分析実行
  - 結果の可視化
  - Twitter投稿

- **pyproject.toml**: プロジェクト設定
  - パッケージ情報
  - ビルド設定
  - 開発依存関係

- **requirements.txt**: 依存関係
  - 必要なパッケージ
  - バージョン情報

## 今後の発展

### 短期的な改善

1. **分析機能の拡充**
   - テクニカル分析モジュールの追加
   - 機械学習による予測機能
   - リスク分析ツール

2. **データソースの拡張**
   - 複数の金融データソースの統合
   - リアルタイムデータの取得
   - 代替データの統合

3. **可視化の強化**
   - インタラクティブなダッシュボード
   - カスタマイズ可能なチャート
   - レポート生成機能

### 中長期的な展望

1. **API化**
   - RESTful APIの提供
   - クライアントライブラリの開発
   - 認証・認可機能の実装

2. **拡張性の向上**
   - プラグインシステムの導入
   - カスタム指標の追加機能
   - マルチアセット対応

3. **コミュニティ機能**
   - 分析結果の共有機能
   - ユーザー間のコラボレーション
   - 分析テンプレートの共有

4. **ドキュメントとサポート**
   - 詳細なAPIドキュメント
   - チュートリアルの充実
   - コミュニティサポートの強化

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

## Twitter APIの認証について

### 認証方式

このアプリケーションはTwitter API v2のOAuth 1.0認証を使用しています。

### 認証の問題と解決策

## セットアップ

### 環境変数

* 開発環境でのみ使用: `OAUTHLIB_INSECURE_TRANSPORT=1`
  * 注意: 本番環境ではHTTPSを使用する必要があります

### Twitterの認証設定

認証には以下のファイルを作成してください：

```
credentials/twitter_config.json
```

## 使用方法

### アルゴリズム開発

Dockerプロジェクトでのアルゴリズム開発のスニペット

```python
# main.py の例
from AlgorithmImports import *

# 自作モジュールをインポート
from analyzer.data_fetcher import fetch_data_for_symbol
from analyzer.indicators import calculate_rsi, calculate_sma
from utils.helpers import log_message

class MyAnalysisAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)
        self.symbol = self.AddEquity("SPY", Resolution.Daily).Symbol

        # データ取得モジュールを利用 (例: 履歴データを取得して初期化に使うなど)
        # ※ OnData でリアルタイムに処理する場合は不要なことも
        # historical_data = fetch_data_for_symbol(self.symbol.Value, self.StartDate, self.EndDate)
        log_message("Initialization complete using custom utils.")

        # 指標用のインジケーターを登録 (TA-Libラッパーなどを使う場合)
        # または、自作モジュールで計算した結果をカスタムデータとして利用するなど

    def OnData(self, data):
        if not data.ContainsKey(self.symbol):
            return

        # 現在の価格データフレームを取得 (例)
        current_slice_df = self.History(self.symbol, 20, Resolution.Daily) # 過去20日分

        if current_slice_df.empty:
            return

        # 指標計算モジュールを利用
        rsi_value = calculate_rsi(current_slice_df['close'])
        sma_value = calculate_sma(current_slice_df['close'])

        # デバッグ出力に自作ロガーを利用
        log_message(f"Current RSI: {rsi_value[-1]}, SMA: {sma_value[-1]}")

        # ここに取引ロジックを記述
        # if rsi_value[-1] < 30 and data[self.symbol].Close > sma_value[-1]:
        #    self.SetHoldings(self.symbol, 1.0)
        # elif rsi_value[-1] > 70 and data[self.symbol].Close < sma_value[-1]:
        #    self.Liquidate(self.symbol)

    # ... (他のメソッド) ...
    ```

## 機械学習機能

### セットアップ

必要なPythonライブラリをインストールします。`uv` がインストールされていることを確認してください。

```bash
# ML関連の依存関係を追加 (初回のみ)
uv add mlflow fastapi "uvicorn[standard]" PyYAML python-dotenv scikit-learn pandas numpy yfinance ta-lib

# もし uv add で ta-lib がエラーになる場合、個別にインストールが必要かもしれません
# 例: Windows 用 Wheel ファイルを使用する場合など (詳細は環境に合わせて調整)
# uv add lib/ta_lib-0.6.3-cp313-cp313-win_amd64.whl

# プロジェクトの依存関係をインストール/同期
uv sync
```

設定ファイル (`config/training_config.yaml`, `config/api_config.yaml`) を確認し、必要に応じて調整してください。特に、特徴量リストやモデルパラメータをカスタマイズできます。

### モデル学習

`scripts/train_model.py` スクリプトを使用してモデルを学習させます。

```bash
python scripts/train_model.py --symbol AAPL --period 5y --config config/training_config.yaml
```

*   `--symbol`: 学習対象の銘柄コード (デフォルト: AAPL)
*   `--period`: 学習に使用するデータの期間 (デフォルト: 5y)
*   `--config`: 学習設定ファイルのパス (デフォルト: config/training_config.yaml)
*   `--mlflow_tracking_uri`: (オプション) MLflow Tracking Server の URI
*   `--experiment_name`: (オプション) MLflow の Experiment 名
*   `--run_name`: (オプション) MLflow の Run 名

学習が完了すると、パラメータ、メトリクス、学習済みモデル (`model` ディレクトリ内)、およびスケーラー (`preprocessing/scaler.pkl`) が MLflow Run のアーティファクトとして記録されます。デフォルトでは、カレントディレクトリに `mlruns` ディレクトリが作成されます。

### MLflow UI

学習結果を確認するには、MLflow UI を起動します。

```bash
mlflow ui
```

ブラウザで `http://localhost:5000` (デフォルト) を開くと、実験結果をインタラクティブに確認できます。

### 予測 (スクリプト)

学習済みモデルを使用して予測を行うには、`scripts/predict.py` スクリプトを使用します。

```bash
# 例: 特定の MLflow Run ID を指定して予測
python scripts/predict.py --symbol AAPL --period 1y --model_uri runs:/YOUR_RUN_ID/model --output_file result/predictions/AAPL_predictions.csv --config config/training_config.yaml
```

*   `--symbol`: 予測対象の銘柄コード
*   `--period`: 予測に必要な入力データの期間
*   `--model_uri`: 使用する学習済みモデルの MLflow URI (`runs:/<run_id>/model` の形式)
*   `--output_file`: (オプション) 予測結果を保存するCSVファイルのパス
*   `--config`: 特徴量エンジニアリングのパラメータを取得するための設定ファイルパス (学習時と同じものを指定)

### 予測 (API)

学習済みモデルをサービングする FastAPI サーバーを起動できます。

1.  **環境変数の設定 (任意)**:
    APIが使用するモデルのURIを環境変数 `MODEL_URI` で指定できます。設定しない場合は `config/api_config.yaml` の `default_model_uri` が参照されます。
    ```bash
    # 例 (PowerShell):
    $env:MODEL_URI="runs:/YOUR_RUN_ID/model"
    # 例 (Bash):
    export MODEL_URI="runs:/YOUR_RUN_ID/model"
    ```

2.  **APIサーバーの起動**:
    ```bash
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    ```

3.  **APIへのリクエスト**:
    サーバーが起動したら、別のターミナルから `curl` などでリクエストを送信できます。
    ```bash
    curl -X POST http://localhost:8000/predict/AAPL
    ```
    成功すると、最新の予測結果がJSON形式で返されます。

APIドキュメントは `http://localhost:8000/docs` で確認できます。
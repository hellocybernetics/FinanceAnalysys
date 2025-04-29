# simple_backtest.py
import pandas as pd
import numpy as np
import vectorbt as vbt
import talib
from datetime import datetime
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
from src.core.data.yahoo.fetch_from_yahoo import fetch_stock_data

# vectorbtのグローバル設定
vbt.settings.array_wrapper['freq'] = 'D'  # 日次の頻度を設定

class DataManager:
    def __init__(self, base_dir='result'):
        self.base_dir = base_dir
        self.backtest_dir = os.path.join(base_dir, 'backtest')
        self.ensure_directories()
        self.symbol = ""
    
    def ensure_directories(self):
        """必要なディレクトリを作成"""
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.backtest_dir, exist_ok=True)
    
    def save_data(self, symbol, df, metadata=None):
        self.symbol = symbol
        """データを保存する"""
        symbol_dir = os.path.join(self.backtest_dir, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        
        # タイムゾーン情報を削除して保存
        df.index = pd.to_datetime(df.index, utc=True).tz_convert('UTC').tz_localize(None)
        
        # データの保存
        df.to_csv(os.path.join(symbol_dir, 'price_data.csv'))
        
        # メタデータの保存
        if metadata:
            with open(os.path.join(symbol_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f)
    
    def load_data(self, symbol):
        """データを読み込む"""
        symbol_dir = os.path.join(self.backtest_dir, symbol)
        
        # データの読み込み
        df = pd.read_csv(
            os.path.join(symbol_dir, 'price_data.csv'),
            index_col=0,
            parse_dates=True
        )
        
        # タイムゾーン情報を削除
        df.index = pd.to_datetime(df.index, utc=True).tz_convert('UTC').tz_localize(None)
        
        # メタデータの読み込み
        metadata_path = os.path.join(symbol_dir, 'metadata.json')
        metadata = None
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return df, metadata

# データの準備
def prepare_data(symbol, period='1y', interval='1d', use_saved=True, base_dir='result'):
    data_manager = DataManager(base_dir=base_dir)
    
    if use_saved:
        try:
            # 保存されたデータを読み込む
            df, metadata = data_manager.load_data(symbol)
            print(f"保存されたデータを読み込みました: {symbol}")
            
            # データの検証
            if df.empty:
                print("保存されたデータが空です。新規にデータを取得します。")
                raise FileNotFoundError
            
            return df
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f"保存されたデータが見つからないか、空です。新規にデータを取得します: {symbol}")
    
    # Yahoo Financeからデータを取得
    df, _ = fetch_stock_data(
        symbol=symbol,
        period=period,  # 指定された期間を使用
        interval=interval
    )
    
    if df is None or df.empty:
        raise ValueError(f"Failed to fetch data for {symbol}")
    
    # タイムゾーン情報の処理
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('UTC').tz_localize(None)
    
    # データ型の変換
    df['Close'] = df['Close'].astype(np.float64)
    
    # インデックスの頻度を設定
    df = df.resample('D').last()  # 日次の頻度を設定
    
    # データの保存
    metadata = {
        'symbol': symbol,
        'fetch_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    data_manager.save_data(symbol, df, metadata)
    
    return df


def custom_momentum_oscillator(df, period=14):
    momentum = talib.MOM(df['Close'], timeperiod=period)
    abs_momentum = talib.MOM(df['Close'], timeperiod=1).abs()
    df['CMO'] = momentum.rolling(window=period).sum() / \
                abs_momentum.rolling(window=period).sum() * 100
    return df['CMO']

def custom_trix(df, period=14):
    df['Trix'] = talib.EMA(df['Close'], timeperiod=period)
    df['Trix'] = talib.EMA(df['Trix'], timeperiod=period)
    df['Trix_Signal'] = talib.EMA(df['Trix'], timeperiod=period)
    return df['Trix'], df['Trix_Signal']

# Chande Momentum Oscillator (CMO)
def chande_momentum_oscillator(df, period=14):
    df['CMO'] = ((df['Close'] - df['Close'].shift(period)).rolling(window=period).sum()) / \
                ((df['Close'] - df['Close'].shift(1)).abs().rolling(window=period).sum()) * 100
    return df['CMO']

# Trix Indicator
def trix(df, period=14):
    df['Trix'] = df['Close'].ewm(span=period, min_periods=period).mean()
    df['Trix_Signal'] = df['Trix'].ewm(span=period, min_periods=period).mean()
    return df['Trix'], df['Trix_Signal']

# シグナル生成
def generate_signals(df, cmo_period, trix_period):

    df['CMO'] = chande_momentum_oscillator(df, cmo_period)
    df['Trix'], df['Trix_Signal'] = trix(df, trix_period)


    # Define entry and exit signals based on CMO and Trix
    df['Entry'] = (
        (df['CMO'] > 0) &  # CMO crosses above 0
        (df['Trix'] > df['Trix_Signal'])   # Trix crosses above Trix Signal line
    )

    df['Exit'] = (
        (df['CMO'] < 0) &  # CMO crosses below 0
        (df['Trix'] < df['Trix_Signal'])   # Trix crosses below Trix Signal line
    )

    # Convert signals to boolean arrays
    entries = df['Entry'].to_numpy()
    exits = df['Exit'].to_numpy()
    return entries, exits


# バックテストの実行
def run_backtest(df, initial_capital=100000, base_dir='result'):
    try:
        # データ型の確認と変換
        if not isinstance(df['Close'].dtype, np.float64):
            df['Close'] = df['Close'].astype(np.float64)
        
        # インデックスの型を確認
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True).tz_convert('UTC').tz_localize(None)
        
        # インデックスの頻度を確認と設定
        df = df.resample('D').last()
        
        # Define the range of periods for optimization
        cmo_period_range = range(2, 21)  # CMO period starts from 2
        trix_period_range = range(2, 21)  # Example range for Trix period

        # Create an empty matrix to store the total returns for each combination of periods
        total_returns_matrix = np.zeros((len(cmo_period_range), len(trix_period_range)))
        all_portfolios = [[[] for _ in range(len(trix_period_range))] for _ in range(len(cmo_period_range))]
        # Loop over all combinations of periods and store the total returns
        for i, cmo_period in enumerate(cmo_period_range):
            for j, trix_period in enumerate(trix_period_range):
                entries, exits = generate_signals(df, cmo_period, trix_period)
                portfolio = vbt.Portfolio.from_signals(
                    df['Close'],
                    entries,
                    exits,
                    init_cash=initial_capital,  # initial capital
                    fees=0.0002,  # trading fees
                    freq='D'  # daily frequency
                )
                total_returns_matrix[i, j] = portfolio.total_return()
                all_portfolios[i][j].append(portfolio)

        # Find the best parameters based on the highest total return
        best_cmo_period_idx, best_trix_period_idx = np.unravel_index(np.argmax(total_returns_matrix), total_returns_matrix.shape)
        best_cmo_period = cmo_period_range[best_cmo_period_idx]
        best_trix_period = trix_period_range[best_trix_period_idx]

        print(f"Best CMO Period: {best_cmo_period}")
        print(f"Best Trix Period: {best_trix_period}")
        print(f"Best Total Return: {total_returns_matrix[best_cmo_period_idx, best_trix_period_idx]}")

        # Plot the heatmap of total returns
        plt.figure(figsize=(10, 8))
        sns.heatmap(total_returns_matrix, annot=False, cmap="YlGnBu", xticklabels=trix_period_range, yticklabels=cmo_period_range)
        plt.title('Heatmap of Total Returns for CMO and Trix Periods')
        plt.xlabel('Trix Period')
        plt.ylabel('CMO Period')
        

        # 結果の表示
        print("\n=== バックテスト結果 ===")
        
        # 最適なパラメータでのポートフォリオを取得
        portfolio = all_portfolios[best_cmo_period_idx][best_trix_period_idx][0]
        
        # 基本統計情報の取得と保存
        stats = portfolio.stats()
        stats_dict = stats.to_dict()
        
        # 統計情報に最適パラメータを追加
        stats_dict['best_parameters'] = {
            'cmo_period': best_cmo_period,
            'trix_period': best_trix_period
        }
        
        # TimestampとTimedeltaオブジェクトを文字列に変換
        def convert_timestamps(obj):
            if isinstance(obj, pd.Timestamp):
                return obj.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(obj, pd.Timedelta):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_timestamps(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_timestamps(item) for item in obj]
            return obj
        

        
        # ポートフォリオのプロット
        try:
            print("\nプロットを生成中...")
            # プロットの生成
            fig = portfolio.plot()
            fig.show()
            
            
        except Exception as e:
            print(f"プロットの生成中にエラーが発生しました: {str(e)}")
            print("代わりに基本的なパフォーマンス指標を表示します。")
            print(f"累積リターン: {portfolio.cumulative_returns().iloc[-1]:.2%}")
            print(f"最大ドローダウン: {portfolio.max_drawdown():.2%}")
            print(f"取引回数: {len(portfolio.orders)}")
        
        return portfolio
    except Exception as e:
        print(f"バックテスト実行中にエラーが発生しました: {str(e)}")
        raise

# メイン処理
def main():
    # パラメータの設定
    symbol = 'AAPL'  # S&P 500 ETF
    period = '2y'  # 1年間のデータを取得
    interval = '1d'  # 日次のデータを取得
    initial_capital = 100000
    use_saved_data = True  # 保存されたデータを使用するかどうか
    base_dir = 'result'  # ベースディレクトリ
    
    try:
        # データの準備
        df = prepare_data(symbol, period, interval, use_saved=use_saved_data, base_dir=base_dir)
        
        # データの基本情報を表示
        print("\n=== データ情報 ===")
        print(f"データ期間: {df.index[0]} から {df.index[-1]}")
        print(f"データ件数: {len(df)}")
        print(f"最初の5行:\n{df.head()}")

        # バックテストの実行
        portfolio = run_backtest(df, initial_capital, base_dir=base_dir)
        
        # 結果の表示
        print("\n=== バックテスト結果 ===")
        print(portfolio.stats())
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")


if __name__ == '__main__':
    main()
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
def chande_momentum_oscillator_talib(close_prices, period=14):
    return talib.CMO(close_prices, timeperiod=period)

# Trix Indicator
def trix_talib(close_prices, period=14):
    trix_val = talib.TRIX(close_prices, timeperiod=period)
    trix_signal = talib.EMA(trix_val, timeperiod=9) # Use 9-period EMA for signal
    return trix_val, trix_signal

# シグナル生成 (空売り対応)
def generate_signals(df_close, cmo_period, trix_period):
    """CMOとTrixに基づいてロング・ショートシグナルを生成"""
    cmo = chande_momentum_oscillator_talib(df_close, period=cmo_period)
    trix, trix_signal = trix_talib(df_close, period=trix_period)

    # ロングシグナル
    entries_pd = (cmo > 0) & (trix > trix_signal) & (trix.shift(1) <= trix_signal.shift(1))
    exits_pd = (cmo < 0) & (trix < trix_signal) & (trix.shift(1) >= trix_signal.shift(1))

    # ショートシグナル (対称ロジック)
    short_entries_pd = exits_pd.copy()
    short_exits_pd = entries_pd.copy()

    return (
        entries_pd.to_numpy(),
        exits_pd.to_numpy(),
        short_entries_pd.to_numpy(),
        short_exits_pd.to_numpy(),
    )

# バックテストの実行
def run_backtest(df, initial_capital=100000, base_dir='result', allow_shorting=True):
    try:
        # データ型の確認と変換
        if not isinstance(df['Close'].dtype, np.float64):
            df['Close'] = df['Close'].astype(np.float64)
        
        # インデックスの型を確認
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True).tz_convert('UTC').tz_localize(None)
        
        # インデックスの頻度を確認と設定
        df = df.resample('D').ffill()
        df.index.freq = df.index.inferred_freq
        df.dropna(subset=['Close'], inplace=True)
        
        # Define the range of periods for optimization
        cmo_period_range = range(5, 21)  # CMO period starts from 5
        trix_period_range = range(5, 21)  # Example range for Trix period

        # Create an empty matrix to store the total returns for each combination of periods
        total_returns_matrix = np.full((len(cmo_period_range), len(trix_period_range)), np.nan)
        portfolios_dict = {}

        print("最適化ループを開始...")
        # Loop over all combinations of periods and store the total returns
        for i, cmo_period in enumerate(cmo_period_range):
            for j, trix_period in enumerate(trix_period_range):
                # シグナル生成 (4つの配列を受け取る)
                entries, exits, short_entries, short_exits = generate_signals(
                    df['Close'], cmo_period, trix_period
                )

                # エントリーシグナルがない場合はスキップ
                if not np.any(entries) and (not allow_shorting or not np.any(short_entries)):
                    total_returns_matrix[i, j] = -1 # または np.nan
                    continue

                # ポートフォリオ引数の構築
                portfolio_kwargs = {
                    'close': df['Close'],
                    'entries': entries,
                    'exits': exits,
                    'freq': df.index.freq,
                    'init_cash': initial_capital,
                    'fees': 0.001, # 手数料を少し変更
                    'sl_stop': 0.05 # 例: 固定ストップロス 5%
                }
                if allow_shorting:
                    portfolio_kwargs['short_entries'] = short_entries
                    portfolio_kwargs['short_exits'] = short_exits

                # ポートフォリオシミュレーション
                try:
                    portfolio = vbt.Portfolio.from_signals(**portfolio_kwargs)
                    total_returns_matrix[i, j] = portfolio.total_return()
                    portfolios_dict[(cmo_period, trix_period)] = portfolio
                except Exception as pf_exc:
                    print(f"Error running portfolio for CMO={cmo_period}, Trix={trix_period}: {pf_exc}")
                    total_returns_matrix[i, j] = np.nan
            print(f"CMO Period {cmo_period} Done.") # 進捗表示

        # --- 最適化結果の処理 ---
        if np.isnan(total_returns_matrix).all():
            print("有効なリターンが得られませんでした。")
            return None

        # 最適パラメータ特定
        best_cmo_period_idx, best_trix_period_idx = np.unravel_index(
            np.nanargmax(total_returns_matrix), total_returns_matrix.shape
        )
        best_cmo_period = cmo_period_range[best_cmo_period_idx]
        best_trix_period = trix_period_range[best_trix_period_idx]
        best_return = total_returns_matrix[best_cmo_period_idx, best_trix_period_idx]

        print(f"\nBest CMO Period: {best_cmo_period}")
        print(f"Best Trix Period: {best_trix_period}")
        print(f"Best Total Return: {best_return:.2%}")

        # --- プロット --- 
        # 1. ヒートマップ
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            total_returns_matrix,
            annot=False,
            cmap="viridis", # Changed colormap
            xticklabels=trix_period_range,
            yticklabels=cmo_period_range
        )
        plt.title('Total Return Heatmap (CMO vs Trix Periods)')
        plt.xlabel('Trix Period')
        plt.ylabel('CMO Period')
        plt.tight_layout()
        # plt.show() # Temporarily disable plot showing -> Re-enable heatmap plot
        plt.show()

        # 最適ポートフォリオ取得
        best_pf = portfolios_dict.get((best_cmo_period, best_trix_period))

        if best_pf is None:
            print("最適ポートフォリオが見つかりませんでした。")
            return None

        stats = best_pf.stats()

        # Calculate estimated position size -> REMOVED THIS BLOCK
        # position_size = None
        # try:
        #     position_size = ((best_pf.value() - best_pf.cash()) / best_pf.close).fillna(0)
        # except Exception as e:
        #     st.warning(f"ポジションサイズの計算中にエラーが発生しました: {e}")

        # --- Accurate Position Size Calculation --- 
        print("\nCalculating accurate position size from trades...")
        accurate_position_size = pd.Series(0.0, index=df.index) # Initialize with zeros based on original df index
        try:
            records_df = best_pf.trades.records
            if not records_df.empty:
                # Ensure indices are within the bounds of the main df index
                entry_indices = df.index[records_df['entry_idx']]
                exit_indices = df.index[records_df['exit_idx']]
                
                # Calculate entry changes (+size for long, -size for short)
                entry_adj_size = records_df['size'] * np.where(records_df['direction'] == 0, 1, -1)
                entry_changes = pd.Series(entry_adj_size.values, index=entry_indices)

                # Calculate exit changes (-size for long, +size for short)
                exit_adj_size = -entry_adj_size 
                exit_changes = pd.Series(exit_adj_size.values, index=exit_indices)

                # Combine entry and exit changes
                all_changes = pd.concat([entry_changes, exit_changes])

                # Group by index (timestamp) and sum changes happening at the same time
                net_changes = all_changes.groupby(all_changes.index).sum()

                # Align with the main dataframe index and calculate cumulative sum
                aligned_changes, _ = net_changes.align(accurate_position_size, fill_value=0.0)
                accurate_position_size = aligned_changes.cumsum()
                print("Accurate position size calculated successfully.")
            else:
                print("No trades found in records, position size remains 0.")
        except Exception as calc_err:
            print(f"ポジションサイズの正確な計算中にエラー: {calc_err}")
            accurate_position_size = None # Indicate failure
        # --- End Accurate Position Size Calculation ---

        # --- Debugging Final Position ---
        print("\n--- Debugging Final Position --- S")
        try:
            stats_for_debug = best_pf.stats() # Get stats for End Value
            end_value_from_stats = stats_for_debug.get('End Value', np.nan)
            
            print("Last 5 Trade Records:")
            # Use display options to show more columns if needed
            with pd.option_context('display.max_rows', 5, 'display.max_columns', None, 'display.width', 1000):
                print(best_pf.trades.records.tail()) # Print last few trade records
            
            if accurate_position_size is not None:
                 print("\nCalculated Position Size (Last 5 days):")
                 print(accurate_position_size.tail()) # Print last few calculated position values
                 final_calc_pos = accurate_position_size.iloc[-1]
                 prev_calc_pos = accurate_position_size.iloc[-2] if len(accurate_position_size) > 1 else 0.0
                 print(f"Final calculated position size: {final_calc_pos}")
                 print(f"Previous day's calculated position size: {prev_calc_pos}")
                 
                 print("\nPortfolio Value (Last 5 days):")
                 print(best_pf.value().tail())
                 print("\nCash Value (Last 5 days):")
                 print(best_pf.cash().tail())
                 
                 final_cash = best_pf.cash().iloc[-1]
                 final_close = best_pf.close.iloc[-1]
                 
                 print(f"\nStats End Value: {end_value_from_stats:,.2f}")
                 print(f"Final Cash: {final_cash:,.2f}")
                 print(f"Final Close Price: {final_close:,.2f}")
                 
                 # Check consistency: Does Final Cash equal Stats End Value if final position is 0?
                 print(f"Is Final Cash approximately equal to Stats End Value? {np.isclose(final_cash, end_value_from_stats)}")
                 
                 # Calculate what End Value would be if previous day's position was held
                 estimated_end_value_if_held = final_cash + (prev_calc_pos * final_close)
                 print(f"Estimated End Value if prev position was held: {estimated_end_value_if_held:,.2f}")
                 
            else:
                 print("Accurate position size calculation failed.")
                 
            # Also check the status of the last trade directly
            if not best_pf.trades.records.empty:
                 last_trade = best_pf.trades.records.iloc[-1]
                 print(f"\nStatus of the last trade (id={last_trade['id']}): {last_trade['status']}")
                 # Cast indices to int for comparison and indexing
                 entry_idx_int = last_trade['entry_idx'].astype(int)
                 exit_idx_int = last_trade['exit_idx'].astype(int)
                 print(f"  Entry Idx: {entry_idx_int}, Exit Idx: {exit_idx_int}") 
                 last_df_index = df.index[-1]
                 # Check if exit_idx exists and corresponds to the last day
                 if pd.notna(last_trade['exit_idx']) and df.index[exit_idx_int] == last_df_index: 
                     print(f"  Note: Last trade exited on the last day ({last_df_index.date()}).")

        except Exception as debug_err:
            print(f"Error during final position debugging: {debug_err}")
        print("--- Debugging Final Position --- E")
        # --- End Debugging ---

        # Generate portfolio plot for the best parameters
        print("\nPlotting best portfolio performance...")
        try:
            fig_pf = best_pf.plot(subplots=[
                # 'position', # Removed position plot due to warning
                'orders',
                'trade_pnl',
                'cum_returns'
            ])
            fig_pf.show() # Ensure this is enabled
        except Exception as e:
            # Restore original error handling for portfolio plot
            print(f"ポートフォリオプロット中にエラー: {str(e)}")
            print("基本指標を表示:")
            print(f"  累積リターン: {best_pf.cumulative_returns().iloc[-1]:.2%}")
            print(f"  最大ドローダウン: {best_pf.max_drawdown():.2%}")
            print(f"  取引回数: {len(best_pf.orders)}")

        # 3. ポジション推移のプロット (正確な値)
        if accurate_position_size is not None and not accurate_position_size.empty:
            print("\nPlotting accurate positions...")
            plt.figure(figsize=(15, 5))
            # Plot the calculated accurate position size
            plt.plot(accurate_position_size.index, accurate_position_size.values, label='Accurate Position Size') 
            plt.title(f'Accurate Position Size Over Time (CMO={best_cmo_period}, Trix={best_trix_period})') 
            plt.ylabel('Position Size (Shares)')
            plt.xlabel('Date')
            plt.grid(True, linestyle='dotted')
            plt.axhline(0, color='black', linewidth=0.5) # Zero line
            plt.tight_layout()
            plt.show() # Ensure this is enabled
        elif accurate_position_size is not None: # Handle empty case (e.g., no trades)
             print("計算された正確なポジションサイズが空かゼロのみです。プロットをスキップします。")
        else: # Handle calculation error case
             print("ポジションサイズの計算に失敗したため、プロットをスキップします。")

        return best_pf # Return the best portfolio object

    except Exception as e:
        print(f"バックテスト実行中にエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        raise

# メイン処理
def main():
    # パラメータの設定
    symbol = 'AAPL'
    period = '2y'
    interval = '1d'
    initial_capital = 100000
    use_saved_data = False # Set to False to fetch fresh data for testing
    base_dir = 'result'

    try:
        # データの準備
        df = prepare_data(symbol, period, interval, use_saved=use_saved_data, base_dir=base_dir)
        
        # データの基本情報を表示
        print("\n=== データ情報 ===")
        print(f"データ期間: {df.index[0]} から {df.index[-1]}")
        print(f"データ件数: {len(df)}")
        print(f"最初の5行:\n{df.head()}")

        # バックテストの実行 (空売りを有効に)
        portfolio = run_backtest(df, initial_capital, base_dir=base_dir, allow_shorting=True)
        
        # 結果の表示
        if portfolio:
            print("\n=== バックテスト結果 (最適パラメータ) ===")
            print(portfolio.stats())
        else:
            print("バックテストは完了しましたが、有効な結果は得られませんでした。")

    except Exception as e:
        print(f"メイン処理でエラーが発生しました: {str(e)}")


if __name__ == '__main__':
    main()
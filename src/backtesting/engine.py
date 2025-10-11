"""
バックテストエンジンの実装
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import os
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType

from ..data.data_fetcher import DataFetcher
from .strategy import Strategy

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    バックテスト実行エンジン
    ストラテジーを受け取り、バックテストを実行して結果を返す
    """
    
    def __init__(self, use_vectorbt=True):
        """
        バックテストエンジンを初期化する
        
        Args:
            use_vectorbt (bool): データ取得にvectorbtを使用するかどうか (現在は常にTrueとして扱われます)
        """
        self.data_fetcher = DataFetcher(use_vectorbt=True)
        logger.info("BacktestEngine initialized (using vectorbt)")
    
    def fetch_data(self, symbols, period='1y', interval='1d'):
        """
        バックテスト用のデータを取得する
        
        Args:
            symbols (list): データを取得する銘柄のリスト
            period (str): データ取得期間
            interval (str): データの間隔
            
        Returns:
            dict: シンボルをキー、DataFrameを値とする辞書
        """
        return self.data_fetcher.fetch_data(symbols, period=period, interval=interval)
    
    def run_backtest(self, strategy, data, initial_capital=10000.0, commission=0.001):
        """
        指定されたストラテジーに対してバックテストを実行する
        
        Args:
            strategy (Strategy): バックテストするストラテジーのインスタンス
            data (pd.DataFrame): OHLCV価格データを含むDataFrame
            initial_capital (float): 初期資本
            commission (float): 取引手数料（1取引あたりの割合）
            
        Returns:
            dict: バックテスト結果を含む辞書
        """
        if not isinstance(strategy, Strategy):
            raise TypeError("strategy must be an instance of Strategy")
        
        logger.info(f"Running backtest with strategy '{strategy.name}'")
        
        # --- Frequency Handling ---
        # Try to infer frequency from the data
        inferred_freq = pd.infer_freq(data.index)
        
        # If inference fails, try to determine from the interval between first two data points
        if not inferred_freq and len(data.index) > 1:
            # Calculate the time difference between consecutive data points
            time_diff = data.index[1] - data.index[0]
            
            # Map common time differences to frequency strings
            if time_diff <= pd.Timedelta(minutes=1):
                inferred_freq = 'T'  # Minute
            elif time_diff <= pd.Timedelta(minutes=5):
                inferred_freq = '5T'  # 5 Minutes
            elif time_diff <= pd.Timedelta(minutes=15):
                inferred_freq = '15T'  # 15 Minutes
            elif time_diff <= pd.Timedelta(minutes=30):
                inferred_freq = '30T'  # 30 Minutes
            elif time_diff <= pd.Timedelta(hours=1):
                inferred_freq = 'H'  # Hourly
            elif time_diff <= pd.Timedelta(days=1):
                inferred_freq = 'D'  # Daily
            elif time_diff <= pd.Timedelta(weeks=1):
                inferred_freq = 'W'  # Weekly
            else:
                inferred_freq = 'M'  # Monthly (approximate)
        
        if inferred_freq:
            freq_to_use = inferred_freq
            logger.info(f"Inferred data frequency: {freq_to_use}")
        else:
            freq_to_use = 'D' # Default to daily if inference fails
            logger.warning("Could not infer data frequency. Assuming daily ('D').")
        # --- End Frequency Handling ---
        
        signals_df = strategy.generate_signals(data)
        size_series = strategy.calculate_position_size(data, signals_df, initial_capital)
        
        if signals_df is None or not all(col in signals_df.columns for col in ['long_entries', 'long_exits', 'short_entries', 'short_exits']):
            logger.error("Invalid signals data: required signal columns not found")
            portfolio = vbt.Portfolio.from_holding(
                data['Close'], 
                init_cash=initial_capital,
                fees=commission,
                freq=freq_to_use
            )
        else:
            logger.info("Creating portfolio with dynamic size based on 'Amount' (shares)")
            portfolio = vbt.Portfolio.from_signals(
                data['Close'],
                entries=signals_df['long_entries'],
                exits=signals_df['long_exits'],
                short_entries=signals_df['short_entries'],
                short_exits=signals_df['short_exits'],
                size=size_series,
                size_type=SizeType.Amount,
                init_cash=initial_capital,
                fees=commission,
                freq=freq_to_use
            )
        
        stats = portfolio.stats()
        
        # Handle potential None values in stats
        total_return_pct = stats.get('Total Return [%]', 0.0) if stats is not None else 0.0
        total_return = total_return_pct / 100.0 if total_return_pct is not None else 0.0
        
        sharpe_ratio = stats.get('Sharpe Ratio', 0.0) if stats is not None else 0.0
        
        max_drawdown_pct = stats.get('Max Drawdown [%]', 0.0) if stats is not None else 0.0
        max_drawdown = max_drawdown_pct / 100.0 if max_drawdown_pct is not None else 0.0
        
        win_rate_pct = stats.get('Win Rate [%]', None) if stats is not None else None
        win_rate = win_rate_pct / 100.0 if win_rate_pct is not None else None
        
        result = {
            'portfolio': portfolio,
            'stats': stats,
            'signals': signals_df,
            'equity_curve': portfolio.value(),
            'drawdown': getattr(portfolio, 'drawdown', lambda: pd.Series(0, index=data.index))(),
            'trades': portfolio.trades,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
        }
        
        logger.info(f"Backtest completed with total return: {result['total_return']:.2%}")
        return result
    
    def visualize_results(self, result, symbol, strategy_name=None, figsize=(12, 10)):
        """
        バックテスト結果を視覚化する
        
        Args:
            result (dict): run_backtest()から返された結果辞書
            symbol (str): 銘柄シンボル
            strategy_name (str): 戦略名（Noneの場合、result内のstrategy.nameを使用）
            figsize (tuple): 図のサイズ
            
        Returns:
            matplotlib.figure.Figure: 作成された図のオブジェクト
        """
        if strategy_name is None and 'strategy' in result:
            strategy_name = result['strategy'].name
        elif strategy_name is None:
            strategy_name = "Unknown Strategy"
        
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # 1. 価格と売買シグナル
        ax1.plot(result['portfolio'].close.index, result['portfolio'].close, label='Close Price')
        
        # Plot new signals using portfolio.close as the price level
        long_entry_signals = result['portfolio'].close[result['signals']['long_entries']]
        long_exit_signals = result['portfolio'].close[result['signals']['long_exits']]
        short_entry_signals = result['portfolio'].close[result['signals']['short_entries']]
        short_exit_signals = result['portfolio'].close[result['signals']['short_exits']]
        
        ax1.plot(long_entry_signals.index, long_entry_signals, '^', markersize=10, color='g', label='Long Entry')
        ax1.plot(long_exit_signals.index, long_exit_signals, 'v', markersize=10, color='r', label='Long Exit')
        ax1.plot(short_entry_signals.index, short_entry_signals, 'v', markersize=10, color='m', label='Short Entry') # Magenta for short entry
        ax1.plot(short_exit_signals.index, short_exit_signals, '^', markersize=10, color='c', label='Short Exit')   # Cyan for short exit

        # Optionally plot indicators from signals_df if they exist, using portfolio index
        indicator_cols = [col for col in result['signals'].columns if col not in ['long_entries', 'long_exits', 'short_entries', 'short_exits', 'Close']] # Exclude 'Close' if it somehow exists
        for col in indicator_cols:
            ax1.plot(result['portfolio'].close.index, result['signals'][col], label=col, linestyle='--')
        
        ax1.set_ylabel('Price')
        ax1.set_title(f'{symbol} - {strategy_name} Backtest')
        ax1.legend()
        ax1.grid(True)
        
        # 2. 累積リターン
        equity_curve = result['equity_curve']
        equity_curve.plot(ax=ax2, label='Equity Curve')
        ax2.set_ylabel('Equity')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        return fig
    
    def save_results(self, result, symbol, strategy_name, output_dir):
        """
        バックテスト結果を保存する
        
        Args:
            result (dict): バックテスト結果
            symbol (str): 銘柄シンボル
            strategy_name (str): 戦略名
            output_dir (str): 出力ディレクトリ
            
        Returns:
            dict: 保存されたファイルのパスを含む辞書
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_symbol = symbol.replace('/', '-').replace('^', '')
        
        result_paths = {}
        
        if 'signals' in result and result['signals'] is not None:
            signals_path = os.path.join(output_dir, f"{clean_symbol}_{strategy_name}_signals_{timestamp}.csv")
            result['signals'].to_csv(signals_path)
            result_paths['signals'] = signals_path
        
        if 'stats' in result and result['stats'] is not None:
            stats_path = os.path.join(output_dir, f"{clean_symbol}_{strategy_name}_stats_{timestamp}.csv")
            if isinstance(result['stats'], (pd.Series, pd.DataFrame)):
                 result['stats'].to_csv(stats_path)
            else:
                 try:
                     pd.Series(result['stats']).to_csv(stats_path)
                 except Exception as e:
                     logger.warning(f"Could not save stats to CSV: {e}")

            result_paths['stats'] = stats_path
        
        fig = self.visualize_results(result, symbol, strategy_name)
        chart_path = os.path.join(output_dir, f"{clean_symbol}_{strategy_name}_chart_{timestamp}.png")
        fig.savefig(chart_path)
        plt.close(fig)
        result_paths['chart'] = chart_path
        
        if 'trades' in result and result['trades'] is not None and len(result['trades']) > 0:
            trades_path = os.path.join(output_dir, f"{clean_symbol}_{strategy_name}_trades_{timestamp}.csv")
            trades_records = result['trades'].records
            trades_df = pd.DataFrame(trades_records)
            trades_df.to_csv(trades_path)
            result_paths['trades'] = trades_path
        
        logger.info(f"Backtest results saved to {output_dir}")
        return result_paths

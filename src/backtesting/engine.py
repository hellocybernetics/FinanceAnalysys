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
        inferred_freq = pd.infer_freq(data.index)
        if inferred_freq:
            freq_to_use = inferred_freq
            logger.info(f"Inferred data frequency: {freq_to_use}")
        else:
            freq_to_use = 'D' # Default to daily if inference fails
            logger.warning("Could not infer data frequency. Assuming daily ('D').")
        # --- End Frequency Handling ---
        
        signals = strategy.generate_signals(data)
        
        positions = strategy.calculate_position_size(data, signals, initial_capital)
        
        if signals is None or 'signal' not in signals.columns:
            logger.error("Invalid signals data: signal column not found")
            portfolio = vbt.Portfolio.from_holding(
                data['Close'], 
                init_cash=initial_capital,
                fees=commission,
                freq=freq_to_use # Use inferred/default frequency
            )
        else:
            portfolio = vbt.Portfolio.from_signals(
                data['Close'],
                entries=signals['signal'] > 0,
                exits=signals['signal'] < 0,
                init_cash=initial_capital,
                fees=commission,
                freq=freq_to_use # Use inferred/default frequency
            )
        
        stats = portfolio.stats()
        
        result = {
            'portfolio': portfolio,
            'stats': stats,
            'signals': signals,
            'equity_curve': portfolio.value(),
            'drawdown': portfolio.drawdown(),
            'trades': portfolio.trades,
            'total_return': stats.get('Total Return [%]', 0.0) / 100.0,
            'sharpe_ratio': stats.get('Sharpe Ratio', 0.0),
            'max_drawdown': stats.get('Max Drawdown [%]', 0.0) / 100.0,
            'win_rate': stats.get('Win Rate [%]', None) / 100.0 if stats.get('Win Rate [%]') is not None else None,
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
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        ax1 = axes[0]
        if 'signals' in result and result['signals'] is not None and 'Close' in result['signals'].columns:
            result['signals']['Close'].plot(ax=ax1, label='Close Price')
            
            if 'signal' in result['signals'].columns:
                buy_signals = result['signals'][result['signals']['signal'] > 0]
                sell_signals = result['signals'][result['signals']['signal'] < 0]
                
                if not buy_signals.empty:
                    ax1.plot(buy_signals.index, buy_signals['Close'], '^', markersize=10, color='g', label='Buy Signal')
                if not sell_signals.empty:
                    ax1.plot(sell_signals.index, sell_signals['Close'], 'v', markersize=10, color='r', label='Sell Signal')
            
            for col in result['signals'].columns:
                if 'SMA' in col or 'EMA' in col:
                    result['signals'][col].plot(ax=ax1, label=col)
        
        ax1.set_ylabel('Price')
        ax1.set_title(f'{symbol} - {strategy_name} Backtest Results')
        ax1.legend()
        ax1.grid(True)
        
        ax2 = axes[1]
        if 'equity_curve' in result and result['equity_curve'] is not None:
            result['equity_curve'].plot(ax=ax2, label='Equity Curve')
        ax2.set_ylabel('Portfolio Value')
        ax2.legend()
        ax2.grid(True)
        
        ax3 = axes[2]
        if 'drawdown' in result and result['drawdown'] is not None:
            result['drawdown'].plot(ax=ax3, label='Drawdown', color='red')
        ax3.set_ylabel('Drawdown %')
        ax3.set_xlabel('Date')
        ax3.legend()
        ax3.grid(True)
        
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

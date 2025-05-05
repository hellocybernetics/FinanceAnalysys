"""
戦略（Strategy）のベースクラスとサンプル実装
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging
import vectorbt as vbt # Import vectorbt

logger = logging.getLogger(__name__)

class Strategy(ABC):
    """
    トレーディング戦略のベースクラス。
    すべての具体的な戦略はこのクラスを継承し、必要なメソッドを実装する必要がある。
    """
    
    def __init__(self, name="BaseStrategy"):
        """
        戦略を初期化する
        
        Args:
            name (str): 戦略の名前
        """
        self.name = name
        logger.info(f"Strategy '{name}' initialized")
    
    @abstractmethod
    def generate_signals(self, data):
        """
        価格データに基づいて取引シグナルを生成する
        
        Args:
            data (pd.DataFrame): OHLCV価格データを含むDataFrame
            
        Returns:
            pd.DataFrame: 'long_entries', 'long_exits', 'short_entries', 'short_exits' の
                          4つのブール列を含むDataFrame。
        """
        pass
    
    def calculate_position_size(self, data, signals, initial_capital=10000.0):
        """
        各取引に対するポジションサイズを計算する（サブクラスで実装する必要あり）
        
        Args:
            data (pd.DataFrame): OHLCV価格データを含むDataFrame
            signals (pd.DataFrame): 'signal'列を含むDataFrame
            initial_capital (float): 初期資本
            
        Returns:
            pd.Series: 各時点での取引する量 (例: 株数)。シグナルがない時点はNaNまたは0。
        """
        # Default implementation removed, forcing subclasses to implement calculate_position_size
        raise NotImplementedError("Subclasses must implement calculate_position_size")
        # return signals['signal'] * (initial_capital / data['Close'].iloc[0]) # Old static logic


class MovingAverageCrossoverStrategy(Strategy):
    """
    移動平均交差戦略の実装例
    短期移動平均が長期移動平均を上回ったら買い、下回ったら売り
    """
    
    def __init__(self, short_window=20, long_window=50, name="MA_Crossover"):
        """
        移動平均交差戦略を初期化する
        
        Args:
            short_window (int): 短期移動平均の期間
            long_window (int): 長期移動平均の期間
            name (str): 戦略の名前
        """
        super().__init__(name=name)
        self.short_window = short_window
        self.long_window = long_window
        logger.info(f"MA Crossover Strategy initialized with short_window={short_window}, long_window={long_window}")
    
    def generate_signals(self, data):
        """
        移動平均交差に基づいてシグナルを生成する (vectorbt使用)
        
        Args:
            data (pd.DataFrame): OHLCV価格データを含むDataFrame
            
        Returns:
            pd.DataFrame: long_entries, long_exits, short_entries, short_exits を含むDataFrame
        """
        close = data['Close']
        # Calculate SMAs using vectorbt
        short_ma_indicator = vbt.MA.run(close, window=self.short_window, short_name='fast')
        long_ma_indicator = vbt.MA.run(close, window=self.long_window, short_name='slow')
        
        # Generate long entry/exit signals based on crossover
        long_entries = short_ma_indicator.ma_crossed_above(long_ma_indicator.ma)
        long_exits = short_ma_indicator.ma_crossed_below(long_ma_indicator.ma)
        
        # This strategy doesn't generate short signals
        short_entries = pd.Series(False, index=data.index)
        short_exits = pd.Series(False, index=data.index)
        
        signals_df = pd.DataFrame({
            'long_entries': long_entries,
            'long_exits': long_exits,
            'short_entries': short_entries,
            'short_exits': short_exits,
            # Optionally include indicator values for visualization/debugging
            f'SMA_{self.short_window}': short_ma_indicator.ma,
            f'SMA_{self.long_window}': long_ma_indicator.ma
        })

        return signals_df

    def calculate_position_size(self, data, signals, initial_capital=10000.0):
        """固定株数（例: 10株）を取引するサイズ指定を生成"""
        size_series = pd.Series(0.0, index=signals.index) # Default size 0
        # Trade 10 shares on long entry or exit signals
        size_series[signals['long_entries'] | signals['long_exits']] = 10.0
        return size_series


class RSIStrategy(Strategy):
    """
    RSI（相対力指数）に基づく戦略の実装例
    RSIが下限値以下になったら買い、上限値以上になったら売り
    """
    
    def __init__(self, rsi_period=14, oversold=30, overbought=70, name="RSI_Strategy"):
        """
        RSI戦略を初期化する
        
        Args:
            rsi_period (int): RSIの計算期間
            oversold (int): 買いシグナルを出す閾値（この値以下でRSIは買われ過ぎと判断）
            overbought (int): 売りシグナルを出す閾値（この値以上でRSIは売られ過ぎと判断）
            name (str): 戦略の名前
        """
        super().__init__(name=name)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        logger.info(f"RSI Strategy initialized with period={rsi_period}, oversold={oversold}, overbought={overbought}")
    
    def generate_signals(self, data):
        """
        RSIに基づいてシグナルを生成する (vectorbt使用)
        
        Args:
            data (pd.DataFrame): OHLCV価格データを含むDataFrame
            
        Returns:
            pd.DataFrame: long_entries, long_exits, short_entries, short_exits を含むDataFrame
        """
        close = data['Close']
        # Calculate RSI using vectorbt
        rsi_indicator = vbt.RSI.run(close, window=self.rsi_period)
        rsi = rsi_indicator.rsi
        
        # Generate signals based on oversold/overbought levels
        long_entries = rsi < self.oversold
        short_exits = long_entries # Exit short when oversold is reached
        
        short_entries = rsi > self.overbought
        long_exits = short_entries # Exit long when overbought is reached
        
        signals_df = pd.DataFrame({
            'long_entries': long_entries,
            'long_exits': long_exits,
            'short_entries': short_entries,
            'short_exits': short_exits,
            # Optionally include RSI value
            'RSI': rsi
        })
        
        return signals_df

    def calculate_position_size(self, data, signals, initial_capital=10000.0):
        """固定株数（例: 10株）を取引するサイズ指定を生成"""
        size_series = pd.Series(0.0, index=signals.index) # Default size 0
        # Trade 10 shares when any signal occurs
        size_series[signals['long_entries'] | signals['long_exits'] | signals['short_entries'] | signals['short_exits']] = 10.0
        return size_series

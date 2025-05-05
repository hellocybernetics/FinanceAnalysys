"""
戦略（Strategy）のベースクラスとサンプル実装
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging

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
            pd.DataFrame: 'signal'列を含むDataFrame。通常、1=買い, -1=売り, 0=ホールドを表す
        """
        pass
    
    def calculate_position_size(self, data, signals, initial_capital=10000.0):
        """
        各取引に対するポジションサイズを計算する（オプションでオーバーライド可能）
        
        Args:
            data (pd.DataFrame): OHLCV価格データを含むDataFrame
            signals (pd.DataFrame): 'signal'列を含むDataFrame
            initial_capital (float): 初期資本
            
        Returns:
            pd.Series: 各時点での保有する量
        """
        return signals['signal'] * (initial_capital / data['Close'].iloc[0])


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
        移動平均交差に基づいてシグナルを生成する
        
        Args:
            data (pd.DataFrame): OHLCV価格データを含むDataFrame
            
        Returns:
            pd.DataFrame: 'signal'列を含むDataFrame
        """
        signals = data.copy()
        signals['signal'] = 0
        
        signals[f'SMA_{self.short_window}'] = signals['Close'].rolling(window=self.short_window).mean()
        signals[f'SMA_{self.long_window}'] = signals['Close'].rolling(window=self.long_window).mean()
        
        signals['signal'] = 0 # Default to 0
        signals.loc[signals[f'SMA_{self.short_window}'] > signals[f'SMA_{self.long_window}'], 'signal'] = 1  # Buy signal
        signals.loc[signals[f'SMA_{self.short_window}'] < signals[f'SMA_{self.long_window}'], 'signal'] = -1 # Sell signal
        
        return signals


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
        RSIに基づいてシグナルを生成する
        
        Args:
            data (pd.DataFrame): OHLCV価格データを含むDataFrame
            
        Returns:
            pd.DataFrame: 'signal'列を含むDataFrame
        """
        signals = data.copy()
        signals['signal'] = 0
        
        delta = signals['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        signals['RSI'] = 100 - (100 / (1 + rs))
        
        signals['signal'] = np.where(signals['RSI'] < self.oversold, 1, 0)
        signals['signal'] = np.where(signals['RSI'] > self.overbought, -1, signals['signal'])
        
        signals['position'] = signals['signal'].diff()
        
        return signals

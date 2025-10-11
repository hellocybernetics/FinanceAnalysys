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
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
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
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        移動平均交差に基づいてシグナルを生成する (vectorbt使用)
        
        Args:
            data (pd.DataFrame): OHLCV価格データを含むDataFrame
            
        Returns:
            pd.DataFrame: long_entries, long_exits, short_entries, short_exits を含むDataFrame
        """
        close = data['Close']
        # Calculate SMAs using vectorbt
        short_ma_result = vbt.MA.run(close, window=self.short_window)
        long_ma_result = vbt.MA.run(close, window=self.long_window)
        
        # Safely access the ma attribute
        # For vectorbt indicators, we need to handle different return types
        short_ma_values = None
        long_ma_values = None
        
        try:
            # Try to get the ma values directly using getattr with default
            short_ma_values = getattr(short_ma_result, 'ma', None)
            long_ma_values = getattr(long_ma_result, 'ma', None)
        except:
            pass
            
        # If we couldn't get the ma attribute, try other approaches
        if short_ma_values is None:
            try:
                if hasattr(short_ma_result, '__getitem__') and len(short_ma_result) > 0:
                    # If it's a tuple or similar, try to get the first element
                    short_ma_values = short_ma_result[0]
                else:
                    # Fallback to pandas
                    short_ma_values = close.rolling(window=self.short_window).mean()
            except:
                # Final fallback to pandas
                short_ma_values = close.rolling(window=self.short_window).mean()
                
        if long_ma_values is None:
            try:
                if hasattr(long_ma_result, '__getitem__') and len(long_ma_result) > 0:
                    # If it's a tuple or similar, try to get the first element
                    long_ma_values = long_ma_result[0]
                else:
                    # Fallback to pandas
                    long_ma_values = close.rolling(window=self.long_window).mean()
            except:
                # Final fallback to pandas
                long_ma_values = close.rolling(window=self.long_window).mean()
        
        # If we still don't have valid values, fallback to pandas
        if not isinstance(short_ma_values, (pd.Series, np.ndarray)) or len(short_ma_values) != len(close):
            short_ma_values = close.rolling(window=self.short_window).mean()
        if not isinstance(long_ma_values, (pd.Series, np.ndarray)) or len(long_ma_values) != len(close):
            long_ma_values = close.rolling(window=self.long_window).mean()
        
        # Convert to pandas Series if they are numpy arrays
        if isinstance(short_ma_values, np.ndarray):
            short_ma_values = pd.Series(short_ma_values, index=close.index)
        if isinstance(long_ma_values, np.ndarray):
            long_ma_values = pd.Series(long_ma_values, index=close.index)
        
        # Generate long entry/exit signals based on crossover
        long_entries = short_ma_values > long_ma_values
        long_exits = short_ma_values < long_ma_values
        
        # Generate short entry/exit signals based on crossover (opposite of long signals)
        short_entries = short_ma_values < long_ma_values
        short_exits = short_ma_values > long_ma_values
        
        # Shift signals to avoid look-ahead bias
        long_entries = long_entries.shift(1).fillna(False)
        long_exits = long_exits.shift(1).fillna(False)
        short_entries = short_entries.shift(1).fillna(False)
        short_exits = short_exits.shift(1).fillna(False)
        
        signals_df = pd.DataFrame({
            'long_entries': long_entries,
            'long_exits': long_exits,
            'short_entries': short_entries,
            'short_exits': short_exits,
            # Optionally include indicator values for visualization/debugging
            f'SMA_{self.short_window}': short_ma_values,
            f'SMA_{self.long_window}': long_ma_values
        })

        return signals_df

    def calculate_position_size(self, data, signals, initial_capital=10000.0):
        """固定株数（例: 10株）を取引するサイズ指定を生成"""
        size_series = pd.Series(0.0, index=signals.index) # Default size 0
        # Trade 10 shares on any entry or exit signals
        size_series[signals['long_entries'] | signals['long_exits'] | signals['short_entries'] | signals['short_exits']] = 10.0
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
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
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
        
        # Safely access the rsi attribute
        try:
            rsi = getattr(rsi_indicator, 'rsi', close.rolling(window=self.rsi_period).apply(self._rsi_pandas, raw=False))
        except:
            # Fallback to pandas if vectorbt fails
            rsi = close.rolling(window=self.rsi_period).apply(self._rsi_pandas, raw=False)
        
        # Convert to pandas Series if it's not already
        if not isinstance(rsi, pd.Series):
            rsi = pd.Series(rsi, index=close.index)
        
        # Generate signals based on oversold/overbought levels
        # Long entry when RSI crosses below oversold level (indicating upward momentum)
        long_entries = rsi < self.oversold
        # Long exit when RSI crosses above oversold level
        long_exits = rsi > self.oversold
        
        # Short entry when RSI crosses above overbought level (indicating downward momentum)
        short_entries = rsi > self.overbought
        # Short exit when RSI crosses below overbought level
        short_exits = rsi < self.overbought
        
        # Shift signals to avoid look-ahead bias
        long_entries = long_entries.shift(1).fillna(False)
        long_exits = long_exits.shift(1).fillna(False)
        short_entries = short_entries.shift(1).fillna(False)
        short_exits = short_exits.shift(1).fillna(False)
        
        signals_df = pd.DataFrame({
            'long_entries': long_entries,
            'long_exits': long_exits,
            'short_entries': short_entries,
            'short_exits': short_exits,
            # Optionally include RSI value
            'RSI': rsi
        })
        
        return signals_df

    def _rsi_pandas(self, prices):
        """Calculate RSI using pandas"""
        # Make sure we have enough data
        if len(prices) < self.rsi_period + 1:
            return 50  # Return neutral RSI value if not enough data
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        # Avoid division by zero
        rs = gain / loss.where(loss != 0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if len(rsi) > 0 else 50

    def calculate_position_size(self, data, signals, initial_capital=10000.0):
        """固定株数（例: 10株）を取引するサイズ指定を生成"""
        size_series = pd.Series(0.0, index=signals.index) # Default size 0
        # Trade 10 shares when any signal occurs
        size_series[signals['long_entries'] | signals['long_exits'] | signals['short_entries'] | signals['short_exits']] = 10.0
        return size_series

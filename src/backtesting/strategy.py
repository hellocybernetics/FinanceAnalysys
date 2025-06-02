"""
戦略（Strategy）のベースクラスとサンプル実装
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging
from src.analysis.technical_indicators import TechnicalAnalysis

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

class GenericUserStrategy(Strategy):
    """
    A generic strategy that generates signals based on user-defined rules
    and technical indicators.
    """

    def __init__(self, strategy_config, name="GenericUserStrategy"):
        """
        Initializes the GenericUserStrategy.

        Args:
            strategy_config (dict): A dictionary containing the strategy's configuration.
                Expected keys:
                'indicators' (list): A list of indicator configurations, e.g.,
                                     [{'name': 'SMA', 'params': {'length': 20}},
                                      {'name': 'RSI', 'params': {'length': 14}}]
                'buy_condition' (str): A string representing the buy condition, e.g.,
                                       "SMA_20 > Close AND RSI_14 < 30"
                'sell_condition' (str): A string representing the sell condition, e.g.,
                                        "SMA_20 < Close OR RSI_14 > 70"
            name (str): The name of the strategy.
        """
        super().__init__(name=name) # Pass the name to the superclass
        self.indicators_config = strategy_config.get('indicators', [])
        self.buy_condition_str = strategy_config.get('buy_condition', 'False') 
        self.sell_condition_str = strategy_config.get('sell_condition', 'False')
        self.technical_analyzer = TechnicalAnalysis()
        self.evaluation_error = None  # Initialize error storage
        # Ensure logger is accessible, e.g. module-level logger
        logger.info(f"GenericUserStrategy '{self.name}' initialized with buy: '{self.buy_condition_str}', sell: '{self.sell_condition_str}'")

    def generate_signals(self, data):
        """
        Generates trading signals based on the configured indicators and conditions.

        Args:
            data (pd.DataFrame): OHLCV price data.

        Returns:
            pd.DataFrame: DataFrame with a 'signal' column (1=buy, -1=sell, 0=hold).
        """
        # Reset error at the beginning of signal generation
        self.evaluation_error = None

        if data.empty:
            logger.warning(f"Strategy '{self.name}': Input data is empty. Cannot generate signals.")
            signals_df = pd.DataFrame(index=data.index)
            signals_df['signal'] = 0
            return signals_df[['signal']]

        signals_df = data.copy()
        
        try:
            signals_df = self.technical_analyzer.calculate_indicators(signals_df, self.indicators_config)
        except Exception as e:
            logger.error(f"Strategy '{self.name}': Error calculating technical indicators: {e}", exc_info=True)
            signals_df['signal'] = 0
            return signals_df[['signal']]

        signals_df['signal'] = 0 # Initialize signal column

        try:
            if self.buy_condition_str and self.buy_condition_str.strip():
                buy_signals_mask = pd.eval(self.buy_condition_str, engine='python', local_dict=signals_df)
                signals_df.loc[buy_signals_mask, 'signal'] = 1
            else:
                logger.info(f"Strategy '{self.name}': Buy condition is empty or not a valid string. No buy signals generated.")
        except Exception as e:
            logger.error(f"Strategy '{self.name}': Error evaluating buy condition '{self.buy_condition_str}': {e}", exc_info=True)
            self.evaluation_error = f"Error in buy condition '{self.buy_condition_str[:50]}...': {e}"

        try:
            if self.sell_condition_str and self.sell_condition_str.strip():
                sell_signals_mask = pd.eval(self.sell_condition_str, engine='python', local_dict=signals_df)
                signals_df.loc[sell_signals_mask, 'signal'] = np.where(signals_df.loc[sell_signals_mask, 'signal'] == 1, -1, -1) # If buy and sell on same bar, sell wins.
            else:
                logger.info(f"Strategy '{self.name}': Sell condition is empty or not a valid string. No sell signals generated.")
        except Exception as e:
            logger.error(f"Strategy '{self.name}': Error evaluating sell condition '{self.sell_condition_str}': {e}", exc_info=True)
            error_message = f"Error in sell condition '{self.sell_condition_str[:50]}...': {e}"
            if self.evaluation_error is None:
                self.evaluation_error = error_message
            else:
                self.evaluation_error += f"; {error_message}" # Append if there was also a buy error
            
        signals_df['signal'] = signals_df['signal'].astype(int)
        
        logger.info(f"Strategy '{self.name}': Signals generated. Buys: {(signals_df['signal'] == 1).sum()}, Sells: {(signals_df['signal'] == -1).sum()}")
        
        return signals_df[['signal']]

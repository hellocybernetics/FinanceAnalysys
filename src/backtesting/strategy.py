"""
Trading strategy classes for backtesting.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    All strategy implementations should inherit from this class and implement
    the generate_signals method.
    """

    def __init__(self, name: str):
        """
        Initialize the strategy.

        Args:
            name (str): Name of the strategy.
        """
        self.name = name

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the strategy logic.

        Args:
            df (pd.DataFrame): DataFrame containing OHLCV data and technical indicators.

        Returns:
            pd.DataFrame: DataFrame with added 'signal' column where:
                - 1.0 indicates a buy signal
                - -1.0 indicates a sell signal
                - 0.0 indicates no signal (hold)
        """
        pass

    def validate_dataframe(self, df: pd.DataFrame, required_columns: list) -> None:
        """
        Validate that the DataFrame contains required columns.

        Args:
            df (pd.DataFrame): DataFrame to validate.
            required_columns (list): List of required column names.

        Raises:
            ValueError: If required columns are missing.
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame is missing required columns: {missing_columns}")


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover strategy.

    Generates buy signals when short MA crosses above long MA,
    and sell signals when short MA crosses below long MA.
    """

    def __init__(self, short_length: int = 20, long_length: int = 50):
        """
        Initialize the Moving Average Crossover strategy.

        Args:
            short_length (int): Length of the short moving average.
            long_length (int): Length of the long moving average.
        """
        super().__init__(f"MA_Crossover_{short_length}_{long_length}")
        self.short_length = short_length
        self.long_length = long_length

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on moving average crossover.

        Args:
            df (pd.DataFrame): DataFrame containing OHLCV data.

        Returns:
            pd.DataFrame: DataFrame with added signal columns.
        """
        result_df = df.copy()

        # Validate required columns
        self.validate_dataframe(result_df, ['Close'])

        # Calculate moving averages if not already present
        short_ma_col = f'SMA_{self.short_length}'
        long_ma_col = f'SMA_{self.long_length}'

        if short_ma_col not in result_df.columns:
            result_df[short_ma_col] = result_df['Close'].rolling(window=self.short_length).mean()

        if long_ma_col not in result_df.columns:
            result_df[long_ma_col] = result_df['Close'].rolling(window=self.long_length).mean()

        # Initialize signal column
        result_df['signal'] = 0.0

        # Generate signals based on crossover
        # Buy when short MA crosses above long MA
        result_df.loc[
            (result_df[short_ma_col] > result_df[long_ma_col]) &
            (result_df[short_ma_col].shift(1) <= result_df[long_ma_col].shift(1)),
            'signal'
        ] = 1.0

        # Sell when short MA crosses below long MA
        result_df.loc[
            (result_df[short_ma_col] < result_df[long_ma_col]) &
            (result_df[short_ma_col].shift(1) >= result_df[long_ma_col].shift(1)),
            'signal'
        ] = -1.0

        logger.info(f"Generated signals for {self.name}")
        return result_df


class RSIStrategy(BaseStrategy):
    """
    RSI (Relative Strength Index) strategy.

    Generates buy signals when RSI crosses above oversold level,
    and sell signals when RSI crosses below overbought level.
    """

    def __init__(self, rsi_length: int = 14, oversold: float = 30, overbought: float = 70):
        """
        Initialize the RSI strategy.

        Args:
            rsi_length (int): Length of the RSI indicator.
            oversold (float): Oversold threshold (typically 30).
            overbought (float): Overbought threshold (typically 70).
        """
        super().__init__(f"RSI_{rsi_length}_{oversold}_{overbought}")
        self.rsi_length = rsi_length
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on RSI levels.

        Args:
            df (pd.DataFrame): DataFrame containing OHLCV data.

        Returns:
            pd.DataFrame: DataFrame with added signal columns.
        """
        result_df = df.copy()

        # Validate required columns
        self.validate_dataframe(result_df, ['Close'])

        # Calculate RSI if not already present
        rsi_col = f'RSI_{self.rsi_length}'

        if rsi_col not in result_df.columns:
            # Calculate RSI using Wilder's smoothing method
            delta = result_df['Close'].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)

            avg_gain = pd.Series(np.nan, index=result_df.index)
            avg_loss = pd.Series(np.nan, index=result_df.index)

            avg_gain.iloc[self.rsi_length] = gain.iloc[1:self.rsi_length+1].mean()
            avg_loss.iloc[self.rsi_length] = loss.iloc[1:self.rsi_length+1].mean()

            for i in range(self.rsi_length + 1, len(result_df)):
                avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (self.rsi_length - 1) + gain.iloc[i]) / self.rsi_length
                avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (self.rsi_length - 1) + loss.iloc[i]) / self.rsi_length

            rs = avg_gain / avg_loss
            result_df[rsi_col] = 100 - (100 / (1 + rs))

        # Initialize signal column
        result_df['signal'] = 0.0

        # Generate signals based on RSI levels
        # Buy when RSI crosses above oversold level
        result_df.loc[
            (result_df[rsi_col] > self.oversold) &
            (result_df[rsi_col].shift(1) <= self.oversold),
            'signal'
        ] = 1.0

        # Sell when RSI crosses below overbought level
        result_df.loc[
            (result_df[rsi_col] < self.overbought) &
            (result_df[rsi_col].shift(1) >= self.overbought),
            'signal'
        ] = -1.0

        logger.info(f"Generated signals for {self.name}")
        return result_df

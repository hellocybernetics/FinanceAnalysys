"""
Technical analysis module for calculating technical indicators using vectorbt.
"""

import pandas as pd
import numpy as np
import logging

try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    logging.warning("vectorbt not available, some indicators will not work")

logger = logging.getLogger(__name__)

class TechnicalAnalysis:
    """
    Class for performing technical analysis on financial data.
    """
    
    def __init__(self):
        """
        Initialize the TechnicalAnalysis class.
        """
        self.indicator_calculators = {
            'SMA': self._calculate_sma,
            'EMA': self._calculate_ema,
            'RSI': self._calculate_rsi,
            'MACD': self._calculate_macd,
            'BBands': self._calculate_bbands,
            'ATR': self._calculate_atr,
            'Stochastic': self._calculate_stochastic,
            'ADX': self._calculate_adx,
            'WILLR': self._calculate_willr,
        }
    
    def _calculate_sma(self, result_df, params):
        """Calculates Simple Moving Average (SMA)."""
        length = params.get('length', 20)
        if VECTORBT_AVAILABLE:
            sma = vbt.MA.run(result_df['Close'], length, short_name=f'SMA_{length}')
            result_df[f'SMA_{length}'] = sma.ma
        else:
            result_df[f'SMA_{length}'] = result_df['Close'].rolling(window=length).mean()
        logger.info(f"Calculated SMA with length {length}")

    def _calculate_ema(self, result_df, params):
        """Calculates Exponential Moving Average (EMA)."""
        length = params.get('length', 50)
        if VECTORBT_AVAILABLE:
            ema = vbt.MA.run(result_df['Close'], length, short_name=f'EMA_{length}', ewm=True)
            result_df[f'EMA_{length}'] = ema.ma
        else:
            result_df[f'EMA_{length}'] = result_df['Close'].ewm(span=length, adjust=False).mean()
        logger.info(f"Calculated EMA with length {length}")

    def _calculate_rsi(self, result_df, params):
        """Calculates Relative Strength Index (RSI)."""
        length = params.get('length', 14)
        if VECTORBT_AVAILABLE:
            rsi = vbt.RSI.run(result_df['Close'], window=length, short_name=f'RSI_{length}')
            result_df[f'RSI_{length}'] = rsi.rsi
        else:
            # Calculate RSI using pandas
            delta = result_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
            rs = gain / loss
            result_df[f'RSI_{length}'] = 100 - (100 / (1 + rs))
        logger.info(f"Calculated RSI with length {length}")

    def _calculate_macd(self, result_df, params):
        """Calculates Moving Average Convergence Divergence (MACD)."""
        fast = params.get('fast', 12)
        slow = params.get('slow', 26)
        signal = params.get('signal', 9)

        if VECTORBT_AVAILABLE:
            fast_ema = vbt.MA.run(result_df['Close'], fast, short_name=f'EMA_{fast}', ewm=True)
            slow_ema = vbt.MA.run(result_df['Close'], slow, short_name=f'EMA_{slow}', ewm=True)
            macd_line = fast_ema.ma - slow_ema.ma
            signal_line = vbt.MA.run(macd_line, signal, short_name=f'Signal_{signal}', ewm=True).ma
        else:
            # Calculate MACD using pandas
            fast_ema = result_df['Close'].ewm(span=fast, adjust=False).mean()
            slow_ema = result_df['Close'].ewm(span=slow, adjust=False).mean()
            macd_line = fast_ema - slow_ema
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        histogram = macd_line - signal_line

        result_df[f'MACD_{fast}_{slow}'] = macd_line
        result_df[f'MACD_Signal_{signal}'] = signal_line
        result_df[f'MACD_Hist_{fast}_{slow}_{signal}'] = histogram
        logger.info(f"Calculated MACD with fast={fast}, slow={slow}, signal={signal}")

    def _calculate_bbands(self, result_df, params):
        """Calculates Bollinger Bands."""
        length = params.get('length', 20)
        std = params.get('std', 2)

        if VECTORBT_AVAILABLE:
            middle = vbt.MA.run(result_df['Close'], length, short_name=f'SMA_{length}').ma
        else:
            middle = result_df['Close'].rolling(window=length).mean()
            
        rolling_std = result_df['Close'].rolling(window=length).std()

        upper = middle + (rolling_std * std)
        lower = middle - (rolling_std * std)

        result_df[f'BBL_{length}_{std}'] = lower
        result_df[f'BBM_{length}_{std}'] = middle
        result_df[f'BBU_{length}_{std}'] = upper
        logger.info(f"Calculated Bollinger Bands with length={length}, std={std}")

    def _calculate_atr(self, result_df, params):
        """Calculates Average True Range (ATR)."""
        length = params.get('length', 14)
        if VECTORBT_AVAILABLE:
            atr = vbt.ATR.run(
                high=result_df['High'],
                low=result_df['Low'],
                close=result_df['Close'],
                window=length,
                short_name=f'ATR_{length}'
            )
            result_df[f'ATR_{length}'] = atr.atr
        else:
            # Calculate ATR using pandas
            high = result_df['High']
            low = result_df['Low']
            close = result_df['Close']
            tr1 = abs(high - low)
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            result_df[f'ATR_{length}'] = tr.rolling(window=length).mean()
        logger.info(f"Calculated ATR with length {length}")

    def _calculate_stochastic(self, result_df, params):
        """Calculates Stochastic Oscillator."""
        k = params.get('k', 14)
        d = params.get('d', 3)

        if VECTORBT_AVAILABLE:
            try:
                stoch_factory = vbt.IndicatorFactory.from_talib('STOCH')
                stoch = stoch_factory.run(
                    high=result_df['High'],
                    low=result_df['Low'],
                    close=result_df['Close'],
                    fastk_period=k,
                    slowk_period=d,
                    slowd_period=d
                )
                result_df[f'STOCHk_{k}_{d}'] = stoch.slowk
                result_df[f'STOCHd_{k}_{d}'] = stoch.slowd
                logger.info(f"Calculated Stochastic with k={k}, d={d} using TA-Lib")
            except Exception as e:
                logger.error(f"Error calculating Stochastic: {e}. Make sure TA-Lib is installed and data has sufficient length.")
                self._calculate_stochastic_pandas(result_df, k, d)
        else:
            self._calculate_stochastic_pandas(result_df, k, d)
            
    def _calculate_stochastic_pandas(self, result_df, k, d):
        """Calculate Stochastic Oscillator using pandas."""
        high_k = result_df['High'].rolling(window=k).max()
        low_k = result_df['Low'].rolling(window=k).min()
        
        k_percent = 100 * ((result_df['Close'] - low_k) / (high_k - low_k))
        k_percent_d = k_percent.rolling(window=d).mean()
        d_percent = k_percent_d.rolling(window=d).mean()
        
        result_df[f'STOCHk_{k}_{d}'] = k_percent_d
        result_df[f'STOCHd_{k}_{d}'] = d_percent
        logger.info(f"Calculated Stochastic with k={k}, d={d} using pandas")

    def _calculate_adx(self, result_df, params):
        """Calculates Average Directional Index (ADX)."""
        length = params.get('length', 14)
        if VECTORBT_AVAILABLE:
            try:
                adx_factory = vbt.IndicatorFactory.from_talib('ADX')
                adx_output = adx_factory.run(
                    high=result_df['High'],
                    low=result_df['Low'],
                    close=result_df['Close'],
                    timeperiod=length
                )
                result_df[f'ADX_{length}'] = adx_output.real
                logger.info(f"Calculated ADX with length {length} using TA-Lib")
            except Exception as e:
                logger.error(f"Error calculating ADX: {e}. Make sure TA-Lib is installed and data has sufficient length.")
                result_df[f'ADX_{length}'] = np.nan
        else:
            logger.warning(f"ADX calculation requires TA-Lib. Using placeholder values.")
            result_df[f'ADX_{length}'] = np.nan

    def _calculate_willr(self, result_df, params):
        """Calculates Williams %R."""
        length = params.get('length', 14)
        if VECTORBT_AVAILABLE:
            try:
                willr_factory = vbt.IndicatorFactory.from_talib('WILLR')
                willr_output = willr_factory.run(
                    high=result_df['High'],
                    low=result_df['Low'],
                    close=result_df['Close'],
                    timeperiod=length
                )
                result_df[f'WILLR_{length}'] = willr_output.real
                logger.info(f"Calculated Williams %R with length {length} using TA-Lib")
            except Exception as e:
                logger.error(f"Error calculating Williams %R: {e}. Make sure TA-Lib is installed and data has sufficient length.")
                self._calculate_willr_pandas(result_df, length)
        else:
            self._calculate_willr_pandas(result_df, length)
            
    def _calculate_willr_pandas(self, result_df, length):
        """Calculate Williams %R using pandas."""
        high_max = result_df['High'].rolling(window=length).max()
        low_min = result_df['Low'].rolling(window=length).min()
        
        willr = ((high_max - result_df['Close']) / (high_max - low_min)) * -100
        result_df[f'WILLR_{length}'] = willr
        logger.info(f"Calculated Williams %R with length {length} using pandas")

    def calculate_indicators(self, df, indicators):
        """
        Calculate technical indicators for the given dataframe using vectorbt.
        
        Args:
            df (pd.DataFrame): Dataframe containing OHLCV data.
            indicators (list): List of indicator configurations.
            
        Returns:
            pd.DataFrame: Dataframe with added technical indicators.
        """
        result_df = df.copy()
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in result_df.columns]
        
        if missing_columns:
            logger.error(f"Dataframe is missing required columns: {missing_columns}")
            raise ValueError(f"Dataframe is missing required columns: {missing_columns}")
        
        for indicator in indicators:
            name = indicator['name']
            params = indicator.get('params', {})
            
            try:
                calculator = self.indicator_calculators.get(name)

                if calculator:
                    calculator(result_df, params)
                else:
                    logger.warning(f"Indicator '{name}' not implemented")
            
            except Exception as e:
                logger.error(f"Error calculating indicator '{name}': {e}")
        
        return result_df

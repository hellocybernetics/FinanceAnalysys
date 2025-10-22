"""
Technical analysis module for calculating technical indicators using vectorbt.
"""

import pandas as pd
import numpy as np
import logging
import functools
import hashlib
from datetime import datetime, timedelta

try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    logging.warning("vectorbt not available, some indicators will not work")

logger = logging.getLogger(__name__)

# Simple in-memory cache for computed indicators
indicator_cache = {}

def cache_indicator(func):
    """Decorator to cache indicator calculation results."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a cache key based on function name and arguments
        # Convert DataFrame to a hashable form for caching
        cache_args = []
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                # Use a hash of the DataFrame's shape and a sample of its data
                cache_args.append(hashlib.md5(str((arg.shape, arg.iloc[:10].to_dict())).encode()).hexdigest())
            else:
                cache_args.append(str(arg))
        
        cache_kwargs = {k: str(v) for k, v in kwargs.items()}
        cache_key = hashlib.md5(str((func.__name__, tuple(cache_args), tuple(sorted(cache_kwargs.items())))).encode()).hexdigest()
        
        # Check if result is in cache and not expired
        if cache_key in indicator_cache:
            result, timestamp = indicator_cache[cache_key]
            # Cache for 10 minutes
            if datetime.now() - timestamp < timedelta(minutes=10):
                logger.info(f"Returning cached result for {func.__name__}")
                return result
        
        # Compute result and store in cache
        result = func(*args, **kwargs)
        indicator_cache[cache_key] = (result, datetime.now())
        return result
    return wrapper

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
    
    @cache_indicator
    def _calculate_sma(self, result_df, params):
        """Calculates Simple Moving Average (SMA)."""
        length = params.get('length', 20)
        if VECTORBT_AVAILABLE and vbt is not None:
            try:
                sma = vbt.MA.run(result_df['Close'], length, short_name=f'SMA_{length}')
                # Safely access the ma attribute
                try:
                    ma_value = sma.ma
                    if ma_value is not None:
                        result_df[f'SMA_{length}'] = ma_value
                    else:
                        result_df[f'SMA_{length}'] = result_df['Close'].rolling(window=length).mean()
                except (AttributeError, Exception):
                    result_df[f'SMA_{length}'] = result_df['Close'].rolling(window=length).mean()
            except Exception as e:
                logger.warning(f"Vectorbt SMA calculation failed: {e}. Using pandas fallback.")
                result_df[f'SMA_{length}'] = result_df['Close'].rolling(window=length).mean()
        else:
            result_df[f'SMA_{length}'] = result_df['Close'].rolling(window=length).mean()
        logger.info(f"Calculated SMA with length {length}")

    @cache_indicator
    def _calculate_ema(self, result_df, params):
        """Calculates Exponential Moving Average (EMA)."""
        length = params.get('length', 50)
        if VECTORBT_AVAILABLE and vbt is not None:
            try:
                ema = vbt.MA.run(result_df['Close'], length, short_name=f'EMA_{length}', ewm=True)
                # Safely access the ma attribute
                try:
                    ma_value = ema.ma
                    if ma_value is not None:
                        result_df[f'EMA_{length}'] = ma_value
                    else:
                        result_df[f'EMA_{length}'] = result_df['Close'].ewm(span=length, adjust=False).mean()
                except (AttributeError, Exception):
                    result_df[f'EMA_{length}'] = result_df['Close'].ewm(span=length, adjust=False).mean()
            except Exception as e:
                logger.warning(f"Vectorbt EMA calculation failed: {e}. Using pandas fallback.")
                result_df[f'EMA_{length}'] = result_df['Close'].ewm(span=length, adjust=False).mean()
        else:
            result_df[f'EMA_{length}'] = result_df['Close'].ewm(span=length, adjust=False).mean()
        logger.info(f"Calculated EMA with length {length}")

    @cache_indicator
    def _calculate_rsi(self, result_df, params):
        """Calculates Relative Strength Index (RSI)."""
        length = params.get('length', 14)
        if VECTORBT_AVAILABLE and vbt is not None:
            try:
                rsi = vbt.RSI.run(result_df['Close'], window=length, short_name=f'RSI_{length}')
                # Safely access the rsi attribute
                try:
                    rsi_value = rsi.rsi
                    if rsi_value is not None:
                        result_df[f'RSI_{length}'] = rsi_value
                    else:
                        self._calculate_rsi_pandas(result_df, length)
                except (AttributeError, Exception):
                    self._calculate_rsi_pandas(result_df, length)
            except Exception as e:
                logger.warning(f"Vectorbt RSI calculation failed: {e}. Using pandas fallback.")
                self._calculate_rsi_pandas(result_df, length)
        else:
            self._calculate_rsi_pandas(result_df, length)
        logger.info(f"Calculated RSI with length {length}")
    
    def _calculate_rsi_pandas(self, result_df, length):
        """Calculate RSI using pandas following Wilder's smoothing method."""
        delta = result_df['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Calculate initial averages
        avg_gain = pd.Series(np.nan, index=result_df.index)
        avg_loss = pd.Series(np.nan, index=result_df.index)

        avg_gain.iloc[length] = gain.iloc[1:length+1].mean()
        avg_loss.iloc[length] = loss.iloc[1:length+1].mean()

        # Subsequently, use Wilder's smoothing
        for i in range(length + 1, len(result_df)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (length - 1) + gain.iloc[i]) / length
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (length - 1) + loss.iloc[i]) / length

        rs = avg_gain / avg_loss
        result_df[f'RSI_{length}'] = 100 - (100 / (1 + rs))

    @cache_indicator
    def _calculate_macd(self, result_df, params):
        """Calculates Moving Average Convergence Divergence (MACD)."""
        fast = params.get('fast', 12)
        slow = params.get('slow', 26)
        signal = params.get('signal', 9)

        if VECTORBT_AVAILABLE and vbt is not None:
            try:
                fast_ema = vbt.MA.run(result_df['Close'], fast, short_name=f'EMA_{fast}', ewm=True)
                slow_ema = vbt.MA.run(result_df['Close'], slow, short_name=f'EMA_{slow}', ewm=True)
                
                # Safely access the ma attributes
                fast_ma = None
                slow_ma = None
                
                try:
                    fast_ma = fast_ema.ma
                except (AttributeError, Exception):
                    pass
                    
                try:
                    slow_ma = slow_ema.ma
                except (AttributeError, Exception):
                    pass
                
                if fast_ma is not None and slow_ma is not None:
                    macd_line = fast_ma - slow_ma
                    signal_line = vbt.MA.run(macd_line, signal, short_name=f'Signal_{signal}', ewm=True)
                    signal_ma = None
                    try:
                        signal_ma = signal_line.ma
                    except (AttributeError, Exception):
                        signal_ma = macd_line.ewm(span=signal, adjust=False).mean()
                    
                    histogram = macd_line - signal_ma

                    result_df[f'MACD_{fast}_{slow}'] = macd_line
                    result_df[f'MACD_Signal_{signal}'] = signal_ma
                    result_df[f'MACD_Hist_{fast}_{slow}_{signal}'] = histogram
                else:
                    # Fallback to pandas
                    fast_ema_data = result_df['Close'].ewm(span=fast, adjust=False).mean()
                    slow_ema_data = result_df['Close'].ewm(span=slow, adjust=False).mean()
                    macd_line = fast_ema_data - slow_ema_data
                    signal_line_data = macd_line.ewm(span=signal, adjust=False).mean()
                    histogram = macd_line - signal_line_data

                    result_df[f'MACD_{fast}_{slow}'] = macd_line
                    result_df[f'MACD_Signal_{signal}'] = signal_line_data
                    result_df[f'MACD_Hist_{fast}_{slow}_{signal}'] = histogram
            except Exception as e:
                logger.warning(f"Vectorbt MACD calculation failed: {e}. Using pandas fallback.")
                # Calculate MACD using pandas
                fast_ema = result_df['Close'].ewm(span=fast, adjust=False).mean()
                slow_ema = result_df['Close'].ewm(span=slow, adjust=False).mean()
                macd_line = fast_ema - slow_ema
                signal_line = macd_line.ewm(span=signal, adjust=False).mean()
                histogram = macd_line - signal_line

                result_df[f'MACD_{fast}_{slow}'] = macd_line
                result_df[f'MACD_Signal_{signal}'] = signal_line
                result_df[f'MACD_Hist_{fast}_{slow}_{signal}'] = histogram
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

    @cache_indicator
    def _calculate_bbands(self, result_df, params):
        """Calculates Bollinger Bands."""
        length = params.get('length', 20)
        std = params.get('std', 2)

        if VECTORBT_AVAILABLE and vbt is not None:
            try:
                middle_ma = vbt.MA.run(result_df['Close'], length, short_name=f'SMA_{length}')
                # Safely access the ma attribute
                try:
                    middle = middle_ma.ma
                    if middle is None:
                        middle = result_df['Close'].rolling(window=length).mean()
                except (AttributeError, Exception):
                    middle = result_df['Close'].rolling(window=length).mean()
            except Exception as e:
                logger.warning(f"Vectorbt BBands calculation failed: {e}. Using pandas fallback.")
                middle = result_df['Close'].rolling(window=length).mean()
        else:
            middle = result_df['Close'].rolling(window=length).mean()
            
        rolling_std = result_df['Close'].rolling(window=length).std()

        upper = middle + (rolling_std * std)
        lower = middle - (rolling_std * std)

        result_df[f'BBL_{length}_{std}'] = lower
        result_df[f'BBM_{length}_{std}'] = middle
        result_df[f'BBU_{length}_{std}'] = upper
        logger.info(f"Calculated Bollinger Bands with length={length}, std={std}")

    @cache_indicator
    def _calculate_atr(self, result_df, params):
        """Calculates Average True Range (ATR)."""
        length = params.get('length', 14)
        if VECTORBT_AVAILABLE and vbt is not None:
            try:
                atr = vbt.ATR.run(
                    high=result_df['High'],
                    low=result_df['Low'],
                    close=result_df['Close'],
                    window=length,
                    short_name=f'ATR_{length}'
                )
                # Safely access the atr attribute
                try:
                    atr_value = atr.atr
                    if atr_value is not None:
                        result_df[f'ATR_{length}'] = atr_value
                    else:
                        self._calculate_atr_pandas(result_df, length)
                except (AttributeError, Exception):
                    self._calculate_atr_pandas(result_df, length)
            except Exception as e:
                logger.warning(f"Vectorbt ATR calculation failed: {e}. Using pandas fallback.")
                self._calculate_atr_pandas(result_df, length)
        else:
            self._calculate_atr_pandas(result_df, length)
        logger.info(f"Calculated ATR with length {length}")
    
    def _calculate_atr_pandas(self, result_df, length):
        """Calculate ATR using pandas."""
        high = result_df['High']
        low = result_df['Low']
        close = result_df['Close']
        tr1 = abs(high - low)
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        result_df[f'ATR_{length}'] = tr.rolling(window=length).mean()

    @cache_indicator
    def _calculate_stochastic(self, result_df, params):
        """Calculates Stochastic Oscillator."""
        k = params.get('k', 14)
        d = params.get('d', 3)

        if VECTORBT_AVAILABLE and vbt is not None:
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
                # Safely access the attributes
                slowk = None
                slowd = None
                try:
                    slowk = stoch.slowk
                except (AttributeError, Exception):
                    pass
                try:
                    slowd = stoch.slowd
                except (AttributeError, Exception):
                    pass
                
                if slowk is not None and slowd is not None:
                    result_df[f'STOCHk_{k}_{d}'] = slowk
                    result_df[f'STOCHd_{k}_{d}'] = slowd
                else:
                    self._calculate_stochastic_pandas(result_df, k, d)
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

    @cache_indicator
    def _calculate_adx(self, result_df, params):
        """Calculates Average Directional Index (ADX)."""
        length = params.get('length', 14)
        if VECTORBT_AVAILABLE and vbt is not None:
            try:
                adx_factory = vbt.IndicatorFactory.from_talib('ADX')
                adx_output = adx_factory.run(
                    high=result_df['High'],
                    low=result_df['Low'],
                    close=result_df['Close'],
                    timeperiod=length
                )
                # Safely access the real attribute
                try:
                    real_value = adx_output.real
                    if real_value is not None:
                        result_df[f'ADX_{length}'] = real_value
                    else:
                        logger.warning(f"ADX calculation failed. Using placeholder values.")
                        result_df[f'ADX_{length}'] = np.nan
                except (AttributeError, Exception):
                    logger.warning(f"ADX calculation failed. Using placeholder values.")
                    result_df[f'ADX_{length}'] = np.nan
                logger.info(f"Calculated ADX with length {length} using TA-Lib")
            except Exception as e:
                logger.error(f"Error calculating ADX: {e}. Make sure TA-Lib is installed and data has sufficient length.")
                result_df[f'ADX_{length}'] = np.nan
        else:
            logger.warning(f"ADX calculation requires TA-Lib. Using placeholder values.")
            result_df[f'ADX_{length}'] = np.nan

    @cache_indicator
    def _calculate_willr(self, result_df, params):
        """Calculates Williams %R."""
        length = params.get('length', 14)
        if VECTORBT_AVAILABLE and vbt is not None:
            try:
                willr_factory = vbt.IndicatorFactory.from_talib('WILLR')
                willr_output = willr_factory.run(
                    high=result_df['High'],
                    low=result_df['Low'],
                    close=result_df['Close'],
                    timeperiod=length
                )
                # Safely access the real attribute
                try:
                    real_value = willr_output.real
                    if real_value is not None:
                        result_df[f'WILLR_{length}'] = real_value
                    else:
                        self._calculate_willr_pandas(result_df, length)
                except (AttributeError, Exception):
                    self._calculate_willr_pandas(result_df, length)
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
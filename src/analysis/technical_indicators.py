"""
Technical analysis module for calculating technical indicators using vectorbt.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
import logging

logger = logging.getLogger(__name__)

class TechnicalAnalysis:
    """
    Class for performing technical analysis on financial data.
    """
    
    def __init__(self):
        """
        Initialize the TechnicalAnalysis class.
        """
        pass
    
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
                if name == 'SMA':
                    length = params.get('length', 20)
                    sma = vbt.MA.run(result_df['Close'], length, short_name=f'SMA_{length}')
                    result_df[f'SMA_{length}'] = sma.ma
                    logger.info(f"Calculated SMA with length {length}")
                
                elif name == 'EMA':
                    length = params.get('length', 50)
                    ema = vbt.MA.run(result_df['Close'], length, short_name=f'EMA_{length}', ewm=True)
                    result_df[f'EMA_{length}'] = ema.ma
                    logger.info(f"Calculated EMA with length {length}")
                
                elif name == 'RSI':
                    length = params.get('length', 14)
                    rsi = vbt.RSI.run(result_df['Close'], window=length, short_name=f'RSI_{length}')
                    result_df[f'RSI_{length}'] = rsi.rsi
                    logger.info(f"Calculated RSI with length {length}")
                
                elif name == 'MACD':
                    fast = params.get('fast', 12)
                    slow = params.get('slow', 26)
                    signal = params.get('signal', 9)
                    
                    fast_ema = vbt.MA.run(result_df['Close'], fast, short_name=f'EMA_{fast}', ewm=True)
                    slow_ema = vbt.MA.run(result_df['Close'], slow, short_name=f'EMA_{slow}', ewm=True)
                    
                    # Calculate MACD line (fast EMA - slow EMA)
                    macd_line = fast_ema.ma - slow_ema.ma
                    
                    # Calculate signal line (EMA of MACD line)
                    signal_line = vbt.MA.run(macd_line, signal, short_name=f'Signal_{signal}', ewm=True).ma
                    
                    # Calculate histogram (MACD line - signal line)
                    histogram = macd_line - signal_line
                    
                    result_df[f'MACD_{fast}_{slow}'] = macd_line
                    result_df[f'MACD_Signal_{signal}'] = signal_line
                    result_df[f'MACD_Hist_{fast}_{slow}_{signal}'] = histogram
                    logger.info(f"Calculated MACD with fast={fast}, slow={slow}, signal={signal}")
                
                elif name == 'BBands':
                    length = params.get('length', 20)
                    std = params.get('std', 2)
                    
                    # Calculate Bollinger Bands manually
                    middle = vbt.MA.run(result_df['Close'], length, short_name=f'SMA_{length}').ma
                    
                    # Calculate standard deviation
                    rolling_std = result_df['Close'].rolling(window=length).std()
                    
                    upper = middle + (rolling_std * std)
                    lower = middle - (rolling_std * std)
                    
                    result_df[f'BBL_{length}_{std}'] = lower
                    result_df[f'BBM_{length}_{std}'] = middle
                    result_df[f'BBU_{length}_{std}'] = upper
                    logger.info(f"Calculated Bollinger Bands with length={length}, std={std}")
                
                elif name == 'ATR':
                    length = params.get('length', 14)
                    atr = vbt.ATR.run(
                        high=result_df['High'], 
                        low=result_df['Low'], 
                        close=result_df['Close'], 
                        window=length,
                        short_name=f'ATR_{length}'
                    )
                    result_df[f'ATR_{length}'] = atr.atr
                    logger.info(f"Calculated ATR with length {length}")
                
                elif name == 'ADX':
                    length = params.get('length', 14)
                    logger.warning(f"ADX indicator not implemented in vectorbt")
                
                elif name == 'Stochastic':
                    k = params.get('k', 14)
                    d = params.get('d', 3)
                    
                    stoch = vbt.Stochastic.run(
                        high=result_df['High'], 
                        low=result_df['Low'], 
                        close=result_df['Close'], 
                        k_window=k, 
                        d_window=d,
                        short_name=f'STOCH_{k}_{d}'
                    )
                    
                    result_df[f'STOCHk_{k}_{d}'] = stoch.percent_k
                    result_df[f'STOCHd_{k}_{d}'] = stoch.percent_d
                    logger.info(f"Calculated Stochastic with k={k}, d={d}")
                
                else:
                    logger.warning(f"Indicator '{name}' not implemented")
            
            except Exception as e:
                logger.error(f"Error calculating indicator '{name}': {e}")
        
        return result_df

"""
Technical analysis module for calculating technical indicators using talib.
"""

import pandas as pd
import numpy as np
import talib
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
        Calculate technical indicators for the given dataframe using talib.
        
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
                    result_df[f'SMA_{length}'] = talib.SMA(result_df['Close'], timeperiod=length)
                    logger.info(f"Calculated SMA with length {length}")
                
                elif name == 'EMA':
                    length = params.get('length', 50)
                    result_df[f'EMA_{length}'] = talib.EMA(result_df['Close'], timeperiod=length)
                    logger.info(f"Calculated EMA with length {length}")
                
                elif name == 'RSI':
                    length = params.get('length', 14)
                    result_df[f'RSI_{length}'] = talib.RSI(result_df['Close'], timeperiod=length)
                    logger.info(f"Calculated RSI with length {length}")
                
                elif name == 'MACD':
                    fast = params.get('fast', 12)
                    slow = params.get('slow', 26)
                    signal = params.get('signal', 9)
                    
                    macd, signal_line, histogram = talib.MACD(
                        result_df['Close'], 
                        fastperiod=fast, 
                        slowperiod=slow, 
                        signalperiod=signal
                    )
                    
                    result_df[f'MACD_{fast}_{slow}'] = macd
                    result_df[f'MACD_Signal_{signal}'] = signal_line
                    result_df[f'MACD_Hist_{fast}_{slow}_{signal}'] = histogram
                    logger.info(f"Calculated MACD with fast={fast}, slow={slow}, signal={signal}")
                
                elif name == 'BBands':
                    length = params.get('length', 20)
                    std = params.get('std', 2)
                    
                    upper, middle, lower = talib.BBANDS(
                        result_df['Close'], 
                        timeperiod=length, 
                        nbdevup=std, 
                        nbdevdn=std,
                        matype=0  # Simple Moving Average
                    )
                    
                    result_df[f'BBU_{length}_{std}'] = upper
                    result_df[f'BBM_{length}_{std}'] = middle
                    result_df[f'BBL_{length}_{std}'] = lower
                    logger.info(f"Calculated Bollinger Bands with length={length}, std={std}")
                
                elif name == 'ATR':
                    length = params.get('length', 14)
                    result_df[f'ATR_{length}'] = talib.ATR(
                        result_df['High'], 
                        result_df['Low'], 
                        result_df['Close'], 
                        timeperiod=length
                    )
                    logger.info(f"Calculated ATR with length {length}")
                
                elif name == 'ADX':
                    length = params.get('length', 14)
                    result_df[f'ADX_{length}'] = talib.ADX(
                        result_df['High'], 
                        result_df['Low'], 
                        result_df['Close'], 
                        timeperiod=length
                    )
                    logger.info(f"Calculated ADX with length {length}")
                
                elif name == 'Stochastic':
                    k = params.get('k', 14)
                    d = params.get('d', 3)
                    
                    slowk, slowd = talib.STOCH(
                        result_df['High'], 
                        result_df['Low'], 
                        result_df['Close'], 
                        fastk_period=k, 
                        slowk_period=3, 
                        slowk_matype=0, 
                        slowd_period=d, 
                        slowd_matype=0
                    )
                    
                    result_df[f'STOCHk_{k}_{d}'] = slowk
                    result_df[f'STOCHd_{k}_{d}'] = slowd
                    logger.info(f"Calculated Stochastic with k={k}, d={d}")
                
                else:
                    logger.warning(f"Indicator '{name}' not implemented")
            
            except Exception as e:
                logger.error(f"Error calculating indicator '{name}': {e}")
        
        return result_df

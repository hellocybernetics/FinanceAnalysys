"""
Technical analysis module for calculating technical indicators using pandas_ta.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
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
        Calculate technical indicators for the given dataframe using pandas_ta.
        
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
                    result_df[f'SMA_{length}'] = ta.sma(result_df['Close'], length=length)
                    logger.info(f"Calculated SMA with length {length}")
                
                elif name == 'EMA':
                    length = params.get('length', 50)
                    result_df[f'EMA_{length}'] = ta.ema(result_df['Close'], length=length)
                    logger.info(f"Calculated EMA with length {length}")
                
                elif name == 'RSI':
                    length = params.get('length', 14)
                    result_df[f'RSI_{length}'] = ta.rsi(result_df['Close'], length=length)
                    logger.info(f"Calculated RSI with length {length}")
                
                elif name == 'MACD':
                    fast = params.get('fast', 12)
                    slow = params.get('slow', 26)
                    signal = params.get('signal', 9)
                    
                    macd_result = ta.macd(
                        result_df['Close'], 
                        fast=fast, 
                        slow=slow, 
                        signal=signal
                    )
                    
                    result_df[f'MACD_{fast}_{slow}'] = macd_result['MACD']
                    result_df[f'MACD_Signal_{signal}'] = macd_result['MACDs']
                    result_df[f'MACD_Hist_{fast}_{slow}_{signal}'] = macd_result['MACDh']
                    logger.info(f"Calculated MACD with fast={fast}, slow={slow}, signal={signal}")
                
                elif name == 'BBands':
                    length = params.get('length', 20)
                    std = params.get('std', 2)
                    
                    bbands_result = ta.bbands(
                        result_df['Close'], 
                        length=length, 
                        std=std
                    )
                    
                    result_df[f'BBL_{length}_{std}'] = bbands_result['BBL']
                    result_df[f'BBM_{length}_{std}'] = bbands_result['BBM']
                    result_df[f'BBU_{length}_{std}'] = bbands_result['BBU']
                    logger.info(f"Calculated Bollinger Bands with length={length}, std={std}")
                
                elif name == 'ATR':
                    length = params.get('length', 14)
                    result_df[f'ATR_{length}'] = ta.atr(
                        high=result_df['High'], 
                        low=result_df['Low'], 
                        close=result_df['Close'], 
                        length=length
                    )
                    logger.info(f"Calculated ATR with length {length}")
                
                elif name == 'ADX':
                    length = params.get('length', 14)
                    adx_result = ta.adx(
                        high=result_df['High'], 
                        low=result_df['Low'], 
                        close=result_df['Close'], 
                        length=length
                    )
                    result_df[f'ADX_{length}'] = adx_result[f'ADX_{length}']
                    logger.info(f"Calculated ADX with length {length}")
                
                elif name == 'Stochastic':
                    k = params.get('k', 14)
                    d = params.get('d', 3)
                    
                    stoch_result = ta.stoch(
                        high=result_df['High'], 
                        low=result_df['Low'], 
                        close=result_df['Close'], 
                        k=k, 
                        d=d
                    )
                    
                    result_df[f'STOCHk_{k}_{d}'] = stoch_result[f'STOCHk_{k}']
                    result_df[f'STOCHd_{k}_{d}'] = stoch_result[f'STOCHd_{d}']
                    logger.info(f"Calculated Stochastic with k={k}, d={d}")
                
                else:
                    logger.warning(f"Indicator '{name}' not implemented")
            
            except Exception as e:
                logger.error(f"Error calculating indicator '{name}': {e}")
        
        return result_df

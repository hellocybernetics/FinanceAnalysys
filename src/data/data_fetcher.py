"""
Data fetcher module for retrieving financial data using vectorbt or yfinance.
"""

import os
import pandas as pd
import yfinance as yf
import vectorbt as vbt
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataFetcher:
    """
    Class for fetching financial data from various sources.
    """
    
    def __init__(self, use_vectorbt=True):
        """
        Initialize the DataFetcher.
        
        Args:
            use_vectorbt (bool): Whether to use vectorbt for data fetching (True) or yfinance (False).
        """
        self.use_vectorbt = use_vectorbt
    
    def fetch_data(self, symbols, period='1y', interval='1d'):
        """
        Fetch data for the given symbols.
        
        Args:
            symbols (list): List of symbols to fetch data for.
            period (str): Time period to fetch data for.
            interval (str): Data interval.
            
        Returns:
            dict: Dictionary mapping symbols to their respective dataframes.
        """
        if not symbols:
            raise ValueError("No symbols provided")
        
        data = {}
        
        if self.use_vectorbt:
            try:
                logger.info(f"Fetching data for {symbols} using vectorbt")
                vbt_data = vbt.YFData.download(symbols, period=period, interval=interval)
                
                for symbol in symbols:
                    if len(symbols) > 1:
                        symbol_data = vbt_data.select(symbol)
                        data[symbol] = symbol_data.to_pandas()
                    else:
                        data[symbol] = vbt_data.to_pandas()
                    
                    logger.info(f"Successfully fetched data for {symbol} using vectorbt")
            except Exception as e:
                logger.error(f"Error fetching data with vectorbt: {e}")
                logger.info("Falling back to yfinance...")
                self.use_vectorbt = False
        
        if not self.use_vectorbt:
            for symbol in symbols:
                try:
                    logger.info(f"Fetching data for {symbol} using yfinance")
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(period=period, interval=interval)
                    
                    if not df.empty:
                        if not isinstance(df.index, pd.DatetimeIndex):
                            df.index = pd.to_datetime(df.index)
                        
                        if df.index.tz is not None:
                            df.index = df.index.tz_convert('UTC').tz_localize(None)
                        
                        data[symbol] = df
                        logger.info(f"Successfully fetched data for {symbol} using yfinance")
                    else:
                        logger.warning(f"No data found for {symbol}")
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
        
        return data
    
    def save_data(self, data, output_dir):
        """
        Save the fetched data to CSV files.
        
        Args:
            data (dict): Dictionary mapping symbols to their respective dataframes.
            output_dir (str): Directory to save the data to.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for symbol, df in data.items():
            clean_symbol = symbol.replace('/', '-').replace('^', '')
            filename = f"{clean_symbol}_{timestamp}.csv"
            filepath = os.path.join(output_dir, filename)
            
            df.to_csv(filepath)
            logger.info(f"Saved data for {symbol} to {filepath}")

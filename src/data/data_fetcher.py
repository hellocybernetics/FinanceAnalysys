"""
Data fetcher module for retrieving financial data using vectorbt or yfinance.
"""

import os
import pandas as pd
import yfinance as yf
import vectorbt as vbt
from datetime import datetime, timedelta
import glob
import re
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
        self.failed_symbols = []
        self.company_names = {}
    
    def get_company_name(self, symbol):
        """
        Get the company name for a symbol.
        
        Args:
            symbol (str): The symbol to get the company name for.
            
        Returns:
            str: The company name, or the symbol if the company name couldn't be retrieved.
        """
        if symbol in self.company_names:
            return self.company_names[symbol]
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if 'shortName' in info:
                name = info['shortName']
            elif 'longName' in info:
                name = info['longName']
            else:
                name = symbol
                
            self.company_names[symbol] = name
            return name
        except Exception as e:
            logger.warning(f"Could not retrieve company name for {symbol}: {e}")
            self.company_names[symbol] = symbol
            return symbol
    
    def find_latest_data_file(self, symbol, output_dir):
        """
        Find the latest data file for a symbol.
        
        Args:
            symbol (str): The symbol to find data for.
            output_dir (str): The directory to search in.
            
        Returns:
            tuple: (filepath, timestamp) if found, (None, None) otherwise.
        """
        clean_symbol = symbol.replace('/', '-').replace('^', '')
        pattern = os.path.join(output_dir, f"{clean_symbol}_*.csv")
        files = glob.glob(pattern)
        
        if not files:
            return None, None
        
        latest_file = None
        latest_timestamp = None
        
        for file in files:
            filename = os.path.basename(file)
            match = re.search(r'_(\d{8}_\d{6})\.csv$', filename)
            
            if match:
                timestamp_str = match.group(1)
                try:
                    file_timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    
                    if latest_timestamp is None or file_timestamp > latest_timestamp:
                        latest_timestamp = file_timestamp
                        latest_file = file
                except ValueError:
                    continue
        
        return latest_file, latest_timestamp
    
    def is_data_recent(self, timestamp, max_age_minutes=30):
        """
        Check if data is recent enough to use.
        
        Args:
            timestamp (datetime): The timestamp to check.
            max_age_minutes (int): Maximum age in minutes.
            
        Returns:
            bool: True if the data is recent enough, False otherwise.
        """
        if timestamp is None:
            return False
        
        now = datetime.now()
        max_age = timedelta(minutes=max_age_minutes)
        
        return (now - timestamp) <= max_age
    
    def clean_old_data(self, symbol, output_dir):
        """
        Delete old data files for a symbol.
        
        Args:
            symbol (str): The symbol to clean data for.
            output_dir (str): The directory to clean in.
        """
        clean_symbol = symbol.replace('/', '-').replace('^', '')
        pattern = os.path.join(output_dir, f"{clean_symbol}_*.csv")
        files = glob.glob(pattern)
        
        for file in files:
            try:
                os.remove(file)
                logger.info(f"Deleted old data file: {file}")
            except Exception as e:
                logger.warning(f"Could not delete file {file}: {e}")
    
    def load_cached_data(self, filepath):
        """
        Load data from a cached CSV file.
        
        Args:
            filepath (str): Path to the CSV file.
            
        Returns:
            pd.DataFrame: The loaded dataframe, or None if loading failed.
        """
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            return df
        except Exception as e:
            logger.error(f"Error loading cached data from {filepath}: {e}")
            return None
    
    def fetch_data(self, symbols, period='1y', interval='1d', output_dir=None, use_cache=True, cache_max_age=30):
        """
        Fetch data for the given symbols, using cache if available and recent.
        
        Args:
            symbols (list): List of symbols to fetch data for.
            period (str): Time period to fetch data for.
            interval (str): Data interval.
            output_dir (str): Directory to look for cached data.
            use_cache (bool): Whether to use cached data if available.
            cache_max_age (int): Maximum age of cached data in minutes.
            
        Returns:
            dict: Dictionary mapping symbols to their respective dataframes.
        """
        if not symbols:
            raise ValueError("No symbols provided")
        
        data = {}
        self.failed_symbols = []
        
        if use_cache and output_dir:
            for symbol in symbols:
                latest_file, timestamp = self.find_latest_data_file(symbol, output_dir)
                
                if latest_file and self.is_data_recent(timestamp, cache_max_age):
                    logger.info(f"Using cached data for {symbol} from {latest_file}")
                    df = self.load_cached_data(latest_file)
                    
                    if df is not None and not df.empty:
                        data[symbol] = df
                        self.get_company_name(symbol)
                        continue
                elif latest_file:
                    self.clean_old_data(symbol, output_dir)
        
        # Fetch data for symbols not in cache
        symbols_to_fetch = [s for s in symbols if s not in data]
        
        if not symbols_to_fetch:
            logger.info("All data loaded from cache")
            return data
        
        if self.use_vectorbt:
            try:
                logger.info(f"Fetching data for {symbols_to_fetch} using vectorbt")
                vbt_data = vbt.YFData.download(symbols_to_fetch, period=period, interval=interval)
                
                for symbol in symbols_to_fetch:
                    try:
                        if len(symbols_to_fetch) > 1:
                            symbol_data = vbt_data.select(symbol)
                            df = symbol_data.to_pandas()
                        else:
                            df = vbt_data.to_pandas()
                        
                        if df is not None and not df.empty:
                            data[symbol] = df
                            logger.info(f"Successfully fetched data for {symbol} using vectorbt")
                            self.get_company_name(symbol)
                        else:
                            logger.warning(f"No data found for {symbol} using vectorbt")
                            self.failed_symbols.append(symbol)
                    except Exception as e:
                        logger.error(f"Error processing {symbol} data from vectorbt: {e}")
                        self.failed_symbols.append(symbol)
            except Exception as e:
                logger.error(f"Error fetching data with vectorbt: {e}")
                logger.info("Falling back to yfinance...")
                self.use_vectorbt = False
        
        if not self.use_vectorbt:
            for symbol in symbols_to_fetch:
                if symbol in data:  # Skip if already fetched by vectorbt
                    continue
                    
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
                        
                        self.get_company_name(symbol)
                    else:
                        logger.warning(f"No data found for {symbol}")
                        self.failed_symbols.append(symbol)
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
                    self.failed_symbols.append(symbol)
        
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
    
    def get_failed_symbols(self):
        """
        Get the list of symbols that failed to fetch.
        
        Returns:
            list: List of symbols that failed to fetch.
        """
        return self.failed_symbols

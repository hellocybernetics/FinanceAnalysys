"""
Data fetcher module for retrieving financial data using vectorbt or yfinance.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import glob
import re
import logging
import functools
import hashlib

try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    logging.warning("vectorbt not available, using yfinance for data fetching")

logger = logging.getLogger(__name__)

# Simple in-memory cache for computed results
cache = {}

def cache_result(func):
    """Decorator to cache function results based on arguments."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a cache key based on function name and arguments
        cache_key = hashlib.md5(str((func.__name__, args, sorted(kwargs.items()))).encode()).hexdigest()
        
        # Check if result is in cache and not expired
        if cache_key in cache:
            result, timestamp = cache[cache_key]
            # Cache for 5 minutes
            if datetime.now() - timestamp < timedelta(minutes=5):
                logger.info(f"Returning cached result for {func.__name__}")
                return result
        
        # Compute result and store in cache
        result = func(*args, **kwargs)
        cache[cache_key] = (result, datetime.now())
        return result
    return wrapper

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
        self.use_vectorbt = use_vectorbt and VECTORBT_AVAILABLE
        if use_vectorbt and not VECTORBT_AVAILABLE:
            logger.warning("vectorbt requested but not available, falling back to yfinance")
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
    
    @cache_result
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
        
        if self.use_vectorbt and VECTORBT_AVAILABLE and vbt is not None:
            try:
                logger.info(f"Fetching data for {symbols_to_fetch} using vectorbt")
                vbt_data = vbt.YFData.download(symbols_to_fetch, period=period, interval=interval)
                full_df = vbt_data.get()
                
                # Check if we have valid data
                if full_df is not None:
                    # Convert to DataFrame if needed
                    if not isinstance(full_df, pd.DataFrame):
                        # Try to convert to DataFrame
                        try:
                            full_df = pd.DataFrame(full_df)
                        except:
                            # If conversion fails, log and continue with yfinance
                            logger.warning(f"Could not convert vectorbt result to DataFrame for symbols: {symbols_to_fetch}")
                            full_df = None
                    
                    if full_df is not None and isinstance(full_df, pd.DataFrame) and not full_df.empty:
                        for symbol in symbols_to_fetch:
                            try:
                                symbol_df = None

                                if isinstance(full_df.columns, pd.MultiIndex):
                                    # Identify which level holds the symbol labels
                                    symbol_level = None
                                    for level_idx in range(full_df.columns.nlevels):
                                        level_values = full_df.columns.get_level_values(level_idx)
                                        if symbol in level_values:
                                            symbol_level = level_idx
                                            break

                                    if symbol_level is not None:
                                        extracted = full_df.xs(symbol, axis=1, level=symbol_level)
                                        # xs can return a Series when only one column remains
                                        if isinstance(extracted, pd.Series):
                                            symbol_df = extracted.to_frame(name=extracted.name if extracted.name else 'Close')
                                        else:
                                            symbol_df = extracted
                                    else:
                                        logger.warning(
                                            "Symbol %s not found in any MultiIndex level of vectorbt result", symbol
                                        )
                                elif symbol in full_df.columns:
                                    # Single-level columns with symbols as labels
                                    symbol_df = full_df[symbol]
                                elif len(symbols_to_fetch) == 1 and symbol == symbols_to_fetch[0]:
                                    # vectorbt returns a single DataFrame when only one symbol requested
                                    symbol_df = full_df
                                else:
                                    logger.warning(f"Unexpected DataFrame structure when processing {symbol}")
                                    self.failed_symbols.append(symbol)
                                    continue

                                if symbol_df is not None:
                                    if isinstance(symbol_df, pd.Series):
                                        symbol_df = symbol_df.to_frame(name=symbol_df.name or 'Close')

                                    # Flatten any remaining MultiIndex columns into simple strings
                                    if isinstance(symbol_df.columns, pd.MultiIndex):
                                        symbol_df.columns = ["_".join(map(str, col)).strip() for col in symbol_df.columns]
                                    else:
                                        symbol_df.columns = [str(col) for col in symbol_df.columns]

                                    if not symbol_df.empty:
                                        data[symbol] = symbol_df
                                        logger.info(f"Successfully processed data for {symbol} using vectorbt")
                                        self.get_company_name(symbol)
                                    else:
                                        logger.warning(f"No data extracted for {symbol} from vectorbt result")
                                        self.failed_symbols.append(symbol)
                                else:
                                    self.failed_symbols.append(symbol)
                            except KeyError:
                                logger.error(f"Symbol {symbol} not found in vectorbt results.")
                                self.failed_symbols.append(symbol)
                            except Exception as e:
                                logger.error(f"Error processing {symbol} data from vectorbt result: {e}")
                                self.failed_symbols.append(symbol)
                    else:
                        logger.warning(f"No valid data returned by vectorbt.YFData.get() for symbols: {symbols_to_fetch}")
                        self.failed_symbols.extend(symbols_to_fetch)

            except Exception as e:
                logger.error(f"Error fetching or processing data with vectorbt: {e}")
                logger.info("Falling back to yfinance...")
                self.use_vectorbt = False
                symbols_to_fetch_yf = [s for s in symbols_to_fetch if s not in data and s not in self.failed_symbols]
                symbols_to_fetch = symbols_to_fetch_yf
        
        if not self.use_vectorbt or (self.use_vectorbt and symbols_to_fetch):
            for symbol in symbols_to_fetch:
                if symbol in data:
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

"""
Fundamentals data fetcher for retrieving financial statements and company information.
"""

import pandas as pd
import yfinance as yf
from typing import Dict, Optional
import logging
from src.core.exceptions import DataFetchError
from src.core.models import CompanyInfo

logger = logging.getLogger(__name__)


class FundamentalsFetcher:
    """
    Class for fetching fundamental financial data from yfinance.
    """

    def __init__(self):
        """Initialize the FundamentalsFetcher."""
        self._info_cache: Dict[str, Dict] = {}

    def fetch_company_info(self, symbol: str) -> CompanyInfo:
        """
        Fetch company information for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            CompanyInfo object with company details

        Raises:
            DataFetchError: If company information cannot be fetched
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Cache the raw info for later use
            self._info_cache[symbol] = info

            return CompanyInfo(
                symbol=symbol,
                name=info.get('longName') or info.get('shortName') or symbol,
                sector=info.get('sector'),
                industry=info.get('industry'),
                market_cap=info.get('marketCap'),
                employees=info.get('fullTimeEmployees'),
                description=info.get('longBusinessSummary'),
                website=info.get('website'),
                country=info.get('country')
            )
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {e}")
            raise DataFetchError(f"Failed to fetch company info for {symbol}: {e}")

    def fetch_financials(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch income statement (financials) for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            DataFrame with financial data, or None if not available
        """
        try:
            ticker = yf.Ticker(symbol)
            financials = ticker.financials

            if financials is not None and not financials.empty:
                logger.info(f"Successfully fetched financials for {symbol}")
                return financials
            else:
                logger.warning(f"No financial data available for {symbol}")
                return None
        except Exception as e:
            logger.error(f"Error fetching financials for {symbol}: {e}")
            return None

    def fetch_balance_sheet(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch balance sheet for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            DataFrame with balance sheet data, or None if not available
        """
        try:
            ticker = yf.Ticker(symbol)
            balance_sheet = ticker.balance_sheet

            if balance_sheet is not None and not balance_sheet.empty:
                logger.info(f"Successfully fetched balance sheet for {symbol}")
                return balance_sheet
            else:
                logger.warning(f"No balance sheet data available for {symbol}")
                return None
        except Exception as e:
            logger.error(f"Error fetching balance sheet for {symbol}: {e}")
            return None

    def fetch_cash_flow(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch cash flow statement for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            DataFrame with cash flow data, or None if not available
        """
        try:
            ticker = yf.Ticker(symbol)
            cash_flow = ticker.cashflow

            if cash_flow is not None and not cash_flow.empty:
                logger.info(f"Successfully fetched cash flow for {symbol}")
                return cash_flow
            else:
                logger.warning(f"No cash flow data available for {symbol}")
                return None
        except Exception as e:
            logger.error(f"Error fetching cash flow for {symbol}: {e}")
            return None

    def fetch_key_stats(self, symbol: str) -> Dict:
        """
        Fetch key statistics for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with key statistics
        """
        try:
            # Use cached info if available
            if symbol in self._info_cache:
                info = self._info_cache[symbol]
            else:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                self._info_cache[symbol] = info

            # Extract key statistics
            stats = {
                'currentPrice': info.get('currentPrice'),
                'previousClose': info.get('previousClose'),
                'open': info.get('open'),
                'dayHigh': info.get('dayHigh'),
                'dayLow': info.get('dayLow'),
                'volume': info.get('volume'),
                'averageVolume': info.get('averageVolume'),
                'marketCap': info.get('marketCap'),
                'beta': info.get('beta'),
                'trailingPE': info.get('trailingPE'),
                'forwardPE': info.get('forwardPE'),
                'dividendYield': info.get('dividendYield'),
                'payoutRatio': info.get('payoutRatio'),
                'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh'),
                'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow'),
                'priceToBook': info.get('priceToBook'),
                'debtToEquity': info.get('debtToEquity'),
                'returnOnEquity': info.get('returnOnEquity'),
                'returnOnAssets': info.get('returnOnAssets'),
                'revenueGrowth': info.get('revenueGrowth'),
                'earningsGrowth': info.get('earningsGrowth'),
                'profitMargins': info.get('profitMargins'),
                'operatingMargins': info.get('operatingMargins'),
                'grossMargins': info.get('grossMargins')
            }

            logger.info(f"Successfully fetched key stats for {symbol}")
            return stats
        except Exception as e:
            logger.error(f"Error fetching key stats for {symbol}: {e}")
            return {}

    def get_cached_info(self, symbol: str) -> Optional[Dict]:
        """
        Get cached ticker info if available.

        Args:
            symbol: Stock symbol

        Returns:
            Cached info dictionary, or None if not cached
        """
        return self._info_cache.get(symbol)

"""
Fundamental analysis service for unified fundamental analysis operations.
"""

from typing import List
import logging
from src.core.models import (
    FundamentalAnalysisRequest,
    FundamentalAnalysisResult,
    CompanyInfo,
    FinancialRatios
)
from src.core.exceptions import DataFetchError
from src.data.fundamentals import FundamentalsFetcher
from src.analysis.fundamental_metrics import FundamentalMetrics

logger = logging.getLogger(__name__)


class FundamentalAnalysisService:
    """
    Unified service for fundamental analysis.
    Provides a single interface for WebGUI, MCP, and CLI clients.
    """

    def __init__(self):
        """Initialize the fundamental analysis service."""
        self.fundamentals_fetcher = FundamentalsFetcher()
        self.metrics_calculator = FundamentalMetrics()

    def analyze(self, request: FundamentalAnalysisRequest) -> List[FundamentalAnalysisResult]:
        """
        Perform fundamental analysis on multiple symbols.

        Args:
            request: FundamentalAnalysisRequest with analysis parameters

        Returns:
            List of FundamentalAnalysisResult for each symbol

        Raises:
            DataFetchError: If critical data fetching fails
        """
        logger.info(f"Starting fundamental analysis for {len(request.symbols)} symbols")
        results = []

        for symbol in request.symbols:
            try:
                result = self._analyze_symbol(symbol, request)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")
                # Continue with other symbols instead of failing completely
                continue

        logger.info(f"Fundamental analysis completed for {len(results)} symbols")
        return results

    def _analyze_symbol(
        self,
        symbol: str,
        request: FundamentalAnalysisRequest
    ) -> FundamentalAnalysisResult:
        """
        Analyze a single symbol.

        Args:
            symbol: Stock symbol
            request: Analysis request parameters

        Returns:
            FundamentalAnalysisResult for the symbol
        """
        # Fetch company info (required)
        try:
            company_info = self.fundamentals_fetcher.fetch_company_info(symbol)
        except DataFetchError as e:
            logger.error(f"Failed to fetch company info for {symbol}: {e}")
            raise

        # Fetch financial statements (optional)
        financials = None
        balance_sheet = None
        cash_flow = None

        if request.include_financials:
            financials = self.fundamentals_fetcher.fetch_financials(symbol)
            balance_sheet = self.fundamentals_fetcher.fetch_balance_sheet(symbol)
            cash_flow = self.fundamentals_fetcher.fetch_cash_flow(symbol)

        # Calculate financial ratios (optional)
        ratios = None
        if request.include_ratios:
            # Get cached info for ratio calculations
            info = self.fundamentals_fetcher.get_cached_info(symbol) or {}
            ratios = self.metrics_calculator.calculate_all_ratios(
                info=info,
                financials=financials,
                balance_sheet=balance_sheet
            )

        return FundamentalAnalysisResult(
            symbol=symbol,
            company_info=company_info,
            financials=financials,
            balance_sheet=balance_sheet,
            cash_flow=cash_flow,
            ratios=ratios
        )

    def get_company_info(self, symbol: str) -> CompanyInfo:
        """
        Get company information only.

        Args:
            symbol: Stock symbol

        Returns:
            CompanyInfo object
        """
        return self.fundamentals_fetcher.fetch_company_info(symbol)

    def get_ratios(self, symbol: str) -> FinancialRatios:
        """
        Get financial ratios only.

        Args:
            symbol: Stock symbol

        Returns:
            FinancialRatios object
        """
        # Fetch required data
        try:
            company_info = self.fundamentals_fetcher.fetch_company_info(symbol)
        except DataFetchError as e:
            logger.error(f"Failed to fetch company info for {symbol}: {e}")
            raise

        financials = self.fundamentals_fetcher.fetch_financials(symbol)
        balance_sheet = self.fundamentals_fetcher.fetch_balance_sheet(symbol)

        # Calculate ratios
        info = self.fundamentals_fetcher.get_cached_info(symbol) or {}
        return self.metrics_calculator.calculate_all_ratios(
            info=info,
            financials=financials,
            balance_sheet=balance_sheet
        )

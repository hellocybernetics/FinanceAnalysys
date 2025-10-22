"""
Technical analysis service for unified technical analysis operations.
"""

from typing import List, Dict, Optional, Any
import pandas as pd
import logging
from src.core.models import (
    TechnicalAnalysisRequest,
    TechnicalAnalysisResult,
    TechnicalSummary,
    IndicatorConfig
)
from src.core.exceptions import DataFetchError, IndicatorCalculationError
from src.data.data_fetcher import DataFetcher
from src.analysis.technical_indicators import TechnicalAnalysis
from src.services.cache_manager import CacheManager

logger = logging.getLogger(__name__)


class TechnicalAnalysisService:
    """
    Unified service for technical analysis.
    Provides a single interface for WebGUI, MCP, and CLI clients.
    """

    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """
        Initialize the technical analysis service.

        Args:
            cache_manager: Optional cache manager instance (creates new if None)
        """
        self.data_fetcher = DataFetcher()
        self.technical_analyzer = TechnicalAnalysis()
        self.cache_manager = cache_manager or CacheManager()

    def analyze(self, request: TechnicalAnalysisRequest) -> List[TechnicalAnalysisResult]:
        """
        Perform technical analysis on multiple symbols.

        Args:
            request: TechnicalAnalysisRequest with analysis parameters

        Returns:
            List of TechnicalAnalysisResult for each symbol

        Raises:
            DataFetchError: If data fetching fails
            IndicatorCalculationError: If indicator calculation fails
        """
        logger.info(f"Starting technical analysis for {len(request.symbols)} symbols")
        results = []

        # Fetch data for all symbols
        try:
            data = self.data_fetcher.fetch_data(
                symbols=request.symbols,
                period=request.period,
                interval=request.interval,
                use_cache=request.use_cache,
                cache_max_age=request.cache_max_age
            )
        except Exception as e:
            logger.error(f"Data fetch failed: {e}")
            raise DataFetchError(f"Failed to fetch data: {e}")

        # Process each symbol
        for symbol, df in data.items():
            try:
                result = self._analyze_symbol(symbol, df, request.indicators)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")
                # Continue with other symbols instead of failing completely
                continue

        # Log failed symbols
        failed_symbols = self.data_fetcher.get_failed_symbols()
        if failed_symbols:
            logger.warning(f"Failed to fetch data for: {failed_symbols}")

        logger.info(f"Technical analysis completed for {len(results)} symbols")
        return results

    def _analyze_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        indicators: List[IndicatorConfig]
    ) -> TechnicalAnalysisResult:
        """
        Analyze a single symbol.

        Args:
            symbol: Stock symbol
            df: Price data DataFrame
            indicators: List of indicators to calculate

        Returns:
            TechnicalAnalysisResult for the symbol
        """
        # Get company name
        company_name = self.data_fetcher.get_company_name(symbol)

        # Calculate indicators
        try:
            # Convert IndicatorConfig objects to dict format expected by technical_analyzer
            indicator_configs = [
                {'name': ind.name, 'params': ind.params, 'plot': ind.plot}
                for ind in indicators
            ]
            df_with_indicators = self.technical_analyzer.calculate_indicators(df, indicator_configs)
        except Exception as e:
            logger.error(f"Indicator calculation failed for {symbol}: {e}")
            raise IndicatorCalculationError(f"Failed to calculate indicators for {symbol}: {e}")

        # Generate summary
        summary = self._generate_summary(df_with_indicators, indicators)

        # Generate signals
        signals = self._generate_signals(df_with_indicators)

        # Generate chart specification
        chart_spec = self._generate_chart_spec(df_with_indicators, symbol, company_name, indicators)

        return TechnicalAnalysisResult(
            symbol=symbol,
            company_name=company_name,
            data=df_with_indicators,
            summary=summary,
            chart_spec=chart_spec
        )

    def _generate_summary(self, df: pd.DataFrame, indicators: List[IndicatorConfig]) -> TechnicalSummary:
        """
        Generate summary of latest indicator values.

        Args:
            df: DataFrame with indicators
            indicators: List of calculated indicators

        Returns:
            TechnicalSummary object
        """
        if df.empty:
            return TechnicalSummary(
                latest_price=0.0,
                price_change=0.0,
                price_change_pct=0.0
            )

        # Get latest and previous closing prices
        latest_price = float(df['Close'].iloc[-1])
        if len(df) > 1:
            prev_price = float(df['Close'].iloc[-2])
            price_change = latest_price - prev_price
            price_change_pct = (price_change / prev_price) * 100
        else:
            price_change = 0.0
            price_change_pct = 0.0

        # Get volume if available
        volume = float(df['Volume'].iloc[-1]) if 'Volume' in df.columns else None

        # Extract latest indicator values
        indicator_values = {}
        for ind in indicators:
            # Find columns related to this indicator
            for col in df.columns:
                if col.startswith(ind.name):
                    try:
                        value = df[col].iloc[-1]
                        if pd.notna(value):
                            indicator_values[col] = float(value)
                    except Exception:
                        continue

        # Generate simple signals based on indicators
        signals = self._generate_simple_signals(df, indicator_values)

        return TechnicalSummary(
            latest_price=latest_price,
            price_change=price_change,
            price_change_pct=price_change_pct,
            volume=volume,
            indicators=indicator_values,
            signals=signals
        )

    def _generate_simple_signals(self, df: pd.DataFrame, indicator_values: Dict[str, float]) -> Dict[str, str]:
        """
        Generate simple trading signals based on indicators.

        Args:
            df: DataFrame with price and indicators
            indicator_values: Latest indicator values

        Returns:
            Dictionary of signals
        """
        signals = {}

        # RSI signals
        for key, value in indicator_values.items():
            if key.startswith('RSI_'):
                if value < 30:
                    signals['RSI'] = 'oversold'
                elif value > 70:
                    signals['RSI'] = 'overbought'
                else:
                    signals['RSI'] = 'neutral'

        # MACD signals
        if any(k.startswith('MACD_') for k in indicator_values.keys()):
            macd_line = next((v for k, v in indicator_values.items() if k.startswith('MACD_') and 'Signal' not in k and 'Hist' not in k), None)
            signal_line = next((v for k, v in indicator_values.items() if 'MACD_Signal' in k), None)
            if macd_line is not None and signal_line is not None:
                signals['MACD'] = 'bullish' if macd_line > signal_line else 'bearish'

        # Moving average signals
        latest_price = float(df['Close'].iloc[-1])
        for key, value in indicator_values.items():
            if key.startswith('SMA_') or key.startswith('EMA_'):
                indicator_type = key.split('_')[0]
                if latest_price > value:
                    signals[key] = 'above'
                else:
                    signals[key] = 'below'

        return signals

    def _generate_signals(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Generate detailed trading signals (placeholder for future enhancement).

        Args:
            df: DataFrame with price and indicators

        Returns:
            Dictionary of trading signals or None
        """
        # Placeholder for more sophisticated signal generation
        # Can be extended with strategy-based signals
        return None

    def _generate_chart_spec(
        self,
        df: pd.DataFrame,
        symbol: str,
        company_name: str,
        indicators: List[IndicatorConfig]
    ) -> Dict[str, Any]:
        """
        Generate chart specification for visualization.

        Args:
            df: DataFrame with price and indicators
            symbol: Stock symbol
            company_name: Company name
            indicators: List of indicators

        Returns:
            Dictionary containing chart specification
        """
        # Return minimal spec - actual rendering done by caller (Visualizer, WebGUI, etc.)
        indicator_configs = [
            {'name': ind.name, 'params': ind.params, 'plot': ind.plot}
            for ind in indicators
        ]

        return {
            'symbol': symbol,
            'company_name': company_name,
            'indicators': indicator_configs,
            'data_shape': df.shape,
            'date_range': {
                'start': str(df.index[0]) if not df.empty else None,
                'end': str(df.index[-1]) if not df.empty else None
            }
        }

    def get_failed_symbols(self) -> List[str]:
        """
        Get list of symbols that failed during data fetching.

        Returns:
            List of failed symbol strings
        """
        return self.data_fetcher.get_failed_symbols()

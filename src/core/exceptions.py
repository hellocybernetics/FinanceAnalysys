"""
Custom exceptions for the financial analysis system.
"""


class FinanceAnalysisError(Exception):
    """Base exception for all finance analysis errors."""
    pass


class DataFetchError(FinanceAnalysisError):
    """Raised when data fetching fails."""
    pass


class IndicatorCalculationError(FinanceAnalysisError):
    """Raised when indicator calculation fails."""
    pass


class VisualizationError(FinanceAnalysisError):
    """Raised when visualization generation fails."""
    pass


class CacheError(FinanceAnalysisError):
    """Raised when cache operations fail."""
    pass

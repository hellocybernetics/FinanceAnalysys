"""
Pydantic models for type-safe data transfer between services.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
import pandas as pd


# Configuration to allow arbitrary types like pandas DataFrames
class BaseConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


# ============================================================================
# Technical Analysis Models
# ============================================================================

class IndicatorConfig(BaseModel):
    """Configuration for a single technical indicator."""
    name: str = Field(..., description="Indicator name (e.g., 'SMA', 'RSI')")
    params: Dict[str, Any] = Field(default_factory=dict, description="Indicator parameters")
    plot: bool = Field(default=True, description="Whether to include in visualization")


class TechnicalAnalysisRequest(BaseModel):
    """Request for technical analysis."""
    symbols: List[str] = Field(..., description="List of stock symbols to analyze")
    period: str = Field(default="1y", description="Time period (e.g., '1d', '5d', '1mo', '1y')")
    interval: str = Field(default="1d", description="Data interval (e.g., '1m', '1h', '1d')")
    indicators: List[IndicatorConfig] = Field(default_factory=list, description="Technical indicators to calculate")
    use_cache: bool = Field(default=True, description="Whether to use cached data")
    cache_max_age: int = Field(default=30, description="Maximum cache age in minutes")


class TechnicalSummary(BaseModel):
    """Summary of technical indicators for a symbol."""
    latest_price: float
    price_change: float
    price_change_pct: float
    volume: Optional[float] = None
    indicators: Dict[str, float] = Field(default_factory=dict, description="Latest indicator values")
    signals: Dict[str, str] = Field(default_factory=dict, description="Trading signals (buy/sell/hold)")


class TechnicalAnalysisResult(BaseConfig):
    """Result of technical analysis for a single symbol."""
    symbol: str
    company_name: str
    data: pd.DataFrame = Field(..., description="Price data with calculated indicators")
    summary: TechnicalSummary
    chart_spec: Optional[Dict[str, Any]] = Field(default=None, description="Plotly chart specification")
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================================
# Fundamental Analysis Models
# ============================================================================

class FundamentalAnalysisRequest(BaseModel):
    """Request for fundamental analysis."""
    symbols: List[str] = Field(..., description="List of stock symbols to analyze")
    include_financials: bool = Field(default=True, description="Include financial statements")
    include_ratios: bool = Field(default=True, description="Calculate financial ratios")
    include_info: bool = Field(default=True, description="Include company information")


class CompanyInfo(BaseModel):
    """Company information."""
    symbol: str
    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    employees: Optional[int] = None
    description: Optional[str] = None
    website: Optional[str] = None
    country: Optional[str] = None


class ValuationRatios(BaseModel):
    """Valuation ratios."""
    pe_ratio: Optional[float] = Field(default=None, description="Price-to-Earnings ratio")
    pb_ratio: Optional[float] = Field(default=None, description="Price-to-Book ratio")
    ps_ratio: Optional[float] = Field(default=None, description="Price-to-Sales ratio")
    peg_ratio: Optional[float] = Field(default=None, description="PEG ratio")
    ev_to_ebitda: Optional[float] = Field(default=None, description="EV/EBITDA ratio")
    dividend_yield: Optional[float] = Field(default=None, description="Dividend yield")


class ProfitabilityRatios(BaseModel):
    """Profitability ratios."""
    roe: Optional[float] = Field(default=None, description="Return on Equity")
    roa: Optional[float] = Field(default=None, description="Return on Assets")
    gross_margin: Optional[float] = Field(default=None, description="Gross profit margin")
    operating_margin: Optional[float] = Field(default=None, description="Operating profit margin")
    net_margin: Optional[float] = Field(default=None, description="Net profit margin")
    ebitda_margin: Optional[float] = Field(default=None, description="EBITDA margin")


class LiquidityRatios(BaseModel):
    """Liquidity ratios."""
    current_ratio: Optional[float] = Field(default=None, description="Current ratio")
    quick_ratio: Optional[float] = Field(default=None, description="Quick ratio (acid test)")
    cash_ratio: Optional[float] = Field(default=None, description="Cash ratio")
    working_capital: Optional[float] = Field(default=None, description="Working capital")


class LeverageRatios(BaseModel):
    """Leverage/Solvency ratios."""
    debt_to_equity: Optional[float] = Field(default=None, description="Debt-to-Equity ratio")
    debt_to_assets: Optional[float] = Field(default=None, description="Debt-to-Assets ratio")
    interest_coverage: Optional[float] = Field(default=None, description="Interest coverage ratio")
    equity_multiplier: Optional[float] = Field(default=None, description="Equity multiplier")


class FinancialRatios(BaseModel):
    """All financial ratios."""
    valuation: ValuationRatios = Field(default_factory=ValuationRatios)
    profitability: ProfitabilityRatios = Field(default_factory=ProfitabilityRatios)
    liquidity: LiquidityRatios = Field(default_factory=LiquidityRatios)
    leverage: LeverageRatios = Field(default_factory=LeverageRatios)


class FundamentalAnalysisResult(BaseConfig):
    """Result of fundamental analysis for a single symbol."""
    symbol: str
    company_info: CompanyInfo
    financials: Optional[pd.DataFrame] = Field(default=None, description="Income statement")
    balance_sheet: Optional[pd.DataFrame] = Field(default=None, description="Balance sheet")
    cash_flow: Optional[pd.DataFrame] = Field(default=None, description="Cash flow statement")
    ratios: Optional[FinancialRatios] = Field(default=None, description="Financial ratios")
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================================
# Export Models
# ============================================================================

class ExportFormat(str):
    """Supported export formats."""
    PNG = "png"
    JPEG = "jpeg"
    SVG = "svg"
    HTML = "html"
    JSON = "json"


class ExportRequest(BaseModel):
    """Request for chart export."""
    format: Literal["png", "jpeg", "svg", "html", "json"] = Field(default="png")
    width: Optional[int] = Field(default=None, description="Image width in pixels")
    height: Optional[int] = Field(default=None, description="Image height in pixels")
    scale: float = Field(default=1.0, description="Image scale factor")

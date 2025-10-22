"""
Example MCP server implementation using the new service architecture.

This demonstrates how to integrate TechnicalAnalysisService and FundamentalAnalysisService
into an MCP server for use with Claude Code or other MCP clients.

Usage:
    This is a reference implementation. To use it, you would need to:
    1. Install fastmcp: pip install fastmcp
    2. Run: python mcp_server_example.py
    3. Configure Claude Code to use this MCP server
"""

from typing import List, Dict, Any
from fastmcp import FastMCP

from src.services.technical_service import TechnicalAnalysisService
from src.services.fundamental_service import FundamentalAnalysisService
from src.core.models import (
    TechnicalAnalysisRequest,
    FundamentalAnalysisRequest,
    IndicatorConfig
)

# Initialize MCP server
mcp = FastMCP("Financial Analysis")

# Initialize services
technical_service = TechnicalAnalysisService()
fundamental_service = FundamentalAnalysisService()


@mcp.tool()
def analyze_technical(
    symbols: List[str],
    period: str = "1y",
    interval: str = "1d",
    indicators: List[str] = None
) -> Dict[str, Any]:
    """
    Perform technical analysis on stock symbols.

    Args:
        symbols: List of stock symbols (e.g., ['AAPL', 'GOOGL'])
        period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y')
        interval: Data interval ('1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo')
        indicators: List of indicator names (e.g., ['SMA', 'RSI', 'MACD'])

    Returns:
        Dictionary with analysis results for each symbol
    """
    # Default indicators if none provided
    if indicators is None:
        indicators = ['SMA', 'RSI', 'MACD']

    # Convert indicator names to IndicatorConfig objects
    indicator_configs = []
    for ind_name in indicators:
        if ind_name == 'SMA':
            indicator_configs.append(IndicatorConfig(name='SMA', params={'length': 20}))
        elif ind_name == 'RSI':
            indicator_configs.append(IndicatorConfig(name='RSI', params={'length': 14}))
        elif ind_name == 'MACD':
            indicator_configs.append(IndicatorConfig(name='MACD', params={'fast': 12, 'slow': 26, 'signal': 9}))
        elif ind_name == 'EMA':
            indicator_configs.append(IndicatorConfig(name='EMA', params={'length': 50}))
        elif ind_name == 'BBands':
            indicator_configs.append(IndicatorConfig(name='BBands', params={'length': 20, 'std': 2}))

    # Create request
    request = TechnicalAnalysisRequest(
        symbols=symbols,
        period=period,
        interval=interval,
        indicators=indicator_configs
    )

    # Run analysis
    results = technical_service.analyze(request)

    # Convert results to JSON-serializable format
    output = {}
    for result in results:
        output[result.symbol] = {
            'company_name': result.company_name,
            'summary': {
                'latest_price': result.summary.latest_price,
                'price_change': result.summary.price_change,
                'price_change_pct': result.summary.price_change_pct,
                'volume': result.summary.volume,
                'indicators': result.summary.indicators,
                'signals': result.summary.signals
            },
            'data_shape': result.data.shape,
            'timestamp': result.timestamp.isoformat()
        }

    return output


@mcp.tool()
def analyze_fundamental(
    symbols: List[str],
    include_financials: bool = True,
    include_ratios: bool = True
) -> Dict[str, Any]:
    """
    Perform fundamental analysis on stock symbols.

    Args:
        symbols: List of stock symbols (e.g., ['AAPL', 'GOOGL'])
        include_financials: Whether to include financial statements
        include_ratios: Whether to calculate financial ratios

    Returns:
        Dictionary with fundamental analysis results for each symbol
    """
    # Create request
    request = FundamentalAnalysisRequest(
        symbols=symbols,
        include_financials=include_financials,
        include_ratios=include_ratios
    )

    # Run analysis
    results = fundamental_service.analyze(request)

    # Convert results to JSON-serializable format
    output = {}
    for result in results:
        symbol_data = {
            'company_info': {
                'name': result.company_info.name,
                'sector': result.company_info.sector,
                'industry': result.company_info.industry,
                'market_cap': result.company_info.market_cap,
                'employees': result.company_info.employees,
                'country': result.company_info.country,
                'website': result.company_info.website
            },
            'timestamp': result.timestamp.isoformat()
        }

        # Add ratios if available
        if result.ratios:
            symbol_data['ratios'] = {
                'valuation': {
                    'pe_ratio': result.ratios.valuation.pe_ratio,
                    'pb_ratio': result.ratios.valuation.pb_ratio,
                    'ps_ratio': result.ratios.valuation.ps_ratio,
                    'dividend_yield': result.ratios.valuation.dividend_yield
                },
                'profitability': {
                    'roe': result.ratios.profitability.roe,
                    'roa': result.ratios.profitability.roa,
                    'gross_margin': result.ratios.profitability.gross_margin,
                    'net_margin': result.ratios.profitability.net_margin
                },
                'liquidity': {
                    'current_ratio': result.ratios.liquidity.current_ratio,
                    'quick_ratio': result.ratios.liquidity.quick_ratio
                },
                'leverage': {
                    'debt_to_equity': result.ratios.leverage.debt_to_equity,
                    'debt_to_assets': result.ratios.leverage.debt_to_assets
                }
            }

        output[result.symbol] = symbol_data

    return output


@mcp.tool()
def get_company_info(symbol: str) -> Dict[str, Any]:
    """
    Get company information for a single symbol.

    Args:
        symbol: Stock symbol (e.g., 'AAPL')

    Returns:
        Dictionary with company information
    """
    company_info = fundamental_service.get_company_info(symbol)

    return {
        'symbol': symbol,
        'name': company_info.name,
        'sector': company_info.sector,
        'industry': company_info.industry,
        'market_cap': company_info.market_cap,
        'employees': company_info.employees,
        'description': company_info.description,
        'website': company_info.website,
        'country': company_info.country
    }


@mcp.tool()
def compare_stocks(symbols: List[str], metrics: List[str] = None) -> Dict[str, Any]:
    """
    Compare multiple stocks across key metrics.

    Args:
        symbols: List of stock symbols to compare (e.g., ['AAPL', 'GOOGL', 'MSFT'])
        metrics: List of metrics to compare (default: ['price', 'pe_ratio', 'market_cap'])

    Returns:
        Dictionary with comparison data
    """
    if metrics is None:
        metrics = ['price', 'pe_ratio', 'market_cap', 'roe']

    # Get technical data for price
    tech_request = TechnicalAnalysisRequest(
        symbols=symbols,
        period='1d',
        interval='1d',
        indicators=[]
    )
    tech_results = technical_service.analyze(tech_request)

    # Get fundamental data for ratios
    fund_request = FundamentalAnalysisRequest(
        symbols=symbols,
        include_financials=False,
        include_ratios=True
    )
    fund_results = fundamental_service.analyze(fund_request)

    # Build comparison
    comparison = {}
    for tech_result, fund_result in zip(tech_results, fund_results):
        symbol = tech_result.symbol
        comparison[symbol] = {}

        if 'price' in metrics:
            comparison[symbol]['price'] = tech_result.summary.latest_price
        if 'pe_ratio' in metrics and fund_result.ratios:
            comparison[symbol]['pe_ratio'] = fund_result.ratios.valuation.pe_ratio
        if 'market_cap' in metrics:
            comparison[symbol]['market_cap'] = fund_result.company_info.market_cap
        if 'roe' in metrics and fund_result.ratios:
            comparison[symbol]['roe'] = fund_result.ratios.profitability.roe

    return comparison


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()

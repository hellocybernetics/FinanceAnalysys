"""
Fundamental metrics calculator for financial ratios and analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from src.core.models import (
    ValuationRatios,
    ProfitabilityRatios,
    LiquidityRatios,
    LeverageRatios,
    FinancialRatios
)

logger = logging.getLogger(__name__)


class FundamentalMetrics:
    """
    Calculator for fundamental financial metrics and ratios.
    """

    def __init__(self):
        """Initialize the FundamentalMetrics calculator."""
        pass

    def calculate_valuation_ratios(
        self,
        info: Dict,
        financials: Optional[pd.DataFrame] = None
    ) -> ValuationRatios:
        """
        Calculate valuation ratios.

        Args:
            info: Ticker info dictionary from yfinance
            financials: Income statement DataFrame (optional)

        Returns:
            ValuationRatios object
        """
        try:
            return ValuationRatios(
                pe_ratio=info.get('trailingPE') or info.get('forwardPE'),
                pb_ratio=info.get('priceToBook'),
                ps_ratio=info.get('priceToSalesTrailing12Months'),
                peg_ratio=info.get('pegRatio'),
                ev_to_ebitda=info.get('enterpriseToEbitda'),
                dividend_yield=info.get('dividendYield')
            )
        except Exception as e:
            logger.error(f"Error calculating valuation ratios: {e}")
            return ValuationRatios()

    def calculate_profitability_ratios(
        self,
        info: Dict,
        financials: Optional[pd.DataFrame] = None,
        balance_sheet: Optional[pd.DataFrame] = None
    ) -> ProfitabilityRatios:
        """
        Calculate profitability ratios.

        Args:
            info: Ticker info dictionary from yfinance
            financials: Income statement DataFrame (optional)
            balance_sheet: Balance sheet DataFrame (optional)

        Returns:
            ProfitabilityRatios object
        """
        try:
            ratios = ProfitabilityRatios(
                roe=info.get('returnOnEquity'),
                roa=info.get('returnOnAssets'),
                gross_margin=info.get('grossMargins'),
                operating_margin=info.get('operatingMargins'),
                net_margin=info.get('profitMargins'),
                ebitda_margin=info.get('ebitdaMargins')
            )

            # Calculate additional ratios from financial statements if available
            if financials is not None and not financials.empty and balance_sheet is not None and not balance_sheet.empty:
                try:
                    # Get most recent column (latest financial data)
                    latest_col = financials.columns[0]

                    # Calculate ROE if not in info
                    if ratios.roe is None:
                        net_income = self._get_value(financials, 'Net Income', latest_col)
                        total_equity = self._get_value(balance_sheet, 'Total Equity Gross Minority Interest', latest_col) or \
                                     self._get_value(balance_sheet, 'Stockholders Equity', latest_col)
                        if net_income and total_equity and total_equity != 0:
                            ratios.roe = net_income / total_equity

                    # Calculate ROA if not in info
                    if ratios.roa is None:
                        net_income = self._get_value(financials, 'Net Income', latest_col)
                        total_assets = self._get_value(balance_sheet, 'Total Assets', latest_col)
                        if net_income and total_assets and total_assets != 0:
                            ratios.roa = net_income / total_assets

                except Exception as e:
                    logger.warning(f"Could not calculate additional profitability ratios: {e}")

            return ratios
        except Exception as e:
            logger.error(f"Error calculating profitability ratios: {e}")
            return ProfitabilityRatios()

    def calculate_liquidity_ratios(
        self,
        balance_sheet: Optional[pd.DataFrame] = None
    ) -> LiquidityRatios:
        """
        Calculate liquidity ratios.

        Args:
            balance_sheet: Balance sheet DataFrame

        Returns:
            LiquidityRatios object
        """
        if balance_sheet is None or balance_sheet.empty:
            return LiquidityRatios()

        try:
            # Get most recent column (latest balance sheet data)
            latest_col = balance_sheet.columns[0]

            # Get balance sheet items
            current_assets = self._get_value(balance_sheet, 'Current Assets', latest_col)
            current_liabilities = self._get_value(balance_sheet, 'Current Liabilities', latest_col)
            cash = self._get_value(balance_sheet, 'Cash And Cash Equivalents', latest_col) or \
                   self._get_value(balance_sheet, 'Cash', latest_col)
            inventory = self._get_value(balance_sheet, 'Inventory', latest_col)

            # Calculate ratios
            current_ratio = None
            quick_ratio = None
            cash_ratio = None
            working_capital = None

            if current_assets and current_liabilities and current_liabilities != 0:
                current_ratio = current_assets / current_liabilities
                working_capital = current_assets - current_liabilities

                # Quick ratio = (Current Assets - Inventory) / Current Liabilities
                if inventory is not None:
                    quick_ratio = (current_assets - inventory) / current_liabilities
                else:
                    quick_ratio = current_ratio  # Fallback if inventory not available

                # Cash ratio = Cash / Current Liabilities
                if cash is not None:
                    cash_ratio = cash / current_liabilities

            return LiquidityRatios(
                current_ratio=current_ratio,
                quick_ratio=quick_ratio,
                cash_ratio=cash_ratio,
                working_capital=working_capital
            )
        except Exception as e:
            logger.error(f"Error calculating liquidity ratios: {e}")
            return LiquidityRatios()

    def calculate_leverage_ratios(
        self,
        info: Dict,
        balance_sheet: Optional[pd.DataFrame] = None,
        financials: Optional[pd.DataFrame] = None
    ) -> LeverageRatios:
        """
        Calculate leverage/solvency ratios.

        Args:
            info: Ticker info dictionary from yfinance
            balance_sheet: Balance sheet DataFrame (optional)
            financials: Income statement DataFrame (optional)

        Returns:
            LeverageRatios object
        """
        try:
            ratios = LeverageRatios(
                debt_to_equity=info.get('debtToEquity')
            )

            # Calculate additional ratios from financial statements if available
            if balance_sheet is not None and not balance_sheet.empty:
                try:
                    latest_col = balance_sheet.columns[0]

                    total_debt = self._get_value(balance_sheet, 'Total Debt', latest_col) or \
                               self._get_value(balance_sheet, 'Long Term Debt', latest_col)
                    total_equity = self._get_value(balance_sheet, 'Total Equity Gross Minority Interest', latest_col) or \
                                 self._get_value(balance_sheet, 'Stockholders Equity', latest_col)
                    total_assets = self._get_value(balance_sheet, 'Total Assets', latest_col)

                    # Debt-to-Equity ratio
                    if ratios.debt_to_equity is None and total_debt and total_equity and total_equity != 0:
                        ratios.debt_to_equity = total_debt / total_equity

                    # Debt-to-Assets ratio
                    if total_debt and total_assets and total_assets != 0:
                        ratios.debt_to_assets = total_debt / total_assets

                    # Equity multiplier
                    if total_assets and total_equity and total_equity != 0:
                        ratios.equity_multiplier = total_assets / total_equity

                except Exception as e:
                    logger.warning(f"Could not calculate balance sheet leverage ratios: {e}")

            # Calculate interest coverage if financial statements available
            if financials is not None and not financials.empty:
                try:
                    latest_col = financials.columns[0]

                    ebit = self._get_value(financials, 'EBIT', latest_col) or \
                          self._get_value(financials, 'Operating Income', latest_col)
                    interest_expense = self._get_value(financials, 'Interest Expense', latest_col)

                    if ebit and interest_expense and interest_expense != 0:
                        # Interest expense is typically negative, so use absolute value
                        ratios.interest_coverage = ebit / abs(interest_expense)

                except Exception as e:
                    logger.warning(f"Could not calculate interest coverage: {e}")

            return ratios
        except Exception as e:
            logger.error(f"Error calculating leverage ratios: {e}")
            return LeverageRatios()

    def calculate_all_ratios(
        self,
        info: Dict,
        financials: Optional[pd.DataFrame] = None,
        balance_sheet: Optional[pd.DataFrame] = None
    ) -> FinancialRatios:
        """
        Calculate all financial ratios.

        Args:
            info: Ticker info dictionary from yfinance
            financials: Income statement DataFrame (optional)
            balance_sheet: Balance sheet DataFrame (optional)

        Returns:
            FinancialRatios object containing all ratio categories
        """
        return FinancialRatios(
            valuation=self.calculate_valuation_ratios(info, financials),
            profitability=self.calculate_profitability_ratios(info, financials, balance_sheet),
            liquidity=self.calculate_liquidity_ratios(balance_sheet),
            leverage=self.calculate_leverage_ratios(info, balance_sheet, financials)
        )

    def _get_value(self, df: pd.DataFrame, row_name: str, col: str) -> Optional[float]:
        """
        Safely get a value from a DataFrame.

        Args:
            df: DataFrame to extract from
            row_name: Row index name
            col: Column name

        Returns:
            Float value or None if not found
        """
        try:
            if row_name in df.index:
                value = df.loc[row_name, col]
                if pd.notna(value):
                    return float(value)
        except Exception:
            pass
        return None

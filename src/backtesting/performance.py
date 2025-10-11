"""
Performance calculation module for backtesting.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PerformanceCalculator:
    """
    Class for calculating performance metrics from backtest results.
    """

    @staticmethod
    def calculate_total_return(equity_curve: pd.Series) -> float:
        """
        Calculate total return from equity curve.

        Args:
            equity_curve (pd.Series): Equity curve over time.

        Returns:
            float: Total return as a decimal (e.g., 0.15 for 15% return).
        """
        if equity_curve.empty or equity_curve.iloc[0] == 0:
            return 0.0

        return (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns (pd.Series): Series of returns.
            risk_free_rate (float): Risk-free rate (annualized).

        Returns:
            float: Sharpe ratio.
        """
        if returns.empty or returns.std() == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown from equity curve.

        Args:
            equity_curve (pd.Series): Equity curve over time.

        Returns:
            float: Maximum drawdown as a decimal (negative value).
        """
        if equity_curve.empty:
            return 0.0

        cumulative_max = equity_curve.cummax()
        drawdown = (equity_curve - cumulative_max) / cumulative_max
        return drawdown.min()

    @staticmethod
    def calculate_win_rate(trades: pd.DataFrame) -> float:
        """
        Calculate win rate from trades.

        Args:
            trades (pd.DataFrame): DataFrame containing trade information with 'pnl' column.

        Returns:
            float: Win rate as a decimal (e.g., 0.6 for 60% win rate).
        """
        if trades.empty or 'pnl' not in trades.columns:
            return 0.0

        winning_trades = trades[trades['pnl'] > 0]
        return len(winning_trades) / len(trades) if len(trades) > 0 else 0.0

    @staticmethod
    def calculate_profit_factor(trades: pd.DataFrame) -> float:
        """
        Calculate profit factor (gross profit / gross loss).

        Args:
            trades (pd.DataFrame): DataFrame containing trade information with 'pnl' column.

        Returns:
            float: Profit factor.
        """
        if trades.empty or 'pnl' not in trades.columns:
            return 0.0

        gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())

        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    @staticmethod
    def calculate_all_metrics(equity_curve: pd.Series, trades: pd.DataFrame) -> dict:
        """
        Calculate all performance metrics.

        Args:
            equity_curve (pd.Series): Equity curve over time.
            trades (pd.DataFrame): DataFrame containing trade information.

        Returns:
            dict: Dictionary containing all performance metrics.
        """
        returns = equity_curve.pct_change().dropna()

        metrics = {
            'total_return': PerformanceCalculator.calculate_total_return(equity_curve),
            'sharpe_ratio': PerformanceCalculator.calculate_sharpe_ratio(returns),
            'max_drawdown': PerformanceCalculator.calculate_max_drawdown(equity_curve),
            'win_rate': PerformanceCalculator.calculate_win_rate(trades),
            'profit_factor': PerformanceCalculator.calculate_profit_factor(trades),
            'total_trades': len(trades),
        }

        logger.info(f"Calculated performance metrics: {metrics}")
        return metrics

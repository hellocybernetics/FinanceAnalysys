"""
Tests for PerformanceCalculator.
"""

import pytest
import pandas as pd
import numpy as np

from src.backtesting.performance import PerformanceCalculator


@pytest.fixture
def performance_calculator():
    """Create a PerformanceCalculator instance."""
    return PerformanceCalculator()


@pytest.fixture
def sample_equity_curve():
    """Create a sample equity curve."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    equity = [10000 + i * 100 + np.random.randn() * 50 for i in range(100)]
    return pd.Series(equity, index=dates)


@pytest.fixture
def sample_trades():
    """Create sample trades data."""
    trades = pd.DataFrame({
        'entry_date': pd.date_range(start='2024-01-01', periods=10, freq='5D'),
        'exit_date': pd.date_range(start='2024-01-06', periods=10, freq='5D'),
        'entry_price': [100, 105, 102, 110, 108, 115, 112, 120, 118, 125],
        'exit_price': [105, 103, 108, 112, 106, 118, 115, 122, 116, 130],
        'pnl': [500, -200, 600, 200, -200, 300, 300, 200, -200, 500],
    })
    return trades


class TestPerformanceCalculator:
    """Test cases for PerformanceCalculator."""

    def test_calculate_total_return_positive(self):
        """Test calculating positive total return."""
        equity = pd.Series([10000, 11000, 12000])
        total_return = PerformanceCalculator.calculate_total_return(equity)

        assert total_return == pytest.approx(0.2, rel=1e-5)  # 20% return

    def test_calculate_total_return_negative(self):
        """Test calculating negative total return."""
        equity = pd.Series([10000, 9000, 8000])
        total_return = PerformanceCalculator.calculate_total_return(equity)

        assert total_return == pytest.approx(-0.2, rel=1e-5)  # -20% return

    def test_calculate_total_return_empty(self):
        """Test calculating total return with empty series."""
        equity = pd.Series([])
        total_return = PerformanceCalculator.calculate_total_return(equity)

        assert total_return == 0.0

    def test_calculate_total_return_zero_initial(self):
        """Test calculating total return with zero initial capital."""
        equity = pd.Series([0, 1000, 2000])
        total_return = PerformanceCalculator.calculate_total_return(equity)

        assert total_return == 0.0

    def test_calculate_sharpe_ratio_positive(self):
        """Test calculating Sharpe ratio with positive returns."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.02, 0.01, -0.005, 0.02])
        sharpe = PerformanceCalculator.calculate_sharpe_ratio(returns)

        assert isinstance(sharpe, float)
        assert sharpe > 0

    def test_calculate_sharpe_ratio_negative(self):
        """Test calculating Sharpe ratio with negative returns."""
        returns = pd.Series([-0.01, -0.02, -0.01, -0.015, -0.02, -0.01, -0.005, -0.02])
        sharpe = PerformanceCalculator.calculate_sharpe_ratio(returns)

        assert isinstance(sharpe, float)
        assert sharpe < 0

    def test_calculate_sharpe_ratio_empty(self):
        """Test calculating Sharpe ratio with empty series."""
        returns = pd.Series([])
        sharpe = PerformanceCalculator.calculate_sharpe_ratio(returns)

        assert sharpe == 0.0

    def test_calculate_sharpe_ratio_zero_std(self):
        """Test calculating Sharpe ratio with zero standard deviation."""
        returns = pd.Series([0.01, 0.01, 0.01, 0.01])
        sharpe = PerformanceCalculator.calculate_sharpe_ratio(returns)

        assert sharpe == 0.0

    def test_calculate_max_drawdown(self):
        """Test calculating maximum drawdown."""
        equity = pd.Series([10000, 11000, 10500, 9500, 10000, 11500])
        max_dd = PerformanceCalculator.calculate_max_drawdown(equity)

        # Maximum drawdown should be when equity drops from 11000 to 9500
        expected_dd = (9500 - 11000) / 11000
        assert max_dd == pytest.approx(expected_dd, rel=1e-5)

    def test_calculate_max_drawdown_no_drawdown(self):
        """Test calculating maximum drawdown with no drawdown."""
        equity = pd.Series([10000, 11000, 12000, 13000])
        max_dd = PerformanceCalculator.calculate_max_drawdown(equity)

        assert max_dd == 0.0

    def test_calculate_max_drawdown_empty(self):
        """Test calculating maximum drawdown with empty series."""
        equity = pd.Series([])
        max_dd = PerformanceCalculator.calculate_max_drawdown(equity)

        assert max_dd == 0.0

    def test_calculate_win_rate(self, sample_trades):
        """Test calculating win rate."""
        win_rate = PerformanceCalculator.calculate_win_rate(sample_trades)

        # 7 winning trades out of 10
        assert win_rate == pytest.approx(0.7, rel=1e-5)

    def test_calculate_win_rate_all_wins(self):
        """Test calculating win rate with all winning trades."""
        trades = pd.DataFrame({'pnl': [100, 200, 300, 400]})
        win_rate = PerformanceCalculator.calculate_win_rate(trades)

        assert win_rate == 1.0

    def test_calculate_win_rate_all_losses(self):
        """Test calculating win rate with all losing trades."""
        trades = pd.DataFrame({'pnl': [-100, -200, -300, -400]})
        win_rate = PerformanceCalculator.calculate_win_rate(trades)

        assert win_rate == 0.0

    def test_calculate_win_rate_empty(self):
        """Test calculating win rate with empty DataFrame."""
        trades = pd.DataFrame()
        win_rate = PerformanceCalculator.calculate_win_rate(trades)

        assert win_rate == 0.0

    def test_calculate_win_rate_missing_pnl(self):
        """Test calculating win rate with missing pnl column."""
        trades = pd.DataFrame({'entry_price': [100, 110]})
        win_rate = PerformanceCalculator.calculate_win_rate(trades)

        assert win_rate == 0.0

    def test_calculate_profit_factor(self, sample_trades):
        """Test calculating profit factor."""
        profit_factor = PerformanceCalculator.calculate_profit_factor(sample_trades)

        gross_profit = 500 + 600 + 200 + 300 + 300 + 200 + 500  # = 2600
        gross_loss = 200 + 200 + 200  # = 600
        expected_pf = gross_profit / gross_loss

        assert profit_factor == pytest.approx(expected_pf, rel=1e-5)

    def test_calculate_profit_factor_no_losses(self):
        """Test calculating profit factor with no losses."""
        trades = pd.DataFrame({'pnl': [100, 200, 300]})
        profit_factor = PerformanceCalculator.calculate_profit_factor(trades)

        assert profit_factor == np.inf

    def test_calculate_profit_factor_no_profits(self):
        """Test calculating profit factor with no profits."""
        trades = pd.DataFrame({'pnl': [-100, -200, -300]})
        profit_factor = PerformanceCalculator.calculate_profit_factor(trades)

        assert profit_factor == 0.0

    def test_calculate_profit_factor_empty(self):
        """Test calculating profit factor with empty DataFrame."""
        trades = pd.DataFrame()
        profit_factor = PerformanceCalculator.calculate_profit_factor(trades)

        assert profit_factor == 0.0

    def test_calculate_all_metrics(self, sample_equity_curve, sample_trades):
        """Test calculating all metrics together."""
        metrics = PerformanceCalculator.calculate_all_metrics(
            sample_equity_curve, sample_trades
        )

        # Check that all expected metrics are present
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        assert 'profit_factor' in metrics
        assert 'total_trades' in metrics

        # Check that values are of correct type
        assert isinstance(metrics['total_return'], (float, np.floating))
        assert isinstance(metrics['sharpe_ratio'], (float, np.floating))
        assert isinstance(metrics['max_drawdown'], (float, np.floating))
        assert isinstance(metrics['win_rate'], (float, np.floating))
        assert isinstance(metrics['profit_factor'], (float, np.floating))
        assert isinstance(metrics['total_trades'], (int, np.integer))

        # Check that total_trades matches the length of trades DataFrame
        assert metrics['total_trades'] == len(sample_trades)

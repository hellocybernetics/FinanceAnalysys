"""
Tests for BacktestEngine.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for testing

from src.backtesting.engine import BacktestEngine
from src.backtesting.strategy import MovingAverageCrossoverStrategy, RSIStrategy


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)

    # Generate synthetic price data with an uptrend
    close_prices = 100 + np.cumsum(np.random.randn(100) * 2)
    high_prices = close_prices + np.random.rand(100) * 2
    low_prices = close_prices - np.random.rand(100) * 2
    open_prices = close_prices + np.random.randn(100)

    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, 100),
    }, index=dates)

    return df


@pytest.fixture
def backtest_engine():
    """Create a BacktestEngine instance."""
    return BacktestEngine(use_vectorbt=False)


class TestBacktestEngine:
    """Test cases for BacktestEngine."""

    def test_engine_initialization(self, backtest_engine):
        """Test that the engine initializes correctly."""
        assert backtest_engine is not None
        assert backtest_engine.performance_calculator is not None

    def test_run_with_ma_crossover_strategy(self, backtest_engine, sample_data):
        """Test running a backtest with MovingAverageCrossoverStrategy."""
        strategy = MovingAverageCrossoverStrategy(short_length=10, long_length=20)
        result = backtest_engine.run(
            df=sample_data,
            strategy=strategy,
            initial_capital=10000.0,
            commission=0.001,
        )

        # Check that result contains all expected keys
        assert 'signals' in result
        assert 'trades' in result
        assert 'equity_curve' in result
        assert 'total_return' in result
        assert 'sharpe_ratio' in result
        assert 'max_drawdown' in result
        assert 'win_rate' in result
        assert 'profit_factor' in result
        assert 'total_trades' in result

        # Check that signals DataFrame has signal column
        assert 'signal' in result['signals'].columns

        # Check that equity curve is a Series
        assert isinstance(result['equity_curve'], pd.Series)

        # Check that trades is a DataFrame
        assert isinstance(result['trades'], pd.DataFrame)

    def test_run_with_rsi_strategy(self, backtest_engine, sample_data):
        """Test running a backtest with RSIStrategy."""
        strategy = RSIStrategy(rsi_length=14, oversold=30, overbought=70)
        result = backtest_engine.run(
            df=sample_data,
            strategy=strategy,
            initial_capital=10000.0,
            commission=0.001,
        )

        # Check that result contains all expected keys
        assert 'signals' in result
        assert 'trades' in result
        assert 'equity_curve' in result

        # Check that RSI was calculated
        assert 'RSI_14' in result['signals'].columns

    def test_equity_curve_starts_at_initial_capital(self, backtest_engine, sample_data):
        """Test that equity curve starts at initial capital."""
        strategy = MovingAverageCrossoverStrategy(short_length=5, long_length=10)
        initial_capital = 15000.0

        result = backtest_engine.run(
            df=sample_data,
            strategy=strategy,
            initial_capital=initial_capital,
            commission=0.001,
        )

        # First equity value should be close to initial capital
        # (within a small tolerance due to potential immediate trades)
        assert abs(result['equity_curve'].iloc[0] - initial_capital) < initial_capital * 0.1

    def test_commission_reduces_returns(self, backtest_engine, sample_data):
        """Test that commission reduces returns."""
        strategy = MovingAverageCrossoverStrategy(short_length=5, long_length=10)

        # Run with no commission
        result_no_commission = backtest_engine.run(
            df=sample_data,
            strategy=strategy,
            initial_capital=10000.0,
            commission=0.0,
        )

        # Run with commission
        result_with_commission = backtest_engine.run(
            df=sample_data,
            strategy=strategy,
            initial_capital=10000.0,
            commission=0.01,  # 1% commission
        )

        # Final equity with commission should be less than or equal to no commission
        # (unless no trades were executed)
        if result_no_commission['total_trades'] > 0:
            assert (
                result_with_commission['equity_curve'].iloc[-1] <=
                result_no_commission['equity_curve'].iloc[-1]
            )

    def test_visualize_results(self, backtest_engine, sample_data):
        """Test that visualization runs without error."""
        strategy = MovingAverageCrossoverStrategy(short_length=10, long_length=20)
        result = backtest_engine.run(
            df=sample_data,
            strategy=strategy,
            initial_capital=10000.0,
            commission=0.001,
        )

        fig = backtest_engine.visualize_results(
            result=result,
            symbol='TEST',
            strategy_name=strategy.name,
        )

        # Check that figure was created
        assert fig is not None
        assert len(fig.axes) == 3  # Should have 3 subplots

    def test_empty_dataframe_handling(self, backtest_engine):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        strategy = MovingAverageCrossoverStrategy()

        with pytest.raises((ValueError, KeyError)):
            backtest_engine.run(
                df=empty_df,
                strategy=strategy,
                initial_capital=10000.0,
                commission=0.001,
            )

    def test_multiple_strategies_on_same_data(self, backtest_engine, sample_data):
        """Test running multiple strategies on the same data."""
        ma_strategy = MovingAverageCrossoverStrategy(short_length=10, long_length=20)
        rsi_strategy = RSIStrategy(rsi_length=14, oversold=30, overbought=70)

        ma_result = backtest_engine.run(
            df=sample_data,
            strategy=ma_strategy,
            initial_capital=10000.0,
            commission=0.001,
        )

        rsi_result = backtest_engine.run(
            df=sample_data,
            strategy=rsi_strategy,
            initial_capital=10000.0,
            commission=0.001,
        )

        # Both should complete without error
        assert ma_result is not None
        assert rsi_result is not None

        # Results should be independent
        assert not ma_result['equity_curve'].equals(rsi_result['equity_curve'])

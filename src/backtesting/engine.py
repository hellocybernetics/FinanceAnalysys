"""
Backtesting engine for executing and evaluating trading strategies.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.backtesting.strategy import BaseStrategy
from src.backtesting.performance import PerformanceCalculator

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Engine for running backtests on trading strategies.
    """

    def __init__(self, use_vectorbt: bool = False):
        """
        Initialize the backtest engine.

        Args:
            use_vectorbt (bool): Whether to use vectorbt for backtesting (reserved for future use).
        """
        self.use_vectorbt = use_vectorbt
        self.performance_calculator = PerformanceCalculator()

    def run(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        initial_capital: float = 10000.0,
        commission: float = 0.001,
    ) -> dict:
        """
        Run a backtest on the given data with the specified strategy.

        Args:
            df (pd.DataFrame): DataFrame containing OHLCV data.
            strategy (BaseStrategy): Trading strategy to backtest.
            initial_capital (float): Initial capital for the backtest.
            commission (float): Commission rate per trade (e.g., 0.001 for 0.1%).

        Returns:
            dict: Dictionary containing backtest results including:
                - signals: DataFrame with trading signals
                - trades: DataFrame with executed trades
                - equity_curve: Series with equity over time
                - total_return: Total return
                - sharpe_ratio: Sharpe ratio
                - max_drawdown: Maximum drawdown
                - win_rate: Win rate
                - profit_factor: Profit factor
                - total_trades: Total number of trades
        """
        logger.info(f"Running backtest for strategy: {strategy.name}")

        # Generate signals
        df_with_signals = strategy.generate_signals(df)

        # Execute trades based on signals
        trades, equity_curve = self._execute_trades(
            df_with_signals, initial_capital, commission
        )

        # Calculate performance metrics
        metrics = self.performance_calculator.calculate_all_metrics(equity_curve, trades)

        # Combine results
        result = {
            'signals': df_with_signals,
            'trades': trades,
            'equity_curve': equity_curve,
            **metrics,
        }

        logger.info(f"Backtest completed for {strategy.name}")
        return result

    def _execute_trades(
        self,
        df: pd.DataFrame,
        initial_capital: float,
        commission: float,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Execute trades based on signals.

        Args:
            df (pd.DataFrame): DataFrame with signals.
            initial_capital (float): Initial capital.
            commission (float): Commission rate.

        Returns:
            tuple: (trades DataFrame, equity curve Series)
        """
        trades = []
        position = 0  # 0: no position, 1: long position, -1: short position
        entry_price = 0.0
        equity = initial_capital
        cash = initial_capital
        equity_curve = []

        for i, row in df.iterrows():
            signal = row.get('signal', 0.0)
            close_price = row['Close']

            # Execute trade based on signal
            if signal == 1.0 and position == 0:
                # Buy signal - enter long position
                shares = cash / (close_price * (1 + commission))
                position = 1
                entry_price = close_price
                cash = 0
                equity = shares * close_price

                trades.append({
                    'entry_date': i,
                    'entry_price': entry_price,
                    'type': 'buy',
                    'shares': shares,
                })

            elif signal == -1.0 and position == 1:
                # Sell signal - exit long position
                shares = equity / entry_price
                exit_price = close_price
                cash = shares * exit_price * (1 - commission)
                pnl = cash - initial_capital

                # Update the last trade with exit information
                if trades:
                    trades[-1].update({
                        'exit_date': i,
                        'exit_price': exit_price,
                        'pnl': pnl,
                    })

                position = 0
                equity = cash

            # Update equity curve
            if position == 1:
                # If in position, mark-to-market
                shares = equity / entry_price
                equity = shares * close_price
            else:
                equity = cash

            equity_curve.append(equity)

        # Close any open positions at the end
        if position == 1 and trades:
            shares = equity / entry_price
            exit_price = df['Close'].iloc[-1]
            cash = shares * exit_price * (1 - commission)
            pnl = cash - initial_capital

            trades[-1].update({
                'exit_date': df.index[-1],
                'exit_price': exit_price,
                'pnl': pnl,
            })

            equity = cash
            equity_curve[-1] = equity

        trades_df = pd.DataFrame(trades)
        equity_series = pd.Series(equity_curve, index=df.index)

        logger.info(f"Executed {len(trades_df)} trades")
        return trades_df, equity_series

    def visualize_results(
        self,
        result: dict,
        symbol: str,
        strategy_name: str,
    ) -> Figure:
        """
        Visualize backtest results.

        Args:
            result (dict): Backtest results from run method.
            symbol (str): Symbol being backtested.
            strategy_name (str): Name of the strategy.

        Returns:
            Figure: Matplotlib figure with visualization.
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        df = result['signals']
        equity_curve = result['equity_curve']
        trades = result['trades']

        # Plot 1: Price and signals
        ax1 = axes[0]
        ax1.plot(df.index, df['Close'], label='Close Price', linewidth=1)

        # Plot buy signals
        buy_signals = df[df['signal'] == 1.0]
        ax1.scatter(
            buy_signals.index,
            buy_signals['Close'],
            color='green',
            marker='^',
            s=100,
            label='Buy Signal',
            alpha=0.7,
        )

        # Plot sell signals
        sell_signals = df[df['signal'] == -1.0]
        ax1.scatter(
            sell_signals.index,
            sell_signals['Close'],
            color='red',
            marker='v',
            s=100,
            label='Sell Signal',
            alpha=0.7,
        )

        ax1.set_ylabel('Price')
        ax1.set_title(f'{symbol} - {strategy_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Equity curve
        ax2 = axes[1]
        ax2.plot(equity_curve.index, equity_curve, label='Equity', linewidth=1.5)
        ax2.set_ylabel('Equity')
        ax2.set_title('Equity Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Drawdown
        ax3 = axes[2]
        cumulative_max = equity_curve.cummax()
        drawdown = (equity_curve - cumulative_max) / cumulative_max * 100
        ax3.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        ax3.plot(drawdown.index, drawdown, color='red', linewidth=1)
        ax3.set_ylabel('Drawdown (%)')
        ax3.set_xlabel('Date')
        ax3.set_title('Drawdown')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

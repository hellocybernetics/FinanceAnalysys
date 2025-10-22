"""
CLI script for running backtests using the new service architecture.
"""

import os
import sys
import yaml
import json
import argparse
from datetime import datetime
import logging
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtesting.engine import BacktestEngine
from src.backtesting.strategy import MovingAverageCrossoverStrategy, RSIStrategy
from src.data.data_fetcher import DataFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    file_ext = os.path.splitext(config_path)[1].lower()

    if file_ext in ['.yaml', '.yml']:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    elif file_ext == '.json':
        with open(config_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    else:
        raise ValueError(f"Unsupported configuration file format: {file_ext}")


def run_backtest(config_path: Optional[str] = None, output_dir: Optional[str] = None):
    """
    Run backtests using the service layer.

    Args:
        config_path: Path to configuration file (uses default config if None)
        output_dir: Output directory for results
    """
    # Default configuration
    default_config = {
        'data': {
            'symbols': ['AAPL'],
            'period': '1y',
            'interval': '1d'
        },
        'strategies': [
            {
                'name': 'MA_Crossover',
                'params': {
                    'short_length': 20,
                    'long_length': 50
                }
            },
            {
                'name': 'RSI',
                'params': {
                    'rsi_length': 14,
                    'oversold': 30,
                    'overbought': 70
                }
            }
        ],
        'backtest': {
            'initial_capital': 10000,
            'commission': 0.001
        },
        'visualization': {
            'output_dir': 'output'
        }
    }

    # Load and merge config
    config = default_config
    if config_path:
        loaded_config = load_config(config_path)
        if 'data' in loaded_config:
            config['data'].update(loaded_config['data'])
        if 'strategies' in loaded_config:
            config['strategies'] = loaded_config['strategies']
        if 'backtest' in loaded_config:
            config['backtest'].update(loaded_config['backtest'])
        if 'visualization' in loaded_config:
            config['visualization'].update(loaded_config['visualization'])

    # Extract config values
    symbols = config['data']['symbols']
    period = config['data']['period']
    interval = config['data']['interval']
    strategies_config = config['strategies']
    initial_capital = config['backtest'].get('initial_capital', 10000)
    commission = config['backtest'].get('commission', 0.001)

    # Setup output directory
    if output_dir is None:
        output_dir = config['visualization'].get('output_dir', 'output')

    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', output_dir))

    results_dir = os.path.join(output_dir, 'backtest_results')
    os.makedirs(results_dir, exist_ok=True)

    # Initialize services
    data_fetcher = DataFetcher()
    backtest_engine = BacktestEngine()

    # Fetch data
    logger.info(f"Fetching data for {len(symbols)} symbols...")
    try:
        data = data_fetcher.fetch_data(symbols, period=period, interval=interval)
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        sys.exit(1)

    all_results = {}

    # Run backtests for each symbol
    for symbol, df in data.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {symbol}")
        logger.info(f"{'='*60}")

        company_name = data_fetcher.get_company_name(symbol)
        logger.info(f"Company: {company_name}")

        symbol_results = {}

        # Test each strategy
        for strategy_config in strategies_config:
            strategy_name = strategy_config['name']
            params = strategy_config.get('params', {})

            logger.info(f"\nRunning backtest with strategy: {strategy_name}")

            strategy_key = strategy_name.lower()
            strategy = None
            if strategy_key in {"ma_crossover", "moving_average_crossover", "ma"}:
                short_length = params.get('short_length', params.get('short_window', 20))
                long_length = params.get('long_length', params.get('long_window', 50))
                strategy = MovingAverageCrossoverStrategy(
                    short_length=short_length,
                    long_length=long_length
                )
            elif strategy_key in {"rsi"}:
                rsi_length = params.get('rsi_length', params.get('rsi_period', 14))
                oversold = params.get('oversold', 30)
                overbought = params.get('overbought', 70)
                strategy = RSIStrategy(
                    rsi_length=rsi_length,
                    oversold=oversold,
                    overbought=overbought
                )
            else:
                logger.warning(f"Strategy '{strategy_name}' not implemented, skipping...")
                continue

            # Run backtest
            try:
                result = backtest_engine.run(
                    df=df,
                    strategy=strategy,
                    initial_capital=initial_capital,
                    commission=commission
                )
            except Exception as e:
                logger.error(f"Backtest failed for {symbol} with {strategy_name}: {e}")
                continue

            # Display results
            logger.info(f"\nStrategy: {strategy.name}")
            logger.info(f"Total Return: {result['total_return']:.2%}")
            logger.info(f"Sharpe Ratio: {result['sharpe_ratio']:.4f}")
            logger.info(f"Max Drawdown: {result['max_drawdown']:.2%}")
            if result.get('win_rate') is not None:
                logger.info(f"Win Rate: {result['win_rate']:.2%}")
            logger.info(f"Total Trades: {result.get('total_trades', 0)}")

            # Save visualization
            try:
                fig = backtest_engine.visualize_results(
                    result=result,
                    symbol=symbol,
                    strategy_name=strategy.name
                )

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                clean_symbol = symbol.replace('/', '-').replace('^', '')
                chart_path = os.path.join(
                    results_dir,
                    f"{clean_symbol}_{strategy.name}_{timestamp}.png"
                )

                fig.savefig(chart_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved backtest chart to: {chart_path}")

                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception as e:
                logger.error(f"Failed to save visualization: {e}")

            # Store results
            symbol_results[strategy.name] = {
                'performance': {
                    'total_return': float(result['total_return']),
                    'sharpe_ratio': float(result['sharpe_ratio']),
                    'max_drawdown': float(result['max_drawdown']),
                    'win_rate': float(result['win_rate']) if result.get('win_rate') is not None else None,
                    'total_trades': result.get('total_trades', 0)
                }
            }

        all_results[symbol] = symbol_results

    # Save summary JSON
    summary_path = os.path.join(
        results_dir,
        f"backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)

    logger.info(f"\nBacktest summary saved to: {summary_path}")
    logger.info(f"All results saved to: {results_dir}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Run backtests using the new service architecture.'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to configuration file (YAML or JSON)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Directory to save output files'
    )

    args = parser.parse_args()

    # Resolve config path if provided
    if args.config and not os.path.isabs(args.config):
        args.config = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', args.config))

    run_backtest(config_path=args.config, output_dir=args.output_dir)


if __name__ == '__main__':
    main()

"""
バックテストを実行するためのサンプルスクリプト
"""

import os
import sys
import yaml
import json
import argparse
from datetime import datetime
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtesting.engine import BacktestEngine
from src.backtesting.strategy import MovingAverageCrossoverStrategy, RSIStrategy

def load_config(config_path):
    """
    YAMLまたはJSONからコンフィグを読み込む
    
    Args:
        config_path (str): コンフィグファイルのパス
        
    Returns:
        dict: 設定辞書
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    file_ext = os.path.splitext(config_path)[1].lower()
    
    if file_ext == '.yaml' or file_ext == '.yml':
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
    elif file_ext == '.json':
        with open(config_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
    else:
        raise ValueError(f"Unsupported configuration file format: {file_ext}")
    
    return config

def run_backtest(config_path=None, use_vectorbt=True, output_dir=None):
    """
    コンフィグに基づいてバックテストを実行する
    
    Args:
        config_path (str): コンフィグファイルのパス
        use_vectorbt (bool): データ取得にvectorbtを使用するかどうか
        output_dir (str): 結果の出力ディレクトリ
        
    Returns:
        dict: 各シンボルのバックテスト結果
    """
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
                    'short_window': 20,
                    'long_window': 50
                }
            },
            {
                'name': 'RSI',
                'params': {
                    'rsi_period': 14,
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
            'output_dir': 'output',
            'figsize': [12, 10],
            'show_plots': False
        }
    }
    
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
    
    symbols = config['data']['symbols']
    period = config['data']['period']
    interval = config['data']['interval']
    strategies_config = config['strategies']
    
    if output_dir is None:
        output_dir = config['visualization'].get('output_dir', 'output')
    
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', output_dir))
    
    os.makedirs(output_dir, exist_ok=True)
    
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    results_dir = os.path.join(output_dir, 'backtest_results')
    os.makedirs(results_dir, exist_ok=True)
    
    engine = BacktestEngine(use_vectorbt=use_vectorbt)
    
    logger.info(f"Fetching data for {len(symbols)} symbols...")
    data = engine.fetch_data(symbols, period=period, interval=interval)
    
    all_results = {}
    
    for symbol, df in data.items():
        logger.info(f"\nProcessing {symbol}...")
        
        company_name = engine.data_fetcher.get_company_name(symbol)
        logger.info(f"Company name: {company_name}")
        
        symbol_results = {}
        
        for strategy_config in strategies_config:
            strategy_name = strategy_config['name']
            params = strategy_config.get('params', {})
            
            logger.info(f"Running backtest with strategy '{strategy_name}'")
            
            if strategy_name == 'MA_Crossover':
                short_window = params.get('short_window', 20)
                long_window = params.get('long_window', 50)
                strategy = MovingAverageCrossoverStrategy(
                    short_window=short_window,
                    long_window=long_window,
                    name=f"MA_{short_window}_{long_window}"
                )
            elif strategy_name == 'RSI':
                rsi_period = params.get('rsi_period', 14)
                oversold = params.get('oversold', 30)
                overbought = params.get('overbought', 70)
                strategy = RSIStrategy(
                    rsi_period=rsi_period,
                    oversold=oversold,
                    overbought=overbought,
                    name=f"RSI_{rsi_period}_{oversold}_{overbought}"
                )
            else:
                logger.warning(f"Strategy '{strategy_name}' not implemented, skipping...")
                continue
            
            initial_capital = config['backtest'].get('initial_capital', 10000)
            commission = config['backtest'].get('commission', 0.001)
            
            result = engine.run_backtest(
                strategy=strategy,
                data=df,
                initial_capital=initial_capital,
                commission=commission
            )
            
            result_paths = engine.save_results(
                result=result,
                symbol=symbol,
                strategy_name=strategy.name,
                output_dir=results_dir
            )
            
            logger.info(f"Strategy: {strategy.name}")
            logger.info(f"Total Return: {result['total_return']:.2%}")
            logger.info(f"Sharpe Ratio: {result['sharpe_ratio']:.4f}")
            logger.info(f"Max Drawdown: {result['max_drawdown']:.2%}")
            if result['win_rate'] is not None:
                logger.info(f"Win Rate: {result['win_rate']:.2%}")
            logger.info(f"Results saved to: {result_paths['chart']}")
            
            symbol_results[strategy.name] = {
                'performance': {
                    'total_return': float(result['total_return']),
                    'sharpe_ratio': float(result['sharpe_ratio']),
                    'max_drawdown': float(result['max_drawdown']),
                    'win_rate': float(result['win_rate']) if result['win_rate'] is not None else None
                },
                'paths': result_paths
            }
        
        all_results[symbol] = symbol_results
    
    summary_path = os.path.join(results_dir, f"backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
    
    logger.info(f"\nBacktest summary saved to: {summary_path}")
    
    return all_results

def main():
    """
    コマンドライン引数を解析してバックテストを実行する
    """
    parser = argparse.ArgumentParser(description='Run backtests based on configuration.')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to the configuration file (YAML or JSON)')
    parser.add_argument('--use-vectorbt', '-v', action='store_true',
                        help='Use vectorbt for data fetching (default: yfinance)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Directory to save output files to')
    
    args = parser.parse_args()
    
    if args.config and not os.path.isabs(args.config):
        args.config = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', args.config))
    
    run_backtest(config_path=args.config, use_vectorbt=args.use_vectorbt, output_dir=args.output_dir)

if __name__ == '__main__':
    main()

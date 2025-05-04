"""
Script to run financial analysis based on configuration.
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

from src.data.data_fetcher import DataFetcher
from src.analysis.technical_indicators import TechnicalAnalysis
from src.visualization.visualizer import Visualizer

def load_config(config_path):
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        dict: Configuration dictionary.
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

def run_analysis(config_path, use_vectorbt=True, output_dir=None):
    """
    Run financial analysis based on the provided configuration.
    
    Args:
        config_path (str): Path to the configuration file.
        use_vectorbt (bool): Whether to use vectorbt for data fetching.
        output_dir (str): Directory to save output files to.
    """
    config = load_config(config_path)
    
    symbols = config['data']['symbols']
    period = config['data']['period']
    interval = config['data']['interval']
    indicators = config['indicators']
    
    if output_dir is None:
        output_dir = config['visualization'].get('output_dir', 'output')
    
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', output_dir))
    
    os.makedirs(output_dir, exist_ok=True)
    
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    data_fetcher = DataFetcher(use_vectorbt=use_vectorbt)
    technical_analyzer = TechnicalAnalysis()
    
    viz_config = config.get('visualization', {})
    style = viz_config.get('style', 'seaborn')
    figsize = viz_config.get('figsize', [12, 8])
    dpi = viz_config.get('dpi', 300)
    show_plots = viz_config.get('show_plots', False)
    
    visualizer = Visualizer(style=style, figsize=tuple(figsize), dpi=dpi)
    
    logger.info(f"Fetching data for {len(symbols)} symbols...")
    data = data_fetcher.fetch_data(
        symbols, 
        period=period, 
        interval=interval,
        output_dir=data_dir,
        use_cache=True,
        cache_max_age=30
    )
    
    data_fetcher.save_data(data, data_dir)
    
    failed_symbols = data_fetcher.get_failed_symbols()
    if failed_symbols:
        logger.warning(f"Failed to fetch data for {len(failed_symbols)} symbols: {failed_symbols}")
        failed_report_path = visualizer.create_failed_symbols_report(failed_symbols, output_dir)
        logger.info(f"Created TODO list for failed symbols: {failed_report_path}")
    
    results = {}
    for symbol, df in data.items():
        logger.info(f"\nProcessing {symbol}...")
        
        company_name = data_fetcher.get_company_name(symbol)
        logger.info(f"Company name: {company_name}")
        
        logger.info(f"Calculating {len(indicators)} technical indicators...")
        df_with_indicators = technical_analyzer.calculate_indicators(df, indicators)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_symbol = symbol.replace('/', '-').replace('^', '')
        processed_filename = f"{clean_symbol}_processed_{timestamp}.csv"
        processed_filepath = os.path.join(data_dir, processed_filename)
        df_with_indicators.to_csv(processed_filepath)
        logger.info(f"Saved processed data to {processed_filepath}")
        
        logger.info(f"Creating visualization...")
        image_path = None
        
        fig = visualizer.create_plot_figure(
            df_with_indicators, 
            symbol, 
            indicators,
            company_name=company_name
        )

        if fig:
            if images_dir:
                image_path = visualizer.save_figure_to_file(fig, symbol, images_dir)
            elif show_plots:
                try:
                    plt.show()
                except Exception as e:
                    logger.warning(f"Could not display plot interactively for {symbol}: {e}")
                finally:
                    plt.close(fig)
            else:
                logger.info(f"Figure created for {symbol} but not saving or showing.")
                plt.close(fig)
        else:
            logger.warning(f"Skipped saving/showing plot for {symbol} as figure creation failed.")
        
        results[symbol] = {
            'raw_data': df.shape,
            'processed_data': df_with_indicators.shape,
            'image_path': image_path,
            'company_name': company_name
        }
    
    logger.info("\n=== Analysis Summary ===")
    for symbol, result in results.items():
        logger.info(f"{symbol}:")
        logger.info(f"  - Raw data shape: {result['raw_data']}")
        logger.info(f"  - Processed data shape: {result['processed_data']}")
        logger.info(f"  - Image saved to: {result['image_path']}")
    
    return results

def main():
    """
    Main function to parse command line arguments and run the analysis.
    """
    parser = argparse.ArgumentParser(description='Run financial analysis based on configuration.')
    parser.add_argument('--config', '-c', type=str, default='config/analysis_config.yaml',
                        help='Path to the configuration file (YAML or JSON)')
    parser.add_argument('--use-vectorbt', '-v', action='store_true',
                        help='Use vectorbt for data fetching (default: yfinance)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Directory to save output files to')
    
    args = parser.parse_args()
    
    if not os.path.isabs(args.config):
        args.config = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', args.config))
    
    run_analysis(args.config, use_vectorbt=args.use_vectorbt, output_dir=args.output_dir)

if __name__ == '__main__':
    main()

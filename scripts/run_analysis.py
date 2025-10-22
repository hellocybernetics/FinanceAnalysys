"""
CLI script for running technical analysis using the new service architecture.
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

from src.services.technical_service import TechnicalAnalysisService
from src.core.models import TechnicalAnalysisRequest, IndicatorConfig
from src.visualization.visualizer import Visualizer
from src.visualization.export_handler import ExportHandler

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


def run_analysis(config_path: str, output_dir: Optional[str] = None):
    """
    Run technical analysis using service layer.

    Args:
        config_path: Path to configuration file
        output_dir: Output directory for results
    """
    # Load configuration
    config = load_config(config_path)

    symbols = config['data']['symbols']
    period = config['data'].get('period', '1y')
    interval = config['data'].get('interval', '1d')
    indicator_configs_raw = config.get('indicators', [])

    # Setup output directory
    if output_dir is None:
        output_dir = config.get('visualization', {}).get('output_dir', 'output')

    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', output_dir))

    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Convert indicator configs to Pydantic models
    indicator_configs = [
        IndicatorConfig(
            name=ind['name'],
            params=ind.get('params', {}),
            plot=ind.get('plot', True)
        )
        for ind in indicator_configs_raw
    ]

    # Create analysis request
    request = TechnicalAnalysisRequest(
        symbols=symbols,
        period=period,
        interval=interval,
        indicators=indicator_configs,
        use_cache=True,
        cache_max_age=30
    )

    # Initialize services
    logger.info(f"Analyzing {len(symbols)} symbols...")
    service = TechnicalAnalysisService()
    visualizer = Visualizer()
    export_handler = ExportHandler()

    # Run analysis
    try:
        results = service.analyze(request)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

    # Process results
    for result in results:
        logger.info(f"\n{'='*60}")
        logger.info(f"Symbol: {result.symbol} ({result.company_name})")
        logger.info(f"{'='*60}")

        # Display summary
        summary = result.summary
        logger.info(f"Latest Price: ${summary.latest_price:.2f}")
        logger.info(f"Change: ${summary.price_change:.2f} ({summary.price_change_pct:+.2f}%)")
        if summary.volume:
            logger.info(f"Volume: {summary.volume:,.0f}")

        # Display indicators
        if summary.indicators:
            logger.info("\nIndicators:")
            for name, value in summary.indicators.items():
                logger.info(f"  {name}: {value:.2f}")

        # Display signals
        if summary.signals:
            logger.info("\nSignals:")
            for name, signal in summary.signals.items():
                logger.info(f"  {name}: {signal}")

        # Save processed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_symbol = result.symbol.replace('/', '-').replace('^', '')
        csv_path = os.path.join(data_dir, f"{clean_symbol}_processed_{timestamp}.csv")
        result.data.to_csv(csv_path)
        logger.info(f"\nSaved processed data to: {csv_path}")

        # Create and save visualization
        try:
            fig = visualizer.create_plot_figure(
                df=result.data,
                symbol=result.symbol,
                indicators=indicator_configs_raw,
                company_name=result.company_name
            )

            if fig:
                # Try PNG export first
                try:
                    image_path = os.path.join(images_dir, f"{clean_symbol}_analysis_{timestamp}.png")
                    export_handler.export_chart(fig, format='png', output_path=image_path)
                    logger.info(f"Saved chart to: {image_path}")
                except Exception as e:
                    # Fallback to HTML
                    logger.warning(f"PNG export failed: {e}, saving as HTML")
                    html_path = os.path.join(images_dir, f"{clean_symbol}_analysis_{timestamp}.html")
                    export_handler.export_chart(fig, format='html', output_path=html_path)
                    logger.info(f"Saved interactive chart to: {html_path}")
        except Exception as e:
            logger.error(f"Visualization failed for {result.symbol}: {e}")

    # Report failed symbols
    failed_symbols = service.get_failed_symbols()
    if failed_symbols:
        logger.warning(f"\nFailed to fetch data for {len(failed_symbols)} symbols:")
        for symbol in failed_symbols:
            logger.warning(f"  - {symbol}")

    logger.info(f"\nAnalysis complete! Results saved to: {output_dir}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Run financial technical analysis using the new service architecture.'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/analysis_config.yaml',
        help='Path to configuration file (YAML or JSON)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Directory to save output files'
    )

    args = parser.parse_args()

    # Resolve config path
    if not os.path.isabs(args.config):
        args.config = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', args.config))

    run_analysis(args.config, output_dir=args.output_dir)


if __name__ == '__main__':
    main()

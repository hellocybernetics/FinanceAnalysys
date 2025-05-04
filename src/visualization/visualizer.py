"""
Visualization module for creating and saving plots of financial data and indicators.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class Visualizer:
    """
    Class for visualizing financial data and technical indicators.
    """
    
    def __init__(self, style='seaborn-v0_8', figsize=(12, 8), dpi=300):
        """
        Initialize the Visualizer.
        
        Args:
            style (str): Matplotlib style to use.
            figsize (tuple): Figure size (width, height) in inches.
            dpi (int): DPI for saved images.
        """
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        
        sns.set_theme()
        
        if style:
            try:
                plt.style.use(style)
            except Exception as e:
                logger.warning(f"Could not use style '{style}': {e}")
                logger.info("Using default style instead")
    
    def plot_price_with_indicators(self, df, symbol, indicators, company_name=None, output_dir=None, show_plots=False):
        """
        Plot price data with technical indicators.
        
        Args:
            df (pd.DataFrame): Dataframe containing price data and indicators.
            symbol (str): Symbol being plotted.
            indicators (list): List of indicator configurations.
            company_name (str): Company name to display in the title.
            output_dir (str): Directory to save the plots to.
            show_plots (bool): Whether to display the plots.
            
        Returns:
            str: Path to the saved image file, or None if not saved.
        """
        fig, ax1 = plt.subplots(figsize=self.figsize)
        
        ax1.plot(df.index, df['Close'], label='Close Price', color='black', linewidth=1.5)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        for indicator in indicators:
            name = indicator['name']
            params = indicator.get('params', {})
            plot = indicator.get('plot', True)
            
            if not plot:
                continue
            
            if name == 'SMA':
                length = params.get('length', 20)
                col_name = f'SMA_{length}'
                if col_name in df.columns:
                    ax1.plot(df.index, df[col_name], label=f'SMA ({length})', linewidth=1)
            
            elif name == 'EMA':
                length = params.get('length', 50)
                col_name = f'EMA_{length}'
                if col_name in df.columns:
                    ax1.plot(df.index, df[col_name], label=f'EMA ({length})', linewidth=1)
            
            elif name == 'BBands':
                length = params.get('length', 20)
                std = params.get('std', 2)
                upper_col = f'BBU_{length}_{std}'
                middle_col = f'BBM_{length}_{std}'
                lower_col = f'BBL_{length}_{std}'
                
                if all(col in df.columns for col in [upper_col, middle_col, lower_col]):
                    ax1.plot(df.index, df[upper_col], 'r--', label=f'Upper BB ({length}, {std})', alpha=0.7)
                    ax1.plot(df.index, df[middle_col], 'g--', label=f'Middle BB ({length})', alpha=0.7)
                    ax1.plot(df.index, df[lower_col], 'r--', label=f'Lower BB ({length}, {std})', alpha=0.7)
                    ax1.fill_between(df.index, df[upper_col], df[lower_col], alpha=0.1, color='gray')
        
        oscillator_indicators = ['RSI', 'MACD', 'Stochastic']
        has_oscillators = any(ind['name'] in oscillator_indicators for ind in indicators if ind.get('plot', True))
        
        if has_oscillators:
            fig.set_size_inches(self.figsize[0], self.figsize[1] * 1.5)  # Make the figure taller
            
            gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
            
            ax1.set_position(gs[0].get_position(fig))
            ax1.set_subplotspec(gs[0])
            
            ax2 = fig.add_subplot(gs[1])  # RSI
            ax3 = fig.add_subplot(gs[2])  # MACD
            
            oscillator_count = 0
            
            for indicator in indicators:
                name = indicator['name']
                params = indicator.get('params', {})
                plot = indicator.get('plot', True)
                
                if not plot:
                    continue
                
                if name == 'RSI':
                    length = params.get('length', 14)
                    col_name = f'RSI_{length}'
                    
                    if col_name in df.columns:
                        ax2.plot(df.index, df[col_name], label=f'RSI ({length})', color='purple')
                        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
                        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
                        ax2.fill_between(df.index, df[col_name], 70, where=(df[col_name] >= 70), color='r', alpha=0.3)
                        ax2.fill_between(df.index, df[col_name], 30, where=(df[col_name] <= 30), color='g', alpha=0.3)
                        ax2.set_ylabel('RSI')
                        ax2.set_ylim(0, 100)
                        ax2.grid(True, alpha=0.3)
                        oscillator_count += 1
                
                elif name == 'MACD':
                    fast = params.get('fast', 12)
                    slow = params.get('slow', 26)
                    signal = params.get('signal', 9)
                    
                    macd_col = f'MACD_{fast}_{slow}'
                    signal_col = f'MACD_Signal_{signal}'
                    hist_col = f'MACD_Hist_{fast}_{slow}_{signal}'
                    
                    if all(col in df.columns for col in [macd_col, signal_col, hist_col]):
                        ax3.plot(df.index, df[macd_col], label=f'MACD ({fast}, {slow})', color='blue')
                        ax3.plot(df.index, df[signal_col], label=f'Signal ({signal})', color='red')
                        
                        for i in range(len(df) - 1):
                            if df[hist_col].iloc[i] >= 0:
                                ax3.bar(df.index[i], df[hist_col].iloc[i], color='g', alpha=0.5, width=0.7)
                            else:
                                ax3.bar(df.index[i], df[hist_col].iloc[i], color='r', alpha=0.5, width=0.7)
                        
                        ax3.set_ylabel('MACD')
                        ax3.grid(True, alpha=0.3)
                        oscillator_count += 1
            
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper left')
            ax3.legend(loc='upper left')
        else:
            ax1.legend(loc='best')
        
        if company_name and company_name != symbol:
            title = f'{symbol} - {company_name} - Technical Indicators'
        else:
            title = f'{symbol} - Technical Indicators'
            
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_symbol = symbol.replace('/', '-').replace('^', '')
            filename = f"{clean_symbol}_analysis_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved plot for {symbol} to {filepath}")
            
            if not show_plots:
                plt.close(fig)
                return filepath
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return filepath if output_dir else None
        
    def create_failed_symbols_report(self, failed_symbols, output_dir):
        """
        Create a report of symbols that failed to fetch data.
        
        Args:
            failed_symbols (list): List of symbols that failed to fetch.
            output_dir (str): Directory to save the report to.
            
        Returns:
            str: Path to the saved report file, or None if not saved.
        """
        if not failed_symbols:
            logger.info("No failed symbols to report")
            return None
            
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"failed_symbols_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("# TODO: Fix data retrieval for the following symbols\n\n")
            for symbol in failed_symbols:
                f.write(f"- {symbol}\n")
        
        logger.info(f"Saved failed symbols report to {filepath}")
        return filepath

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

# Define indicator categories
OVERLAY_INDICATORS = {'SMA', 'EMA', 'BBands'}
SUBPLOT_INDICATORS = {'RSI', 'MACD', 'Stochastic', 'ADX', 'WILLR'}

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
        
        self.plot_dispatch = {
            'SMA': self._plot_sma,
            'EMA': self._plot_ema,
            'BBands': self._plot_bbands,
            'RSI': self._plot_rsi,
            'MACD': self._plot_macd,
            'Stochastic': self._plot_stochastic,
            'ADX': self._plot_adx,
            'WILLR': self._plot_willr,
        }
        
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
        # Filter indicators that should be plotted
        indicators_to_plot = [ind for ind in indicators if ind.get('plot', True)]
        if not indicators_to_plot and 'Close' not in df.columns:
             logger.warning(f"No indicators to plot and no 'Close' column for {symbol}. Skipping plot.")
             return None

        # Categorize indicators
        overlay_inds = [ind for ind in indicators_to_plot if ind['name'] in OVERLAY_INDICATORS]
        subplot_inds = [ind for ind in indicators_to_plot if ind['name'] in SUBPLOT_INDICATORS]
        num_subplots = len(subplot_inds)

        # --- Create Figure and Axes --- 
        fig = None
        ax1 = None
        subplot_axes = []

        if num_subplots == 0:
            fig, ax1 = plt.subplots(figsize=self.figsize)
        else:
            # Adjust figure height based on number of subplots
            fig_height = self.figsize[1] * (1 + 0.5 * num_subplots)
            fig = plt.figure(figsize=(self.figsize[0], fig_height))
            
            # Define height ratios (give more space to price plot)
            height_ratios = [3] + [1] * num_subplots 
            gs = fig.add_gridspec(num_subplots + 1, 1, height_ratios=height_ratios)
            
            # Create price axes and subplot axes list
            ax1 = fig.add_subplot(gs[0])
            subplot_axes = [fig.add_subplot(gs[i+1], sharex=ax1) for i in range(num_subplots)]
            
            # Hide x-axis labels for all but the bottom subplot
            all_axes = [ax1] + subplot_axes
            for ax in all_axes[:-1]:
                plt.setp(ax.get_xticklabels(), visible=False)
        
        # --- Plot Price Data --- 
        if 'Close' in df.columns:
             ax1.plot(df.index, df['Close'], label='Close Price', color='black', linewidth=1.5)
        else:
             logger.warning(f"'Close' column not found for {symbol}. Cannot plot price.")

        ax1.set_ylabel('Price')

        # --- Plot Overlay Indicators --- 
        for ind in overlay_inds:
            plot_func = self.plot_dispatch.get(ind['name'])
            if plot_func:
                try:
                    plot_func(ax1, df, ind.get('params', {}))
                except Exception as e:
                    logger.error(f"Error plotting overlay indicator {ind['name']}: {e}")
            else:
                 logger.warning(f"Plotting function for overlay indicator {ind['name']} not found.")


        # --- Plot Subplot Indicators --- 
        for ind, ax in zip(subplot_inds, subplot_axes):
            plot_func = self.plot_dispatch.get(ind['name'])
            if plot_func:
                try:
                    plot_func(ax, df, ind.get('params', {}))
                except Exception as e:
                    logger.error(f"Error plotting subplot indicator {ind['name']}: {e}")
            else:
                 logger.warning(f"Plotting function for subplot indicator {ind['name']} not found.")

        # --- Final Touches --- 
        all_axes = [ax1] + subplot_axes
        for ax in all_axes:
            ax.grid(True, alpha=0.3) # Add legend to each axes
            # Set date formatter for x-axis (only needed for the last one if shared)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
            # Move Y-axis ticks and label to the right
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            
            # Get existing handles and labels for legend
            handles, labels = ax.get_legend_handles_labels()
            # Filter out potential duplicate labels (like ADX threshold) before creating legend
            unique_labels = {}
            for handle, label in zip(handles, labels):
                if label not in unique_labels:
                    unique_labels[label] = handle
            ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper left') 

        # Set x-label only on the bottom-most axis
        all_axes[-1].set_xlabel('Date')
        plt.xticks(rotation=45, ha='right') # Apply rotation to the last axes' ticks
        
        # Set Title
        if company_name and company_name != symbol:
            title = f'{symbol} - {company_name} - Technical Analysis'
        else:
            title = f'{symbol} - Technical Analysis'
        plt.suptitle(title, fontsize=16)
        
        # Improve layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust rect to prevent title overlap

        # --- Save / Show --- 
        filepath = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_symbol = symbol.replace('/', '-').replace('^', '') # Clean symbol for filename
            filename = f"{clean_symbol}_analysis_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            try:
                plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Saved plot for {symbol} to {filepath}")
            except Exception as e:
                 logger.error(f"Failed to save plot to {filepath}: {e}")
                 filepath = None # Ensure filepath is None if save fails

            if not show_plots: # Close figure only if saving and not showing
                plt.close(fig)
                # return filepath # Return inside if block ? No, return outside

        if show_plots:
            plt.show()
        elif not output_dir: # Close if not saving and not showing
             plt.close(fig)
        elif output_dir and not show_plots: # Already closed if saved and not showing
            pass # Figure is already closed
             
        return filepath
        
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

    # --- Private Plotting Methods --- 

    def _plot_sma(self, ax, df, params):
        length = params.get('length', 20)
        col_name = f'SMA_{length}'
        if col_name in df.columns:
            ax.plot(df.index, df[col_name], label=f'SMA ({length})', linewidth=1)

    def _plot_ema(self, ax, df, params):
        length = params.get('length', 50)
        col_name = f'EMA_{length}'
        if col_name in df.columns:
            ax.plot(df.index, df[col_name], label=f'EMA ({length})', linewidth=1)

    def _plot_bbands(self, ax, df, params):
        length = params.get('length', 20)
        std = params.get('std', 2)
        upper_col = f'BBU_{length}_{std}'
        middle_col = f'BBM_{length}_{std}'
        lower_col = f'BBL_{length}_{std}'
        if all(col in df.columns for col in [upper_col, middle_col, lower_col]):
            ax.plot(df.index, df[upper_col], 'r--', label=f'Upper BB ({length}, {std})', alpha=0.7, linewidth=1)
            ax.plot(df.index, df[middle_col], 'g--', label=f'Middle BB ({length})', alpha=0.7, linewidth=1)
            ax.plot(df.index, df[lower_col], 'r--', label=f'Lower BB ({length}, {std})', alpha=0.7, linewidth=1)
            ax.fill_between(df.index, df[upper_col], df[lower_col], alpha=0.1, color='gray')

    def _plot_rsi(self, ax, df, params):
        length = params.get('length', 14)
        col_name = f'RSI_{length}'
        if col_name in df.columns:
            ax.plot(df.index, df[col_name], label=f'RSI ({length})', color='purple', linewidth=1)
            ax.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax.fill_between(df.index, df[col_name], 70, where=(df[col_name] >= 70), interpolate=True, color='r', alpha=0.3)
            ax.fill_between(df.index, df[col_name], 30, where=(df[col_name] <= 30), interpolate=True, color='g', alpha=0.3)
            ax.set_ylabel('RSI')
            ax.set_ylim(0, 100)

    def _plot_macd(self, ax, df, params):
        fast = params.get('fast', 12)
        slow = params.get('slow', 26)
        signal = params.get('signal', 9)
        macd_col = f'MACD_{fast}_{slow}'
        signal_col = f'MACD_Signal_{signal}'
        hist_col = f'MACD_Hist_{fast}_{slow}_{signal}'
        if all(col in df.columns for col in [macd_col, signal_col, hist_col]):
            ax.plot(df.index, df[macd_col], label=f'MACD ({fast}, {slow})', color='blue', linewidth=1)
            ax.plot(df.index, df[signal_col], label=f'Signal ({signal})', color='red', linewidth=1)
            # Plot histogram using bars
            colors = np.where(df[hist_col] >= 0, 'g', 'r')
            ax.bar(df.index, df[hist_col], label=f'Hist ({fast},{slow},{signal})', color=colors, alpha=0.5, width=0.7)
            ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
            ax.set_ylabel('MACD')
            # Remove bar label from legend if lines are present
            handles, labels = ax.get_legend_handles_labels()
            ax.legend([h for h, l in zip(handles, labels) if 'Hist' not in l], 
                      [l for l in labels if 'Hist' not in l])


    def _plot_stochastic(self, ax, df, params):
        k = params.get('k', 14)
        d = params.get('d', 3)
        k_col = f'STOCHk_{k}_{d}'
        d_col = f'STOCHd_{k}_{d}'
        if all(col in df.columns for col in [k_col, d_col]):
            ax.plot(df.index, df[k_col], label=f'%K ({k})', color='orange', linewidth=1)
            ax.plot(df.index, df[d_col], label=f'%D ({d})', color='cyan', linewidth=1)
            ax.axhline(y=80, color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=20, color='g', linestyle='--', alpha=0.5)
            ax.fill_between(df.index, df[k_col], 80, where=(df[k_col] >= 80), interpolate=True, color='r', alpha=0.3)
            ax.fill_between(df.index, df[k_col], 20, where=(df[k_col] <= 20), interpolate=True, color='g', alpha=0.3)
            ax.set_ylabel('Stochastic')
            ax.set_ylim(0, 100)

    def _plot_adx(self, ax, df, params):
        length = params.get('length', 14)
        col_name = f'ADX_{length}'
        if col_name in df.columns:
            ax.plot(df.index, df[col_name], label=f'ADX ({length})', color='brown', linewidth=1)
            ax.axhline(y=25, color='grey', linestyle='--', alpha=0.5, label='ADX Threshold (25)') # Common threshold
            ax.set_ylabel('ADX')
            ax.set_ylim(bottom=0) # ADX starts from 0

    def _plot_willr(self, ax, df, params):
        length = params.get('length', 14)
        col_name = f'WILLR_{length}'
        if col_name in df.columns:
            ax.plot(df.index, df[col_name], label=f'Williams %R ({length})', color='lime', linewidth=1)
            ax.axhline(y=-20, color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=-80, color='g', linestyle='--', alpha=0.5)
            ax.fill_between(df.index, df[col_name], -20, where=(df[col_name] >= -20), interpolate=True, color='r', alpha=0.3)
            ax.fill_between(df.index, df[col_name], -80, where=(df[col_name] <= -80), interpolate=True, color='g', alpha=0.3)
            ax.set_ylabel('Williams %R')
            ax.set_ylim(-100, 0)

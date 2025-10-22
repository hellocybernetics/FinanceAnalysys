"""
Visualization module for creating and saving plots of financial data and indicators.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

logger = logging.getLogger(__name__)

# Define indicator categories
OVERLAY_INDICATORS = {'SMA', 'EMA', 'BBands'}
SUBPLOT_INDICATORS = {'RSI', 'MACD', 'Stochastic', 'ADX', 'WILLR'}

class Visualizer:
    """
    Class for visualizing financial data and technical indicators using Plotly.
    """
    
    def __init__(self, style='seaborn', figsize=(12, 8), dpi=300):
        """
        Initialize the Visualizer.
        
        Args:
            style (str): Style parameter (kept for compatibility but not used with Plotly).
            figsize (tuple): Figure size (width, height) in inches.
            dpi (int): DPI parameter (kept for compatibility but not used with Plotly).
        """
        self.figsize = figsize
        # Style and DPI are not used with Plotly but kept for compatibility
        
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
    
    def create_plot_figure(self, df, symbol, indicators, company_name=None, use_candlestick=True):
        """
        Create a Plotly Figure object with price data and technical indicators.
        
        Args:
            df (pd.DataFrame): Dataframe containing price data and indicators.
            symbol (str): Symbol being plotted.
            indicators (list): List of indicator configurations.
            company_name (str): Company name to display in the title.
            use_candlestick (bool): Whether to use candlestick chart or line chart for price data.
            
        Returns:
            plotly.graph_objects.Figure: The generated Figure object, or None if plotting is skipped.
        """
        # Filter indicators that should be plotted
        indicators_to_plot = [ind for ind in indicators if ind.get('plot', True)]
        if not indicators_to_plot and 'Close' not in df.columns:
             logger.warning(f"No indicators to plot and no 'Close' column for {symbol}. Skipping plot creation.")
             return None

        # Categorize indicators
        overlay_inds = [ind for ind in indicators_to_plot if ind['name'] in OVERLAY_INDICATORS]
        subplot_inds = [ind for ind in indicators_to_plot if ind['name'] in SUBPLOT_INDICATORS]
        num_subplots = len(subplot_inds)

        # Calculate dynamic height based on number of subplots
        base_height = 400  # Base height for main price chart
        subplot_height = 200  # Height for each subplot
        total_height = base_height + (num_subplots * subplot_height)
        
        # Create subplots
        if num_subplots == 0:
            fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
        else:
            # Dynamic vertical spacing based on number of subplots
            vertical_spacing = max(0.02, 0.1 / (num_subplots + 1))
            fig = make_subplots(
                rows=num_subplots + 1,  # Only add one more row for the main price chart
                cols=1, 
                shared_xaxes=True, 
                vertical_spacing=vertical_spacing,
                row_heights=[base_height] + [subplot_height] * num_subplots
            )

        # Plot price data (candlestick or line) - only on the first subplot
        if 'Close' in df.columns:
            if use_candlestick and all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                # Add candlestick chart only to the first subplot
                fig.add_trace(
                    go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name='OHLC',
                        increasing_line_color='green',
                        decreasing_line_color='red'
                    ),
                    row=1, col=1
                )
            else:
                # Fallback to line chart
                fig.add_trace(
                    go.Scatter(
                        x=df.index, 
                        y=df['Close'], 
                        mode='lines', 
                        name='Close Price',
                        line=dict(color='black', width=2)
                    ), 
                    row=1, col=1
                )
        else:
            logger.warning(f"'Close' column not found for {symbol}. Cannot plot price.")

        # Plot overlay indicators on the main price chart (first subplot)
        for ind in overlay_inds:
            plot_func = self.plot_dispatch.get(ind['name'])
            if plot_func:
                try:
                    plot_func(fig, df, ind.get('params', {}), row=1, col=1)
                except Exception as e:
                    logger.error(f"Error plotting overlay indicator {ind['name']}: {e}")
            else:
                logger.warning(f"Plotting function for overlay indicator {ind['name']} not found.")

        # Plot subplot indicators - each on its own subplot
        for i, ind in enumerate(subplot_inds):
            plot_func = self.plot_dispatch.get(ind['name'])
            if plot_func:
                try:
                    # Pass the correct row number for each subplot indicator
                    # Row numbers start from 2 since row 1 is for the main price chart
                    plot_func(fig, df, ind.get('params', {}), row=i+2, col=1)
                except Exception as e:
                    logger.error(f"Error plotting subplot indicator {ind['name']}: {e}")
            else:
                logger.warning(f"Plotting function for subplot indicator {ind['name']} not found.")

        # Update layout
        title = 'Technical Analysis'

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            height=total_height,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            # Improve spacing
            margin=dict(l=50, r=50, t=80, b=50),
            # Disable range slider but keep pan functionality
            xaxis=dict(
                rangeslider=dict(visible=False),  # Disable range slider
                fixedrange=False  # Allow panning
            )
        )
        
        # Update y-axis labels for subplots
        for i in range(1, num_subplots + 2):
            if i == 1:
                fig.update_yaxes(title_text="Price", row=i, col=1)
                # Disable range slider but keep pan functionality for the main price chart
                fig.update_xaxes(rangeslider=dict(visible=False), fixedrange=False, row=i, col=1)
            else:
                # For subplot indicators, we'll set titles in their respective plotting functions
                # Disable range slider but keep pan functionality for subplot indicators
                fig.update_xaxes(rangeslider=dict(visible=False), fixedrange=False, row=i, col=1)

        return fig

    def save_figure_to_file(self, fig, symbol, output_dir):
        """
        Save a Plotly Figure object to a file.
        
        Args:
            fig (plotly.graph_objects.Figure): The Figure object to save.
            symbol (str): Symbol for naming the file.
            output_dir (str): Directory to save the plot to.
            
        Returns:
            str: Path to the saved image file, or None if saving fails or fig is None.
        """
        if fig is None:
            logger.warning("Figure object is None, cannot save to file.")
            return None
        if not output_dir:
            logger.warning("Output directory not provided, cannot save figure.")
            return None
            
        filepath = None
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_symbol = symbol.replace('/', '-').replace('^', '') # Clean symbol for filename
            filename = f"{clean_symbol}_analysis_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            
            fig.write_image(filepath)
            logger.info(f"Saved plot for {symbol} to {filepath}")
        except Exception as e:
             logger.error(f"Failed to save plot to {filepath}: {e}")
             filepath = None # Ensure filepath is None if save fails
             
        return filepath

    def save_figure_to_buffer(self, fig, format='png'):
        """
        Save a Plotly Figure object to an in-memory buffer.
        
        Args:
            fig (plotly.graph_objects.Figure): The Figure object to save.
            format (str): The image format (e.g., 'png', 'jpeg').
            
        Returns:
            io.BytesIO: Buffer containing the image data, or None if saving fails or fig is None.
        """
        import io
        if fig is None:
            logger.warning("Figure object is None, cannot save to buffer.")
            return None
            
        buf = io.BytesIO()
        try:
            if format == 'png':
                buf.write(fig.to_image(format='png'))
            elif format == 'jpeg':
                buf.write(fig.to_image(format='jpeg'))
            else:
                buf.write(fig.to_image(format=format))
            buf.seek(0)
            logger.info(f"Saved figure to in-memory buffer (format: {format})")
        except Exception as e:
            logger.error(f"Failed to save figure to buffer: {e}")
            buf = None # Ensure buf is None if save fails
            
        return buf

    # --- Private Plotting Methods --- 

    def _plot_sma(self, fig, df, params, row, col):
        length = params.get('length', 20)
        col_name = f'SMA_{length}'
        if col_name in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df[col_name], 
                    mode='lines', 
                    name=f'SMA ({length})',
                    line=dict(width=1)
                ), 
                row=row, col=col
            )

    def _plot_ema(self, fig, df, params, row, col):
        length = params.get('length', 50)
        col_name = f'EMA_{length}'
        if col_name in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df[col_name], 
                    mode='lines', 
                    name=f'EMA ({length})',
                    line=dict(width=1)
                ), 
                row=row, col=col
            )

    def _plot_bbands(self, fig, df, params, row, col):
        length = params.get('length', 20)
        std = params.get('std', 2)
        upper_col = f'BBU_{length}_{std}'
        middle_col = f'BBM_{length}_{std}'
        lower_col = f'BBL_{length}_{std}'
        if all(col in df.columns for col in [upper_col, middle_col, lower_col]):
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df[upper_col], 
                    mode='lines', 
                    name=f'Upper BB ({length}, {std})',
                    line=dict(color='red', dash='dash', width=1)
                ), 
                row=row, col=col
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df[middle_col], 
                    mode='lines', 
                    name=f'Middle BB ({length})',
                    line=dict(color='green', dash='dash', width=1)
                ), 
                row=row, col=col
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df[lower_col], 
                    mode='lines', 
                    name=f'Lower BB ({length}, {std})',
                    line=dict(color='red', dash='dash', width=1),
                    fill='tonexty',
                    fillcolor='rgba(128, 128, 128, 0.1)'
                ), 
                row=row, col=col
            )

    def _plot_rsi(self, fig, df, params, row, col):
        length = params.get('length', 14)
        col_name = f'RSI_{length}'
        if col_name in df.columns:
            # Only plot the RSI line, not the candlestick chart
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df[col_name], 
                    mode='lines', 
                    name=f'RSI ({length})',
                    line=dict(color='purple', width=1)
                ), 
                row=row, col=col
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=row, col=col)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=row, col=col)
            fig.update_yaxes(range=[0, 100], row=row, col=col)
            fig.update_yaxes(title_text="RSI", row=row, col=col)

    def _plot_macd(self, fig, df, params, row, col):
        fast = params.get('fast', 12)
        slow = params.get('slow', 26)
        signal = params.get('signal', 9)
        macd_col = f'MACD_{fast}_{slow}'
        signal_col = f'MACD_Signal_{signal}'        
        hist_col = f'MACD_Hist_{fast}_{slow}_{signal}'
        if all(col in df.columns for col in [macd_col, signal_col, hist_col]):
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df[macd_col], 
                    mode='lines', 
                    name=f'MACD ({fast}, {slow})',
                    line=dict(color='blue', width=1)
                ), 
                row=row, col=col
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df[signal_col], 
                    mode='lines', 
                    name=f'Signal ({signal})',
                    line=dict(color='red', width=1)
                ), 
                row=row, col=col
            )
            # Plot histogram using bar chart
            colors = ['green' if val >= 0 else 'red' for val in df[hist_col]]
            fig.add_trace(
                go.Bar(
                    x=df.index, 
                    y=df[hist_col], 
                    name=f'Hist ({fast},{slow},{signal})',
                    marker_color=colors,
                    opacity=0.5
                ), 
                row=row, col=col
            )
            fig.add_hline(y=0, line_dash="dash", line_color="grey", row=row, col=col)
            fig.update_yaxes(title_text="MACD", row=row, col=col)

    def _plot_stochastic(self, fig, df, params, row, col):
        k = params.get('k', 14)
        d = params.get('d', 3)
        k_col = f'STOCHk_{k}_{d}'
        d_col = f'STOCHd_{k}_{d}'
        if all(col in df.columns for col in [k_col, d_col]):
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df[k_col], 
                    mode='lines', 
                    name=f'%K ({k})',
                    line=dict(color='orange', width=1)
                ), 
                row=row, col=col
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df[d_col], 
                    mode='lines', 
                    name=f'%D ({d})',
                    line=dict(color='cyan', width=1)
                ), 
                row=row, col=col
            )
            fig.add_hline(y=80, line_dash="dash", line_color="red", row=row, col=col)
            fig.add_hline(y=20, line_dash="dash", line_color="green", row=row, col=col)
            fig.update_yaxes(range=[0, 100], row=row, col=col)
            fig.update_yaxes(title_text="Stochastic", row=row, col=col)

    def _plot_adx(self, fig, df, params, row, col):
        length = params.get('length', 14)
        col_name = f'ADX_{length}'
        if col_name in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df[col_name], 
                    mode='lines', 
                    name=f'ADX ({length})',
                    line=dict(color='brown', width=1)
                ), 
                row=row, col=col
            )
            fig.add_hline(y=25, line_dash="dash", line_color="grey", row=row, col=col, 
                         annotation_text="ADX Threshold (25)", annotation_position="top left")
            fig.update_yaxes(title_text="ADX", row=row, col=col)

    def _plot_willr(self, fig, df, params, row, col):
        length = params.get('length', 14)
        col_name = f'WILLR_{length}'
        if col_name in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, 
                    y=df[col_name], 
                    mode='lines', 
                    name=f'Williams %R ({length})',
                    line=dict(color='lime', width=1)
                ), 
                row=row, col=col
            )
            fig.add_hline(y=-20, line_dash="dash", line_color="red", row=row, col=col)
            fig.add_hline(y=-80, line_dash="dash", line_color="green", row=row, col=col)
            fig.update_yaxes(range=[-100, 0], row=row, col=col)
            fig.update_yaxes(title_text="Williams %R", row=row, col=col)

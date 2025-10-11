"""
Test script to verify candlestick chart functionality
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from src.visualization.visualizer import Visualizer

# Create sample data with OHLCV values
def create_sample_data():
    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate sample price data
    np.random.seed(42)  # For reproducible results
    prices = [100.0]  # Starting price as float
    
    # Generate realistic price movements
    for i in range(1, len(dates)):
        change = np.random.normal(0, 1)  # Random change
        new_price = prices[-1] * (1 + change/100)
        prices.append(new_price)
    
    # Create OHLC data
    data = []
    for i, date in enumerate(dates):
        base_price = prices[i]
        open_price = base_price * (1 + np.random.normal(0, 0.1)/100)
        high_price = base_price * (1 + abs(np.random.normal(0, 0.5))/100)
        low_price = base_price * (1 - abs(np.random.normal(0, 0.5))/100)
        close_price = base_price * (1 + np.random.normal(0, 0.2)/100)
        
        # Ensure high >= open, close and low <= open, close
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'Date': date,
            'Open': float(open_price),
            'High': float(high_price),
            'Low': float(low_price),
            'Close': float(close_price),
            'Volume': int(volume)
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df

def test_candlestick_chart():
    """Test the candlestick chart functionality"""
    print("Creating sample data...")
    df = create_sample_data()
    print(f"Created data with {len(df)} rows")
    print(df.head())
    
    print("\nTesting visualization with candlestick chart...")
    visualizer = Visualizer()
    
    # Test with candlestick chart and multiple indicators
    indicators = [
        {"name": "SMA", "params": {"length": 5}},
        {"name": "SMA", "params": {"length": 20}},
        {"name": "EMA", "params": {"length": 5}},
        {"name": "EMA", "params": {"length": 20}},
        {"name": "BBands", "params": {"length": 20, "std": 2}},
        {"name": "RSI", "params": {"length": 14}}
    ]
    
    fig = visualizer.create_plot_figure(
        df, 
        "TEST", 
        indicators, 
        company_name="Test Company",
        use_candlestick=True
    )
    
    if fig:
        print("Candlestick chart created successfully!")
        # Save to HTML for inspection
        fig.write_html("test_candlestick_chart.html")
        print("Chart saved to test_candlestick_chart.html")
        
        # Verify the structure of the chart
        print(f"\nChart structure:")
        # Convert fig.data to a list to get its length
        data_list = list(fig.data)
        print(f"Number of traces: {len(data_list)}")
        
        # Check that candlestick is in the first subplot
        candlestick_trace = None
        for trace in fig.data:
            if isinstance(trace, go.Candlestick):
                candlestick_trace = trace
                break
                
        if candlestick_trace:
            print("Candlestick trace found in the chart")
        else:
            print("ERROR: Candlestick trace not found in the chart")
    else:
        print("Failed to create candlestick chart")
    
    # Test with line chart for comparison
    print("\nTesting visualization with line chart...")
    fig_line = visualizer.create_plot_figure(
        df, 
        "TEST", 
        indicators, 
        company_name="Test Company",
        use_candlestick=False
    )
    
    if fig_line:
        print("Line chart created successfully!")
        # Save to HTML for inspection
        fig_line.write_html("test_line_chart.html")
        print("Chart saved to test_line_chart.html")
    else:
        print("Failed to create line chart")

if __name__ == "__main__":
    test_candlestick_chart()
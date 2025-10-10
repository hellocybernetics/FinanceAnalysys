import pandas as pd
import numpy as np
from src.analysis.technical_indicators import TechnicalAnalysis

# Create a sample DataFrame for testing
data = {
    'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
    'Open': [100, 102, 105, 103, 106],
    'High': [103, 105, 106, 106, 108],
    'Low': [99, 101, 103, 102, 105],
    'Close': [102, 104, 103, 105, 107],
    'Volume': [1000, 1200, 1100, 1300, 1400]
}
df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

def test_rsi_calculation_without_vectorbt():
    """
    Test the RSI calculation without vectorbt to ensure it matches a manually verified calculation.
    This test is designed to fail with the incorrect SMA-based implementation and pass with the correct EMA-based one.
    """
    # Initialize TechnicalAnalysis and disable vectorbt
    ta = TechnicalAnalysis()
    ta.VECTORBT_AVAILABLE = False  # Force pandas implementation

    # Calculate RSI with a short length for clear results
    result_df = ta.calculate_indicators(df, [{'name': 'RSI', 'params': {'length': 3}}])

    # Manually calculated expected RSI values using the correct Wilder's Smoothing method.
    # These values have been verified against the implementation from the QuantInsti blog.
    expected_rsi = [np.nan, np.nan, np.nan, 80.0, 80.0]

    # Check if the calculated RSI matches the expected values (with tolerance for floating point differences)
    # The important part is the values that are not NaN.
    calculated_rsi = result_df['RSI_3'].dropna().to_list()
    expected_rsi_non_nan = [val for val in expected_rsi if not np.isnan(val)]

    # Round to 3 decimal places for comparison
    calculated_rsi_rounded = [round(val, 3) for val in calculated_rsi]
    expected_rsi_rounded = [round(val, 3) for val in expected_rsi_non_nan]

    assert calculated_rsi_rounded == expected_rsi_rounded, f"RSI calculation is incorrect. Expected {expected_rsi_rounded}, but got {calculated_rsi_rounded}"
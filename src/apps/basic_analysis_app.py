# src/apps/basic_analysis_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Import necessary functions from other modules
# from src.core.data.yahoo.fetch_from_yahoo import fetch_stock_data # Removed
from src.analysis.technical.calc_technical import calculate_technical_indicators
from src.analysis.fundamental.calc_fundamanta import calculate_fundamental_indicators
from src.visualization.simple_plot import simple_price_plot, simple_technical_plot

# @st.cache_data # Removed cache decorator from here
def run_basic_analysis(df, fundamental_data, symbol): # Updated function signature
    """Calculates indicators, generates plots, and returns results based on provided data."""
    # st.write(f"Fetching data for {symbol}...") # Removed
    # df, fundamental_data = fetch_stock_data(symbol, period=period, interval=interval) # Removed

    # Data is now passed as arguments 'df' and 'fundamental_data'
    if df is None or df.empty:
        # This check might be redundant if already checked in streamlit_app.py, but good for safety
        st.error("Input DataFrame is empty or None.")
        return None, None, None

    st.write("Calculating technical indicators...")
    df_tech = calculate_technical_indicators(df.copy()) # Use copy to avoid modifying input df
    if df_tech is None:
        st.warning("Could not calculate technical indicators.")
        df_tech = df # Use original data if calculation fails

    st.write("Generating plots...")
    try:
        fig_price = simple_price_plot(df_tech, symbol)
        fig_tech = simple_technical_plot(df_tech, symbol)
    except Exception as e:
        st.error(f"Error generating plots: {e}")
        fig_price, fig_tech = None, None

    st.write("Processing fundamental data...")
    fundamental_dict = None
    if fundamental_data:
        try:
            # Pass the already fetched fundamental_data
            fundamental_dict = calculate_fundamental_indicators(fundamental_data, symbol)
            if not fundamental_dict:
                 st.warning(f"Could not calculate fundamental indicators for {symbol}.")
        except Exception as e:
            st.error(f"Error processing fundamental data: {e}")
            fundamental_dict = None
    # else:
        # st.info(f"No fundamental data available for {symbol}.") # Message handled in streamlit_app.py

    return fig_price, fig_tech, fundamental_dict 
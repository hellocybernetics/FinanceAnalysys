# Finance Analysis App - Key Features

## 1. Interactive Visualization with Plotly

We've replaced the static matplotlib charts with interactive Plotly charts that provide:

- **Zoom and Pan**: Users can zoom into specific time periods and pan across the chart
- **Hover Information**: Detailed information appears when hovering over data points
- **Dynamic Sizing**: Charts automatically adjust their size based on the number of subplots
- **Better Performance**: Plotly charts are more responsive than static images

## 2. Candlestick Charts

The app now supports candlestick charts as the base visualization:

- **OHLC Data**: Displays Open, High, Low, and Close prices in a traditional candlestick format
- **Color Coding**: Green candles for price increases, red candles for price decreases
- **Fallback Support**: Automatically falls back to line charts if OHLC data is not available

## 3. Unified Interface

We've merged the Single Symbol and Multi-Symbol tabs into a single, dynamic interface:

- **Adaptive Display**: The interface automatically adjusts based on the number of symbols entered
- **Simplified Workflow**: Users no longer need to choose between single or multi-symbol analysis
- **Consistent Experience**: Same interface for both single and multiple symbol analysis

## 4. Improved Layout and Spacing

The visualization has been enhanced to prevent cramped displays:

- **Dynamic Height Calculation**: Chart height adjusts based on the number of subplots
- **Proper Spacing**: Adequate vertical spacing between subplots
- **Responsive Design**: Charts scale properly on different screen sizes

## 5. GitHub Deployment Support

The app is ready for deployment on GitHub Pages and other platforms:

- **GitHub Actions Workflow**: Automated deployment to GitHub Pages
- **Docker Support**: Dockerfile for containerized deployment
- **Deployment Scripts**: PowerShell and shell scripts for easy deployment
- **Static Site Generation**: Support for exporting as static HTML

## 6. Performance Optimizations

Several performance improvements have been implemented:

- **Caching Mechanisms**: Data and indicator calculations are cached to reduce redundant processing
- **Efficient Data Handling**: Optimized data fetching and processing
- **Memory Management**: Better memory usage patterns

## Usage Examples

### Enabling Candlestick Charts

In the sidebar, users can toggle the "Use Candlestick Chart" option to switch between candlestick and line charts.

### Adjusting Chart Size

Users can adjust both the width and height of charts in the sidebar visualization settings.

### Real-time Updates

The app supports real-time data updates with configurable refresh intervals.

## Technical Implementation Details

### Visualization Module

The `src/visualization/visualizer.py` module has been updated to:

1. Support both candlestick and line chart visualization
2. Dynamically calculate chart dimensions based on content
3. Properly space subplots to prevent overlap
4. Maintain backward compatibility with existing APIs

### Main Application

The `technical_analysis_app.py` file has been updated to:

1. Provide UI controls for candlestick visualization
2. Pass the candlestick preference to the visualization module
3. Maintain a unified interface for all analysis types
4. Support GitHub deployment workflows

## Deployment Options

### GitHub Pages

1. Push code to GitHub repository
2. GitHub Actions will automatically build and deploy the static site
3. Access the app at `https://<username>.github.io/<repository>`

### Streamlit Community Cloud

1. Push code to GitHub repository
2. Connect repository to Streamlit Community Cloud
3. Deploy with default settings

### Docker

1. Build Docker image: `docker build -t finance-analysis-app .`
2. Run container: `docker run -p 8501:8501 finance-analysis-app`
3. Access at `http://localhost:8501`

### Local Deployment

1. Run deployment script: `./deploy.sh` or `.\deploy.ps1`
2. Run app: `uv run streamlit run technical_analysis_app.py`
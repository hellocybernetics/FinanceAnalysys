# Changelog

All notable changes to the Finance Analysis App will be documented in this file.

## [Unreleased]

### Added
- Candlestick chart support as the base visualization
- Dynamic chart sizing based on number of subplots
- Unified interface for single and multi-symbol analysis
- GitHub Actions workflow for automatic deployment to GitHub Pages
- Dockerfile for containerized deployment
- Deployment scripts for both Unix and Windows systems
- Interactive Plotly charts with zoom, pan, and hover capabilities

### Changed
- Replaced static matplotlib charts with interactive Plotly charts
- Merged Single Symbol and Multi-Symbol tabs into a unified interface
- Improved layout and spacing to prevent cramped displays
- Enhanced visualization module to support candlestick charts
- Updated README with deployment instructions
- Optimized data fetching and processing performance

### Fixed
- Cramped display issues when multiple plots are shown
- Performance issues with UI updates
- Layout problems with subplot spacing

## [1.0.0] - 2025-10-11

### Initial Release
- Basic technical analysis functionality
- Static matplotlib charts
- Separate tabs for single and multi-symbol analysis
- Basic data fetching and indicator calculation
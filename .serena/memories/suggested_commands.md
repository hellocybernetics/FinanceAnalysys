Setup: `uv sync` (creates .venv, installs deps).
Run analysis: `uv run python scripts/run_analysis.py [--config path/to/config.yaml]`.
Run backtest: `uv run python scripts/run_backtest.py [--config path/to/config.yaml] [--use-vectorbt]`.
Launch Streamlit UI: `uv run streamlit run technical_analysis_app.py` or Windows `pwsh -File scripts/run_streamlit_uv.ps1`.
Tests: `uv run pytest`.
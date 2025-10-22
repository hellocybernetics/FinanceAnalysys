import json
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from scripts import run_backtest as run_module


class _DummyDataFetcher:
    def __init__(self):
        self._company = "Dummy Corp"

    def fetch_data(self, symbols, period=None, interval=None, **kwargs):
        index = pd.date_range("2024-01-01", periods=5, freq="D")
        data = {}
        for symbol in symbols:
            df = pd.DataFrame({
                'Open': [100, 101, 102, 103, 104],
                'High': [101, 102, 103, 104, 105],
                'Low': [99, 100, 101, 102, 103],
                'Close': [100, 102, 101, 104, 105],
                'Volume': [1_000_000] * 5,
            }, index=index)
            data[symbol] = df

        return data

    def get_company_name(self, symbol):
        return f"{self._company} ({symbol})"


class _DummyBacktestEngine:
    def run(self, df, strategy, initial_capital, commission):
        return {
            'signals': df.assign(signal=0.0),
            'trades': pd.DataFrame(),
            'equity_curve': df['Close'],
            'total_return': 0.05,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.1,
            'win_rate': 0.6,
            'profit_factor': 1.8,
            'total_trades': 2,
        }

    def visualize_results(self, result, symbol, strategy_name):
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.plot(result['equity_curve'].index, result['equity_curve'].values)
        ax.set_title(f"{symbol} - {strategy_name}")
        return fig


def test_run_backtest_cli(monkeypatch, tmp_path):
    monkeypatch.setattr(run_module, 'DataFetcher', _DummyDataFetcher)
    monkeypatch.setattr(run_module, 'BacktestEngine', _DummyBacktestEngine)

    run_module.run_backtest(output_dir=str(tmp_path))

    results_dir = Path(tmp_path) / 'backtest_results'
    summary_files = list(results_dir.glob('backtest_summary_*.json'))
    assert summary_files, "Summary JSON was not created"

    with summary_files[0].open('r', encoding='utf-8') as f:
        summary = json.load(f)

    assert 'AAPL' in summary, "Default symbol missing from summary"
    assert summary['AAPL'], "Strategy results should not be empty"

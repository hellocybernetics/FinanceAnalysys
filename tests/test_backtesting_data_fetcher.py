import numpy as np
import pandas as pd
from types import SimpleNamespace

from src.data import data_fetcher


class _DummyYFData:
    """Lightweight stand-in for vectorbt.YFData."""

    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    @classmethod
    def download(cls, symbols, period=None, interval=None):
        return cls(cls._frame_template(symbols))

    @classmethod
    def _frame_template(cls, symbols):
        index = pd.date_range("2024-01-01", periods=3, freq="D")
        fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        columns = pd.MultiIndex.from_product(
            [fields, symbols], names=["Field", "Symbol"]
        )
        values = np.arange(len(index) * len(columns)).reshape(len(index), len(columns))
        return pd.DataFrame(values, index=index, columns=columns)

    def get(self):
        return self._frame


def test_fetch_data_uses_vectorbt_multi_symbol(monkeypatch):
    symbols = ["AAPL", "MSFT"]

    # Prepare vectorbt stub and shared template
    template = _DummyYFData._frame_template(symbols)
    monkeypatch.setattr(_DummyYFData, "_frame", template, raising=False)

    monkeypatch.setattr(data_fetcher, "VECTORBT_AVAILABLE", True, raising=False)
    monkeypatch.setattr(
        data_fetcher,
        "vbt",
        SimpleNamespace(YFData=_DummyYFData),
        raising=False,
    )

    # Guard against accidental yfinance fallback
    def _fail(*_, **__):  # pragma: no cover - safety net
        raise AssertionError("yfinance fallback should not trigger")

    monkeypatch.setattr(data_fetcher, "yf", SimpleNamespace(Ticker=_fail), raising=False)

    data_fetcher.cache.clear()

    fetcher = data_fetcher.DataFetcher(use_vectorbt=True)
    result = fetcher.fetch_data(symbols, use_cache=False)

    assert set(result.keys()) == set(symbols)
    assert fetcher.failed_symbols == []

    for symbol in symbols:
        frame = result[symbol]
        assert isinstance(frame, pd.DataFrame)
        assert not frame.empty
        assert "Close" in frame.columns
        # Ensure the index matches the template produced by vectorbt
        assert frame.index.equals(template.index)

"""Utility helpers for mapping analysis periods to allowed data intervals."""

from __future__ import annotations

from typing import Dict, List

# Based on Yahoo Finance API constraints. Shorter periods permit finer-grained intervals.
PERIOD_INTERVAL_OPTIONS: Dict[str, List[str]] = {
    "1d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d"],
    "5d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d"],
    "1mo": ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "1wk"],
    "3mo": ["5m", "15m", "30m", "60m", "90m", "1h", "1d", "1wk"],
    "6mo": ["15m", "30m", "60m", "90m", "1h", "1d", "1wk", "1mo"],
    "1y": ["30m", "60m", "90m", "1h", "1d", "1wk", "1mo"],
    "2y": ["60m", "90m", "1h", "1d", "1wk", "1mo"],
    "5y": ["1d", "1wk", "1mo"],
    "max": ["1d", "1wk", "1mo", "3mo"],
}


def get_intervals_for_period(period: str) -> List[str]:
    """Return the allowed interval list for the given period.

    Args:
        period: Yahoo Finance period string (e.g. ``"1d"``, ``"6mo"``).

    Returns:
        Ordered list of interval strings. Defaults to daily/weekly/monthly when period
        is unrecognised.
    """

    return PERIOD_INTERVAL_OPTIONS.get(period, ["1d", "1wk", "1mo"])

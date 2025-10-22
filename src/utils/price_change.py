"""
Price change calculation utilities.
株価変動率の計算ユーティリティ
"""

import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta


class PriceChangeCalculator:
    """株価変動率を計算するユーティリティクラス"""

    @staticmethod
    def get_available_periods(period: str, interval: str) -> List[Dict[str, str]]:
        """
        期間と間隔から利用可能な変動率期間を取得

        Args:
            period: データ期間 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y)
            interval: データ間隔 (1d, 1wk, 1mo)

        Returns:
            利用可能な変動率期間のリスト
            [{"value": "1d", "label": "前日比"}, ...]
        """
        # 期間を日数に変換
        period_days = PriceChangeCalculator._period_to_days(period)

        # 間隔を日数に変換
        interval_days = PriceChangeCalculator._interval_to_days(interval)

        # 利用可能な変動率期間
        available = []

        # 前日比 (interval 1日の場合のみ)
        if interval == "1d" and period_days >= 2:
            available.append({"value": "1d", "label": "前日比"})

        # 1週間前比
        if period_days >= 7 and interval_days <= 1:
            available.append({"value": "1w", "label": "1週間前比"})

        # 2週間前比
        if period_days >= 14 and interval_days <= 1:
            available.append({"value": "2w", "label": "2週間前比"})

        # 1ヶ月前比
        if period_days >= 30:
            available.append({"value": "1mo", "label": "1ヶ月前比"})

        # 3ヶ月前比
        if period_days >= 90:
            available.append({"value": "3mo", "label": "3ヶ月前比"})

        # 6ヶ月前比
        if period_days >= 180:
            available.append({"value": "6mo", "label": "6ヶ月前比"})

        # 1年前比
        if period_days >= 365:
            available.append({"value": "1y", "label": "1年前比"})

        # 期間開始時比 (常に利用可能)
        available.append({"value": "period", "label": "期間開始時比"})

        # 最低1つは返す
        if not available:
            available.append({"value": "period", "label": "期間開始時比"})

        return available

    @staticmethod
    def calculate_price_change(
        data: pd.DataFrame,
        change_period: str,
        interval: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        指定期間の価格変動を計算

        Args:
            data: 価格データ (Closeカラム必須)
            change_period: 変動期間 (1d, 1w, 1mo, etc.)
            interval: データ間隔 (1d, 1wk, 1mo)

        Returns:
            (変動額, 変動率) のタプル
        """
        if data is None or data.empty or 'Close' not in data.columns:
            return None, None

        # 最新価格
        latest_price = data['Close'].iloc[-1]

        # 比較対象価格を取得
        compare_price = PriceChangeCalculator._get_compare_price(
            data, change_period, interval
        )

        if compare_price is None:
            return None, None

        # 変動額と変動率を計算
        price_change = latest_price - compare_price
        price_change_pct = (price_change / compare_price) * 100

        return price_change, price_change_pct

    @staticmethod
    def _get_compare_price(
        data: pd.DataFrame,
        change_period: str,
        interval: str
    ) -> Optional[float]:
        """比較対象の価格を取得"""

        if change_period == "period":
            # 期間開始時の価格
            return data['Close'].iloc[0]

        # 必要なデータポイント数を計算
        lookback = PriceChangeCalculator._calculate_lookback(change_period, interval)

        if lookback is None or lookback >= len(data):
            # データが不足している場合は期間開始時
            return data['Close'].iloc[0]

        # lookback分前の価格
        return data['Close'].iloc[-(lookback + 1)]

    @staticmethod
    def _calculate_lookback(change_period: str, interval: str) -> Optional[int]:
        """
        変動期間と間隔から必要なデータポイント数を計算

        Args:
            change_period: 変動期間 (1d, 1w, 1mo, etc.)
            interval: データ間隔 (1d, 1wk, 1mo)

        Returns:
            必要なデータポイント数
        """
        # 変動期間を日数に変換
        period_days = PriceChangeCalculator._change_period_to_days(change_period)

        if period_days is None:
            return None

        # 間隔を日数に変換
        interval_days = PriceChangeCalculator._interval_to_days(interval)

        # 必要なポイント数 = 期間日数 / 間隔日数
        lookback = int(period_days / interval_days)

        return lookback

    @staticmethod
    def _period_to_days(period: str) -> int:
        """データ期間を日数に変換"""
        mapping = {
            "1d": 1,
            "5d": 5,
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
            "10y": 3650,
        }
        return mapping.get(period, 365)

    @staticmethod
    def _change_period_to_days(change_period: str) -> Optional[int]:
        """変動期間を日数に変換"""
        mapping = {
            "1d": 1,
            "1w": 7,
            "2w": 14,
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
        }
        return mapping.get(change_period)

    @staticmethod
    def _interval_to_days(interval: str) -> int:
        """データ間隔を日数に変換"""
        mapping = {
            "1m": 1/390,   # 1分 (取引時間390分/日)
            "5m": 5/390,
            "15m": 15/390,
            "30m": 30/390,
            "1h": 1/6.5,   # 1時間 (取引時間6.5時間/日)
            "1d": 1,
            "1wk": 7,
            "1mo": 30,
        }
        return mapping.get(interval, 1)

    @staticmethod
    def get_default_change_period(period: str, interval: str) -> str:
        """
        期間と間隔から推奨される変動期間を取得

        Args:
            period: データ期間
            interval: データ間隔

        Returns:
            推奨変動期間
        """
        available = PriceChangeCalculator.get_available_periods(period, interval)

        # 優先順位: 前日比 > 1週間前比 > 1ヶ月前比 > その他
        preferred_order = ["1d", "1w", "1mo", "3mo", "period"]

        for pref in preferred_order:
            for option in available:
                if option["value"] == pref:
                    return pref

        # フォールバック: 最初の選択肢
        return available[0]["value"] if available else "period"

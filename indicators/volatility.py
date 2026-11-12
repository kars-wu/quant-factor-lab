"""
波动率类技术指标：布林带、ATR
"""

import numpy as np
import pandas as pd


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """
    布林带。
    返回 DataFrame，列：[upper, middle, lower, bandwidth, pct_b]
    """
    middle = series.rolling(window=period, min_periods=1).mean()
    std = series.rolling(window=period, min_periods=1).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std

    bandwidth = (upper - lower) / middle.replace(0, np.nan)
    denom = (upper - lower).replace(0, np.nan)
    pct_b = (series - lower) / denom

    return pd.DataFrame({
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "bandwidth": bandwidth,
        "pct_b": pct_b,
    })


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    平均真实范围 ATR（Wilder 平滑）。
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False).mean()

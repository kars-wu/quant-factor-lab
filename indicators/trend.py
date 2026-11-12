"""
趋势类技术指标。
全部使用 pandas 原生实现，无 ta-lib C 库依赖。
"""

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """简单移动平均"""
    return series.rolling(window=period, min_periods=1).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    指数移动平均（EWM）。
    使用 adjust=False（固定权重），与大多数交易软件保持一致。
    """
    return series.ewm(span=period, adjust=False).mean()


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD 指标。
    返回 DataFrame，列：[macd, signal, histogram]
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    })


def dma(series: pd.Series, fast: int = 10, slow: int = 50) -> pd.DataFrame:
    """
    DMA 均线差（国内常用）。
    返回 DataFrame，列：[dif, ama]
    """
    dif = sma(series, fast) - sma(series, slow)
    ama = sma(dif, 10)
    return pd.DataFrame({"dif": dif, "ama": ama})


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    平均方向指数 ADX（Wilder 平滑）。

    衡量趋势强度（不区分方向）：
      ADX > 25  — 趋势明确
      ADX 20-25 — 弱趋势/过渡期
      ADX < 20  — 无趋势（震荡市）

    参数
    ----
    high, low, close : 价格序列
    period           : 计算周期（默认14）
    """
    # +DM / -DM
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
                        index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
                         index=high.index)

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Wilder 平滑
    atr_s = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_dm_s = plus_dm.ewm(alpha=1 / period, adjust=False).mean()
    minus_dm_s = minus_dm.ewm(alpha=1 / period, adjust=False).mean()

    # +DI / -DI
    plus_di = 100 * plus_dm_s / atr_s.replace(0, np.nan)
    minus_di = 100 * minus_dm_s / atr_s.replace(0, np.nan)

    # DX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

    # ADX = DX 的 Wilder 平滑
    return dx.ewm(alpha=1 / period, adjust=False).mean()

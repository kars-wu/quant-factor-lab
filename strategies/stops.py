"""
止损价格计算模块。

四种方法在信号触发时（买入前）计算止损价，止盈价由盈亏比自动推导：

  atr      - ATR动态止损：买入价 - N × ATR14
             自适应波动率，高波动股自动给更宽空间
  swing    - 摆动低点止损：过去 N 根K线最低价下方 buffer%
             止损放在有技术意义的支撑位下方
  ma       - 均线止损：信号日 MA(N) 下方 buffer%
             跌破均线即认定走势破坏
  trailing - 追踪止损：信号日近 N 日高点下方 trail_pct%
             随价格上涨追踪最高点，适合趋势策略"让利润奔跑"

用法：
    from strategies.stops import calc_stop_atr, calc_target

    stop  = calc_stop_atr(df, signal_date, n_atr=2.0)
    target = calc_target(buy_price, stop, rr_ratio=2.0)
"""

from __future__ import annotations

import pandas as pd

from indicators.volatility import atr as _calc_atr

# ------------------------------------------------------------------
# 止损价计算
# ------------------------------------------------------------------

def calc_stop_atr(
    df: pd.DataFrame,
    signal_date,
    n_atr: float = 2.0,
    period: int = 14,
) -> float:
    """
    ATR 动态止损。

    stop = close(signal_date) - n_atr × ATR(signal_date)

    参数
    ----
    n_atr  : ATR 倍数，越大止损越宽（默认 2.0）
    period : ATR 计算周期（默认 14）

    注意：止损价最低不超过买入价的 50%（防止极端数据污染）。
    """
    atr_series = _calc_atr(df['high'], df['low'], df['close'], period)
    atr_val = atr_series.get(signal_date)
    buy_price = float(df.loc[signal_date, 'close'])

    if atr_val is None or pd.isna(atr_val) or atr_val <= 0:
        # ATR 数据不足，回退到买入价的 5%
        return buy_price * 0.95

    stop = buy_price - n_atr * float(atr_val)
    return max(stop, buy_price * 0.5)


def calc_stop_swing_low(
    df: pd.DataFrame,
    signal_date,
    lookback: int = 5,
    buffer: float = 0.005,
) -> float:
    """
    摆动低点止损。

    stop = min(low[ signal_date-lookback+1 : signal_date ]) × (1 - buffer)

    含义：跌破近期支撑位（加缓冲避免被分钱波动震出）即出场。

    参数
    ----
    lookback : 向前看多少根 K 线（含当日，默认 5）
    buffer   : 支撑位下方缓冲比例（默认 0.5%）
    """
    loc = df.index.get_loc(signal_date)
    start = max(0, loc - lookback + 1)
    swing_low = float(df['low'].iloc[start: loc + 1].min())
    return swing_low * (1 - buffer)


def calc_stop_ma(
    df: pd.DataFrame,
    signal_date,
    ma_period: int = 20,
    buffer: float = 0.01,
) -> float:
    """
    均线止损。

    stop = MA(ma_period, signal_date) × (1 - buffer)

    含义：跌破均线即认定上升趋势破坏，出场。

    参数
    ----
    ma_period : 均线周期（默认 20 日）
    buffer    : 均线下方缓冲比例（默认 1%，防止贴线震出）
    """
    ma = df['close'].rolling(ma_period, min_periods=ma_period).mean()
    ma_val = ma.get(signal_date)
    buy_price = float(df.loc[signal_date, 'close'])

    if ma_val is None or pd.isna(ma_val):
        # 均线数据不足，回退到买入价的 5%
        return buy_price * 0.95

    return float(ma_val) * (1 - buffer)


def calc_trailing_stop(
    df: pd.DataFrame,
    signal_date,
    trail_pct: float = 0.08,
    lookback: int = 5,
) -> float:
    """
    追踪止损。

    stop = peak_close(lookback) × (1 - trail_pct)

    含义：从近期最高收盘价回撤 trail_pct% 即止损。
    适合趋势策略"让利润奔跑"——趋势好的时候止损随高点上移，
    只在真正转向时离场。

    注意：此函数在信号日计算初始止损价。持仓期间的动态追踪
    由 TradeSimulator._simulate_trades_multiday 逐日更新。
    当前实现是简化版：固定用信号日附近的高点。

    参数
    ----
    trail_pct : 从高点回撤的比例（默认 8%）
    lookback  : 回看多少根 K 线找高点（默认 5）
    """
    loc = df.index.get_loc(signal_date)
    start = max(0, loc - lookback + 1)
    peak_close = float(df['close'].iloc[start: loc + 1].max())
    return peak_close * (1 - trail_pct)


# ------------------------------------------------------------------
# 止盈价推算
# ------------------------------------------------------------------

def calc_target(buy_price: float, stop_price: float, rr_ratio: float) -> float:
    """
    由盈亏比推算止盈价。

    risk   = buy_price - stop_price
    target = buy_price + risk × rr_ratio

    参数
    ----
    rr_ratio : 盈亏比（reward / risk），建议 ≥ 1.5
               例：rr_ratio=2 → 目标盈利是风险的 2 倍
    """
    risk = buy_price - stop_price
    if risk <= 0:
        # 止损价 >= 买入价（异常），给一个保底止盈
        return buy_price * 1.05
    return buy_price + risk * rr_ratio


# ------------------------------------------------------------------
# 方法注册表（供 evaluate_with_stop 按名称调用）
# ------------------------------------------------------------------

STOP_METHODS: dict[str, callable] = {
    'atr':      calc_stop_atr,
    'swing':    calc_stop_swing_low,
    'ma':       calc_stop_ma,
    'trailing': calc_trailing_stop,
}

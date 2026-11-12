"""
信号打分函数。

━━━ 背景 ━━━

策略要回答"买什么"（资产选择）。
当前 PortfolioManager 按信号列表顺序处理，先来的先买。
同一天可能有几十个信号但资金只够买 5 只，需要按信号质量排序。

打分维度：
  - 换手率（活跃度）：越高越容易进出
  - RSI 超卖程度：RSI 越低反弹空间越大（超卖策略）/ RSI 越高动量越强（趋势策略）
  - 成交额（流动性）：越大滑点越小
  - 距 MA60 的位置：离均线越远越可能回归（均值回归策略）

用法：
    from strategies.scoring import score_by_liquidity, score_by_oversold

    # PortfolioManager 在处理同日信号时按分数排序
    candidates.sort(key=lambda t: scores.get(t.symbol, 0), reverse=True)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_by_liquidity(df: pd.DataFrame, signal_dates: list) -> dict:
    """
    按流动性打分：换手率 + 成交额。

    参数
    ----
    df             : 单只股票的行情 DataFrame
    signal_dates   : 信号日列表（pd.Timestamp 或 str）

    返回
    ----
    {signal_date: score} 字典，分数 0~1
    """
    scores = {}
    for date in signal_dates:
        ts = pd.Timestamp(date) if not isinstance(date, pd.Timestamp) else date
        if ts not in df.index:
            scores[date] = 0.0
            continue

        row = df.loc[ts]
        # 换手率排名（越高越好）
        tr = row.get('turnover_rate', 0)
        if pd.isna(tr) or tr <= 0:
            tr_score = 0
        else:
            tr_score = min(tr / 0.30, 1.0)  # 30%换手率满分

        # 成交额排名（越高越好）
        amt = row.get('amount', 0)
        if pd.isna(amt) or amt <= 0:
            amt_score = 0
        else:
            amt_score = min(amt / 1e9, 1.0)  # 10亿成交额满分

        scores[date] = tr_score * 0.5 + amt_score * 0.5

    return scores


def score_by_oversold(df: pd.DataFrame, signal_dates: list, rsi_period: int = 14) -> dict:
    """
    按超卖程度打分：RSI 越低分越高（适合超卖反弹策略）。

    返回 {signal_date: score}，分数 0~1
    """
    close = df['close']
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/rsi_period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/rsi_period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    scores = {}
    for date in signal_dates:
        ts = pd.Timestamp(date) if not isinstance(date, pd.Timestamp) else date
        if ts not in rsi.index or pd.isna(rsi.get(ts)):
            scores[date] = 0.0
        else:
            # RSI=0 → 1.0（极度超卖），RSI=50 → 0.5，RSI=100 → 0
            scores[date] = max(0, 1.0 - float(rsi.loc[ts]) / 100)

    return scores


def score_by_momentum(df: pd.DataFrame, signal_dates: list, lookback: int = 10) -> dict:
    """
    按动量强度打分：近期涨幅越大分越高（适合趋势/追涨策略）。

    返回 {signal_date: score}，分数 0~1
    """
    ret = df['close'].pct_change(lookback)
    # 归一化到 0~1：正收益越高越好，负收益给 0
    scores = {}
    for date in signal_dates:
        ts = pd.Timestamp(date) if not isinstance(date, pd.Timestamp) else date
        if ts not in ret.index or pd.isna(ret.get(ts)):
            scores[date] = 0.0
        else:
            scores[date] = min(max(float(ret.loc[ts]) / 0.30, 0), 1.0)  # 30%涨幅满分

    return scores


def score_combined(df: pd.DataFrame, signal_dates: list, strategy_type: str = 'momentum') -> dict:
    """
    综合打分：流动性 + 策略适配度。

    参数
    ----
    strategy_type : 'momentum'（趋势/追涨）| 'oversold'（超卖反弹）
    """
    liq = score_by_liquidity(df, signal_dates)
    if strategy_type == 'oversold':
        strat = score_by_oversold(df, signal_dates)
    else:
        strat = score_by_momentum(df, signal_dates)

    return {d: liq.get(d, 0) * 0.4 + strat.get(d, 0) * 0.6 for d in signal_dates}

"""绩效指标：口径固定、命名明确，避免用错指标糊掉结论。"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

TRADING_DAYS = 244


def _annualize(series: pd.Series, ann: int = TRADING_DAYS) -> Dict[str, float]:
    series = series.dropna()
    if series.empty:
        return {}
    total = float((1 + series).prod() - 1)
    n = len(series)
    growth = 1 + total
    # 本金亏光或为负时，年化没有意义，直接退化用累计收益代替，避免复数
    annual = float(growth ** (ann / n) - 1) if n > 0 and growth > 0 else total
    vol = float(series.std() * np.sqrt(ann)) if n > 1 else 0.0
    sharpe = float(series.mean() / series.std() * np.sqrt(ann)) if n > 2 and series.std() > 0 else 0.0
    nav = (1 + series).cumprod()
    mdd = float((nav / nav.cummax() - 1).min())
    calmar = float(annual / abs(mdd)) if mdd < 0 else 0.0
    return {
        'n_days': n,
        'total_return': round(total, 6),
        'annual_return': round(annual, 6),
        'annual_vol': round(vol, 6),
        'sharpe': round(sharpe, 4),
        'max_drawdown': round(mdd, 6),
        'calmar': round(calmar, 4),
        'win_rate': round(float((series > 0).mean()), 4),
    }


def stats(daily: pd.Series) -> Dict[str, float]:
    """绝对收益指标。``daily`` 必须是完整交易日序列（空仓日记 0）。"""
    return _annualize(daily)


def excess_stats(daily: pd.Series, benchmark: pd.Series) -> Dict[str, float]:
    """净主动收益指标。

    约定：``active = strategy_net_daily - benchmark_daily``（算术差），
    ``IR`` 用主动收益均值/标准差年化，不与 NAV 比值混用。
    """
    idx = daily.index.intersection(benchmark.index)
    if len(idx) < 30:
        return {}
    active = (daily.loc[idx] - benchmark.loc[idx]).dropna()
    out = _annualize(active)
    out['active_annual_return'] = out.pop('annual_return')
    out['active_vol'] = out.pop('annual_vol')
    out['active_total_return'] = out.pop('total_return')
    out['ir'] = out.pop('sharpe')
    out['excess_sharpe'] = out['ir']
    return out


def yearly_active(daily: pd.Series, benchmark: pd.Series) -> Dict[str, Dict[str, float]]:
    idx = daily.index.intersection(benchmark.index)
    if len(idx) == 0:
        return {}
    active = (daily.loc[idx] - benchmark.loc[idx]).dropna()
    out: Dict[str, Dict[str, float]] = {}
    for year, chunk in active.groupby(active.index.year):
        s = _annualize(chunk)
        out[str(year)] = {
            'active_annual_return': s.get('annual_return', 0.0),
            'excess_sharpe': s.get('sharpe', 0.0),
            'n_days': s.get('n_days', 0),
        }
    return out


def ic_series(score: pd.DataFrame, fwd_ret: pd.DataFrame, mask: pd.DataFrame) -> pd.Series:
    """逐日截面 Spearman IC（辅助说明机制，不替代净收益）。"""
    masked_score = score.where(mask)
    masked_ret = fwd_ret.where(mask)
    vals = {}
    for date in score.index:
        a = masked_score.loc[date].dropna()
        b = masked_ret.loc[date].dropna()
        common = a.index.intersection(b.index)
        if len(common) < 20:
            continue
        vals[date] = float(a.loc[common].corr(b.loc[common], method='spearman'))
    return pd.Series(vals).sort_index()


def summarize(result: Dict[str, float]) -> str:
    if not result:
        return '（无有效结果）'
    keys = ['n_days', 'total_return', 'annual_return', 'sharpe', 'max_drawdown',
            'active_annual_return', 'excess_sharpe', 'ir']
    return ' | '.join(f'{k}={result[k]}' for k in keys if k in result)

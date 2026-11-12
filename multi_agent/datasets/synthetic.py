"""合成数据视图：无需数据库即可跑通全流程、写测试。

数据生成方式（刻意简单，只为验证工程链路，不代表任何市场规律）：

* 每只股票有一个缓慢变化的隐状态 ``s``；
* 次日收益 = 隐状态的横截面排名项 + 噪声；
* 因此「过去动量/反转类」表达式会得到非零 IC，便于验证评估器确实在工作；
* 流动性、市值与可交易掩码独立生成。

真实研究必须换成 ``multi_agent.datasets.sqlite_view``（只读数据库快照）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from multi_agent.datasets.base import DataView


def make_synthetic_view(
    n_dates: int = 700,
    n_symbols: int = 120,
    seed: int = 7,
    signal_strength: float = 0.35,
    snapshot_id: Optional[str] = None,
) -> DataView:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2018-01-01', periods=n_dates)
    symbols = [f'S{i:03d}' for i in range(n_symbols)]

    # 隐状态：AR(1)，慢变；收益与之正相关
    state = np.zeros((n_dates, n_symbols))
    state[0] = rng.normal(size=n_symbols)
    for t in range(1, n_dates):
        state[t] = 0.97 * state[t - 1] + 0.24 * rng.normal(size=n_symbols)

    # 截面排名项 → 次日收益，加入市场共同因子与个体噪声
    rank = pd.DataFrame(state, index=dates).rank(axis=1, pct=True).values
    market = rng.normal(0, 0.008, size=(n_dates, 1))
    idio = rng.normal(0, 0.016, size=(n_dates, n_symbols))
    # 信号项刻意压得很小：合成数据的夏普没有现实含义，只用于验证链路
    ret = 0.0020 * signal_strength * (rank - 0.5) * 2 + market + idio
    ret[0] = 0.0

    close = pd.DataFrame(10 * np.exp(np.cumsum(ret, axis=0)), index=dates, columns=symbols)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.004, close.shape))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, close.shape)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, close.shape)))
    volume = pd.DataFrame(
        rng.lognormal(12.5, 0.6, size=close.shape), index=dates, columns=symbols
    )
    amount = volume * close

    mv = pd.DataFrame(
        np.exp(rng.normal(23.0, 1.0, size=n_symbols)).repeat(1, axis=0).reshape(1, -1)
        * np.exp(np.cumsum(ret, axis=0)) ** 0.5,
        index=dates, columns=symbols,
    )
    turnover = volume / (mv / close) * 100
    pb = pd.DataFrame(
        np.exp(rng.normal(0.6, 0.5, size=(1, n_symbols))).repeat(n_dates, axis=0),
        index=dates, columns=symbols,
    )
    pe = pd.DataFrame(
        np.exp(rng.normal(3.2, 0.6, size=(1, n_symbols))).repeat(n_dates, axis=0),
        index=dates, columns=symbols,
    )

    n_industries = 8
    industry_of = {sym: i % n_industries for i, sym in enumerate(symbols)}
    industry = pd.DataFrame(
        np.tile([industry_of[s] for s in symbols], (n_dates, 1)).astype(float),
        index=dates, columns=symbols,
    )

    mask = pd.DataFrame(True, index=dates, columns=symbols)
    mask.iloc[:30] = False                                   # 上市预热
    susp = rng.random(close.shape) < 0.01                     # 随机停牌
    mask = mask & ~pd.DataFrame(susp, index=dates, columns=symbols)

    panels = {
        'open': open_, 'high': high, 'low': low, 'close': close,
        'volume': volume, 'amount': amount, 'mv': mv,
        'turnover_rate': turnover, 'pb': pb, 'pe_ttm': pe,
        'industry_code': industry,
    }
    # 基准与策略同口径：open-to-open（date d 的收益 = open[d+1]/open[d]-1）
    bench = (open_.shift(-1) / open_ - 1).where(mask).mean(axis=1).dropna()
    sid = snapshot_id or f'synthetic-{n_symbols}x{n_dates}-s{seed}'
    return DataView(
        snapshot_id=sid,
        panels=panels,
        mask=mask,
        benchmark=bench,
        meta={'source': 'synthetic', 'seed': seed, 'signal_strength': signal_strength},
    )

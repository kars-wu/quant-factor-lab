"""示例策略：两个教科书级 demo，用来验证「信号 → T+1 开盘成交 → 净值」链路。

⚠️ 这两个策略只是工程演示，没有任何经过验证的 alpha。回测出来的夏普/收益
不足以说明它们可以实盘使用；想评估真实策略，请用自己的信号函数替换。
"""
from __future__ import annotations

import pandas as pd

from simulation.strategies import StrategySpec, register


def _wide_to_signal(wide: pd.DataFrame) -> pd.Series:
    """把 (date × ticker) 宽表转成引擎需要的 (date, ticker) 长序列。"""
    sig = wide.stack().dropna()
    sig.index = sig.index.set_names(['date', 'ticker'])
    return sig.sort_index()


def momentum_20d_signal(panel_df: pd.DataFrame) -> pd.Series:
    close_w = panel_df['close'].unstack('ticker')
    return _wide_to_signal(close_w.pct_change(20))


def reversal_5d_signal(panel_df: pd.DataFrame) -> pd.Series:
    close_w = panel_df['close'].unstack('ticker')
    return _wide_to_signal(-close_w.pct_change(5))


def register_examples() -> None:
    register(StrategySpec(
        strategy_id='demo_momentum_20d',
        name='示例：20 日动量',
        description='买 20 日涨幅最大端；工程演示，未验证 alpha。',
        signal_fn=momentum_20d_signal,
        top_k=10,
        direction='high',
        candidate_pool=50,
        max_positions=20,
        position_size=0.05,
        pool='hs300_like',
        initial_capital=1_000_000.0,
    ))
    register(StrategySpec(
        strategy_id='demo_reversal_5d',
        name='示例：5 日反转',
        description='买 5 日跌幅最大端；工程演示，未验证 alpha。',
        signal_fn=reversal_5d_signal,
        top_k=10,
        direction='low',
        candidate_pool=50,
        max_positions=20,
        position_size=0.05,
        pool='top_n_amount',
        initial_capital=1_000_000.0,
    ))

"""通用因子库（教科书定义）。

这里收录的是公开、无需调参的经典因子，用来跑通「因子构造 → IC → 分层 → 合成」
这条链路。它不是一套有效 alpha 清单：开源仓库刻意不包含任何经过挖掘、
带有样本内调优痕迹的因子表达式。

想接入自己的因子，有两种方式：

1. 进程内注册：

    from strategies.factor_zoo import register_factor
    register_factor('my_factor', lambda df: df['close'].pct_change(5))

2. 直接传给分析器：

    analyzer.add_factor('my_factor', my_fn)

约定：`fn(df)` 接收单只股票的日线 DataFrame（index=trade_date，
至少含 open/high/low/close/volume/amount），返回与 df 同 index 的 Series。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd

from indicators.oscillator import rsi as _rsi
from indicators.trend import adx as _adx
from indicators.volatility import bollinger_bands as _bb


@dataclass(frozen=True)
class FactorSpec:
    name: str
    fn: Callable[[pd.DataFrame], pd.Series]
    category: str
    description: str


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name].astype(float)
    return pd.Series(np.nan, index=df.index)


def _ret(df: pd.DataFrame) -> pd.Series:
    return df['close'].astype(float).pct_change()


# ─────────────────────────── 通用因子定义 ───────────────────────────

def momentum_20d(df: pd.DataFrame) -> pd.Series:
    return df['close'].pct_change(20)


def momentum_60d(df: pd.DataFrame) -> pd.Series:
    return df['close'].pct_change(60)


def reversal_5d(df: pd.DataFrame) -> pd.Series:
    return -df['close'].pct_change(5)


def volatility_20d(df: pd.DataFrame) -> pd.Series:
    return _ret(df).rolling(20, min_periods=10).std()


def turnover_20d_avg(df: pd.DataFrame) -> pd.Series:
    return _col(df, 'turnover_rate').rolling(20, min_periods=10).mean()


def volume_ratio_20d(df: pd.DataFrame) -> pd.Series:
    vol = df['volume'].astype(float)
    return vol / vol.rolling(20, min_periods=10).mean().replace(0, np.nan)


def amihud_20d(df: pd.DataFrame) -> pd.Series:
    illiq = _ret(df).abs() / df['amount'].astype(float).replace(0, np.nan)
    return illiq.rolling(20, min_periods=10).mean().replace([np.inf, -np.inf], np.nan)


def rsi_14(df: pd.DataFrame) -> pd.Series:
    return _rsi(df['close'], 14)


def bb_position_20(df: pd.DataFrame) -> pd.Series:
    return _bb(df['close'], 20)['pct_b']


def ma_bias_20(df: pd.DataFrame) -> pd.Series:
    ma = df['close'].rolling(20, min_periods=10).mean()
    return df['close'] / ma - 1


def adx_14(df: pd.DataFrame) -> pd.Series:
    return _adx(df['high'], df['low'], df['close'], 14)


def gap_1d(df: pd.DataFrame) -> pd.Series:
    """隔夜跳空：今开 / 昨收 - 1。"""
    return df['open'] / df['close'].shift(1) - 1


def intraday_ret_1d(df: pd.DataFrame) -> pd.Series:
    """日内收益：今收 / 今开 - 1。"""
    return df['close'] / df['open'].replace(0, np.nan) - 1


def clv(df: pd.DataFrame) -> pd.Series:
    """收盘位置（Close Location Value）：收盘在当日区间中的相对位置。"""
    denom = (df['high'] - df['low']).replace(0, np.nan)
    return ((2 * df['close'] - df['high'] - df['low']) / denom).replace([np.inf, -np.inf], np.nan)


def upper_shadow(df: pd.DataFrame) -> pd.Series:
    top = pd.concat([df['open'], df['close']], axis=1).max(axis=1)
    return (df['high'] - top) / df['close'].replace(0, np.nan)


def lower_shadow(df: pd.DataFrame) -> pd.Series:
    bottom = pd.concat([df['open'], df['close']], axis=1).min(axis=1)
    return (bottom - df['low']) / df['close'].replace(0, np.nan)


BUILTIN_FACTORS: List[FactorSpec] = [
    FactorSpec('momentum_20d', momentum_20d, 'momentum', '20 日收益率'),
    FactorSpec('momentum_60d', momentum_60d, 'momentum', '60 日收益率'),
    FactorSpec('reversal_5d', reversal_5d, 'reversal', '5 日收益率取负'),
    FactorSpec('volatility_20d', volatility_20d, 'risk', '20 日收益波动率'),
    FactorSpec('turnover_20d_avg', turnover_20d_avg, 'liquidity', '20 日均换手率'),
    FactorSpec('volume_ratio_20d', volume_ratio_20d, 'liquidity', '成交量 / 20 日均量'),
    FactorSpec('amihud_20d', amihud_20d, 'liquidity', 'Amihud 非流动性'),
    FactorSpec('rsi_14', rsi_14, 'momentum', 'RSI(14)'),
    FactorSpec('bb_position_20', bb_position_20, 'momentum', '布林带 %B(20)'),
    FactorSpec('ma_bias_20', ma_bias_20, 'momentum', '20 日均线乖离'),
    FactorSpec('adx_14', adx_14, 'trend', 'ADX(14) 趋势强度'),
    FactorSpec('gap_1d', gap_1d, 'microstructure', '隔夜跳空'),
    FactorSpec('intraday_ret_1d', intraday_ret_1d, 'microstructure', '日内收益'),
    FactorSpec('clv', clv, 'microstructure', '收盘位置'),
    FactorSpec('upper_shadow', upper_shadow, 'microstructure', '上影线强度'),
    FactorSpec('lower_shadow', lower_shadow, 'microstructure', '下影线强度'),
]

_CUSTOM: Dict[str, FactorSpec] = {}


def register_factor(name: str, fn: Callable[[pd.DataFrame], pd.Series],
                    category: str = 'custom', description: str = '') -> FactorSpec:
    """注册自定义因子；同名覆盖。"""
    spec = FactorSpec(name=name, fn=fn, category=category, description=description)
    _CUSTOM[name] = spec
    return spec


def iter_factors() -> Iterable[FactorSpec]:
    """遍历内置 + 自定义因子。"""
    yield from BUILTIN_FACTORS
    yield from _CUSTOM.values()


def get_factor(name: str) -> FactorSpec:
    for spec in iter_factors():
        if spec.name == name:
            return spec
    raise KeyError(f'unknown factor: {name}')


def describe() -> pd.DataFrame:
    """列出可用因子（便于在 notebook 里查看）。"""
    rows = [
        {'name': s.name, 'category': s.category, 'description': s.description,
         'source': 'builtin' if s.name not in _CUSTOM else 'custom'}
        for s in iter_factors()
    ]
    return pd.DataFrame(rows)

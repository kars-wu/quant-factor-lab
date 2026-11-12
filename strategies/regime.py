"""
市场状态识别（Regime Detection）。

━━━ 背景 ━━━

识别市场状态往往比反复换策略更重要。
当前只有"指数>MA60=牛市"的二元判断，这里扩展为三状态+过渡期。

━━━ 三种基本状态 ━━━

| 状态     | 特征                          | 最佳策略          |
|----------|-------------------------------|-------------------|
| trend   | ADX>25, 收益率方向明确          | 动量/趋势跟随      |
| range   | ADX<20, 波动率低               | 均值回归/网格      |
| crisis  | 波动率>30%(年化), 相关性飙升    | 风控优先/现金为王   |
| transition | 不满足以上任何明确条件        | 观望或概率加权     |

━━━ 用法 ━━━

    from strategies.regime import RegimeDetector

    detector = RegimeDetector()
    regime = detector.detect(index_df)
    # 返回 pd.Series，每日值: 'trend'|'range'|'crisis'|'transition'

    # 在 Screener 中使用
    from strategies.screener import RegimeFilter
    screener = Screener([RegimeFilter(regime_series=regime, allow=['trend'])])
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from indicators.trend import adx


class RegimeDetector:
    """规则法市场状态识别（最实用，无ML依赖）。"""

    def __init__(
        self,
        adx_period: int = 14,
        adx_trend_threshold: float = 25.0,
        adx_range_threshold: float = 20.0,
        vol_lookback: int = 20,
        vol_crisis_threshold: float = 0.30,     # 年化波动率30%
        return_lookback: int = 20,
        return_trend_threshold: float = 0.05,   # 20日收益率5%
    ):
        self._adx_period = adx_period
        self._adx_trend = adx_trend_threshold
        self._adx_range = adx_range_threshold
        self._vol_lookback = vol_lookback
        self._vol_crisis = vol_crisis_threshold
        self._return_lookback = return_lookback
        self._return_trend = return_trend_threshold

    def detect(self, index_df: pd.DataFrame) -> pd.Series:
        """
        对指数 DataFrame 识别每日市场状态。

        参数
        ----
        index_df : 含 open/high/low/close 列，index=trade_date

        返回
        ----
        pd.Series，每日值: 'trend'|'range'|'crisis'|'transition'
        """
        close = index_df['close']

        # ADX（趋势强度）
        adx_series = adx(index_df['high'], index_df['low'], close, self._adx_period)

        # 波动率（年化）
        daily_ret = close.pct_change()
        vol = daily_ret.rolling(self._vol_lookback, min_periods=10).std() * np.sqrt(252)

        # 收益率
        ret = close.pct_change(self._return_lookback)

        # 逐日判断
        regime = pd.Series('transition', index=index_df.index)

        # 危机市优先判断（波动率极高）
        is_crisis = vol > self._vol_crisis

        # 趋势市：ADX 高 + 收益率方向明确
        is_trend = (adx_series > self._adx_trend) & (ret.abs() > self._return_trend)

        # 震荡市：ADX 低 + 波动率低
        is_range = (adx_series < self._adx_range) & (vol < self._vol_crisis * 0.5)

        regime[is_crisis] = 'crisis'
        regime[~is_crisis & is_trend] = 'trend'
        regime[~is_crisis & ~is_trend & is_range] = 'range'

        return regime

    def detect_from_db(
        self,
        db,
        index_symbol: str = '000300',
        start_date: str = None,
        end_date: str = None,
    ) -> pd.Series:
        """从数据库加载指数数据并识别状态。"""
        query = "SELECT trade_date, open, high, low, close FROM index_daily WHERE symbol=?"
        params = [index_symbol]
        if start_date:
            query += " AND trade_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND trade_date <= ?"
            params.append(end_date)
        query += " ORDER BY trade_date"

        rows = db.query(query, tuple(params))
        if not rows:
            logger.warning("无指数数据: {}", index_symbol)
            return pd.Series(dtype=str)

        df = pd.DataFrame(rows, columns=['trade_date', 'open', 'high', 'low', 'close'])
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index('trade_date').astype(float)
        return self.detect(df)


# ------------------------------------------------------------------
# 缓存（避免重复读库和计算）
# ------------------------------------------------------------------

_regime_cache: dict[tuple, pd.Series] = {}


def get_regime(
    db,
    index_symbol: str = '000300',
    adx_period: int = 14,
    force_refresh: bool = False,
) -> pd.Series:
    """
    获取市场状态序列（带缓存）。

    返回 pd.Series，index=trade_date（Timestamp），值为状态字符串。
    """
    cache_key = (index_symbol, adx_period)
    if cache_key in _regime_cache and not force_refresh:
        return _regime_cache[cache_key]

    detector = RegimeDetector(adx_period=adx_period)
    regime = detector.detect_from_db(db, index_symbol)
    _regime_cache[cache_key] = regime
    return regime

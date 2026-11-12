"""
复权因子计算模块。

核心公式：
    后复权：hfq_price(t) = raw_price(t) × hfq_factor(t)
            hfq_factor 上市首日 = 1.0，随除权事件单调不减

    前复权：qfq_price(t) = raw_price(t) × hfq_factor(t) / hfq_factor(anchor_date)
            anchor_date = 回测结束日（保证整个回测期价格连续且可复现）

注意：volume 不做任何复权处理。
"""

import pandas as pd
import numpy as np
from loguru import logger


def _ensure_datetime_index(index_like) -> pd.DatetimeIndex:
    """避免对已经是 DatetimeIndex 的对象重复做高成本转换。"""
    if isinstance(index_like, pd.DatetimeIndex):
        return index_like
    return pd.to_datetime(index_like)


def apply_qfq(
    df: pd.DataFrame,
    hfq_factors: pd.Series,
    anchor_factor: float,
) -> pd.DataFrame:
    """
    对价格 DataFrame 应用前复权。

    参数
    ----
    df : DataFrame
        必须包含 open/high/low/close 列，index 为 trade_date（字符串）
    hfq_factors : Series
        index 为 trade_date，values 为 hfq_factor（后复权因子）
    anchor_factor : float
        归一化锚点因子，通常取回测结束日对应的 hfq_factor。
        固定此值保证同一回测区间的价格序列不随未来除权变化。

    返回
    ----
    DataFrame，price 列已前复权，volume 列保持原值不变。
    """
    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]

    # 对齐因子到价格 DataFrame 的 index。
    # 兼容 df.index / hfq_factors.index 为字符串或 Timestamp 两种情况。
    factor_series = pd.Series(hfq_factors.values, index=_ensure_datetime_index(hfq_factors.index)).sort_index()
    target_index = _ensure_datetime_index(pd.Index(df.index))
    factor_aligned = factor_series.reindex(target_index)
    factor_aligned.index = df.index

    # 若某些日期缺少因子（数据不完整），用前向填充
    missing = factor_aligned.isna().sum()
    if missing > 0:
        logger.debug("{}个交易日缺少复权因子，使用前向/后向填充补全", missing)
        factor_aligned = factor_aligned.ffill().bfill()

    if anchor_factor == 0 or pd.isna(anchor_factor):
        raise ValueError(f"anchor_factor 无效: {anchor_factor}")

    ratio = factor_aligned / anchor_factor

    result = df.copy()
    for col in price_cols:
        result[col] = (df[col] * ratio).round(4)

    return result


def apply_hfq(
    df: pd.DataFrame,
    hfq_factors: pd.Series,
) -> pd.DataFrame:
    """
    对价格 DataFrame 应用后复权（供参考用，回测一般用前复权）。
    """
    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    factor_series = pd.Series(hfq_factors.values, index=_ensure_datetime_index(hfq_factors.index)).sort_index()
    target_index = _ensure_datetime_index(pd.Index(df.index))
    factor_aligned = factor_series.reindex(target_index).ffill().bfill()
    factor_aligned.index = df.index

    result = df.copy()
    for col in price_cols:
        result[col] = (df[col] * factor_aligned).round(4)

    return result


def get_anchor_factor(
    hfq_factors: pd.Series,
    anchor_date: str,
) -> float:
    """
    获取锚点日期（回测结束日）对应的 hfq_factor。
    若该日无因子数据，向前取最近一个有效因子。

    参数
    ----
    hfq_factors : Series，index 为 trade_date（升序排列）
    anchor_date : str，格式 'YYYY-MM-DD'
    """
    factor_series = pd.Series(hfq_factors.values, index=_ensure_datetime_index(hfq_factors.index)).sort_index()
    anchor_ts = pd.Timestamp(anchor_date)

    # 取 <= anchor_date 的最新因子
    valid = factor_series[factor_series.index <= anchor_ts]
    if valid.empty:
        raise ValueError(f"找不到 {anchor_date} 之前的复权因子")
    anchor = float(valid.iloc[-1])
    if anchor <= 0 or np.isnan(anchor):
        raise ValueError(f"锚点因子无效: {anchor}")
    return anchor


def verify_qfq(
    symbol: str,
    test_date: str,
    raw_close: float,
    hfq_factor: float,
    anchor_factor: float,
    expected_qfq: float,
    tolerance: float = 0.02,
) -> bool:
    """
    验证前复权价格计算是否正确。

    用法示例（在初始化后对比 AkShare 直接返回的 qfq 价格）：
        import akshare as ak
        df_qfq = ak.stock_zh_a_hist(symbol='510300', adjust='qfq', ...)
        expected = df_qfq.loc[df_qfq['日期']==test_date, '收盘'].iloc[0]
        verify_qfq('510300', test_date, raw_close, hfq_factor, anchor_factor, expected)

    参数
    ----
    tolerance : 允许误差（元），主要来自浮点精度，通常 <0.02 元
    """
    computed = round(raw_close * hfq_factor / anchor_factor, 4)
    diff = abs(computed - expected_qfq)

    if diff <= tolerance:
        logger.info(
            "复权验证通过 [{}] {}: 计算={:.4f}, 期望={:.4f}, 差={:.6f}",
            symbol, test_date, computed, expected_qfq, diff
        )
        return True
    else:
        logger.error(
            "复权验证失败 [{}] {}: 计算={:.4f}, 期望={:.4f}, 差={:.6f} > {}",
            symbol, test_date, computed, expected_qfq, diff, tolerance
        )
        return False

"""
选股器（Screener）模块。

将"股票过滤"与"入场信号"解耦：
  - Screener：判断每只股票每天是否"允许入场"（返回 bool Series）
  - Signal  ：在允许入场的股票中，找具体的买入时机

用法：
    from strategies.screener import Screener, MarketCapFilter, TurnoverFilter, SCREENER_PRESETS

    # 自定义选股器
    sc = Screener([
        MarketCapFilter(max_yi=200),
        TurnoverFilter(min_pct=0.10, max_pct=0.30),
    ])

    # 或使用预设
    sc = SCREENER_PRESETS["小市值活跃股"]

    # 应用到单只股票 DataFrame
    eligible = sc.apply(df)   # bool Series，True=该日允许入场

设计原则：
  - 每个 Filter 独立、无副作用，只读取 df 不修改
  - 数据不足时（NaN、outstanding_share 缺失）返回 False（严格模式）
  - Screener 串联所有 Filter（AND 逻辑）
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

# ------------------------------------------------------------------
# 抽象基类
# ------------------------------------------------------------------

class BaseFilter(ABC):
    """
    选股过滤器基类。

    子类实现 apply()，接受单只股票的 DataFrame（index=trade_date），
    返回 bool Series（True = 该日满足过滤条件，允许入场）。
    """

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.Series:
        """
        参数
        ----
        df : 单只股票的历史行情，index=trade_date，
             列包含 open, high, low, close, volume, amount,
             turnover_rate（可 NaN），outstanding_share（可 NaN）

        返回
        ----
        pd.Series(dtype=bool)，index 与 df 相同
        """

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"


# ------------------------------------------------------------------
# 具体过滤器
# ------------------------------------------------------------------

class MarketCapFilter(BaseFilter):
    """
    市值区间过滤（亿元）。

    计算方式：total_mv = close × outstanding_share / 1e8
    需要 DataFrame 含 outstanding_share 列（由 DataLoader / Evaluator 注入）。
    若 outstanding_share 缺失，整列返回 False（严格模式）。
    """

    def __init__(self, max_yi: float = 200.0, min_yi: float = 0.0):
        self.max_yi = max_yi
        self.min_yi = min_yi

    def apply(self, df: pd.DataFrame) -> pd.Series:
        false_series = pd.Series(False, index=df.index)

        if "outstanding_share" not in df.columns:
            return false_series

        outstanding = df["outstanding_share"]
        if outstanding.isna().all():
            return false_series

        mv = df["close"] * outstanding / 1e8
        return ((mv >= self.min_yi) & (mv < self.max_yi)).fillna(False)


class TurnoverFilter(BaseFilter):
    """
    换手率区间过滤（小数形式，0.10 = 10%）。

    数据来源优先级：
      1. df["turnover_rate"] 列（已存入数据库，小数形式）
      2. 若列缺失或全 NaN，则用 volume / outstanding_share 估算
      3. 若 outstanding_share 也缺失，返回 False
    """

    def __init__(self, min_pct: float = 0.05, max_pct: float = 0.30):
        self.min_pct = min_pct
        self.max_pct = max_pct

    def apply(self, df: pd.DataFrame) -> pd.Series:
        false_series = pd.Series(False, index=df.index)

        # 优先用数据库存的 turnover_rate
        if "turnover_rate" in df.columns and df["turnover_rate"].notna().any():
            tr = df["turnover_rate"].astype(float)
        elif "outstanding_share" in df.columns and df["outstanding_share"].notna().any():
            outstanding = df["outstanding_share"].replace(0, float("nan"))
            tr = df["volume"] / outstanding
        else:
            return false_series

        return ((tr > self.min_pct) & (tr < self.max_pct)).fillna(False)


class MAFilter(BaseFilter):
    """
    均线位置过滤。

    判断收盘价是否在 N 日均线的特定方向。

    参数
    ----
    period        : 均线周期
    direction     : "above" 要求收盘价 >= MA × (1 + tolerance_pct)
                    "below" 要求收盘价 <= MA × (1 + tolerance_pct)
    tolerance_pct : 允许偏离均线的幅度（默认 0 = 恰好在均线上方/下方）
                    常用于 "above" + tolerance_pct=-0.10 允许跌破10%以内
    """

    def __init__(
        self,
        period: int = 20,
        direction: str = "above",
        tolerance_pct: float = 0.0,
    ):
        self.period = period
        self.direction = direction
        self.tolerance_pct = tolerance_pct

    def apply(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        ma = close.rolling(self.period, min_periods=self.period).mean()
        threshold = ma * (1 + self.tolerance_pct)

        if self.direction == "above":
            return (close >= threshold).fillna(False)
        elif self.direction == "below":
            return (close <= threshold).fillna(False)
        else:
            raise ValueError(f"direction 只支持 'above'/'below'，当前: {self.direction}")


class RSIFilter(BaseFilter):
    """
    RSI 区间过滤。

    参数
    ----
    period    : RSI 计算周期（默认 14）
    max_rsi   : RSI 上限（默认 100 = 不限制上限）
    min_rsi   : RSI 下限（默认 0 = 不限制下限）

    例：RSIFilter(max_rsi=30) → 超卖区域（RSI < 30）
        RSIFilter(min_rsi=70) → 超买区域（RSI > 70）
    """

    def __init__(self, period: int = 14, max_rsi: float = 100.0, min_rsi: float = 0.0):
        self.period = period
        self.max_rsi = max_rsi
        self.min_rsi = min_rsi

    def apply(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / self.period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / self.period, adjust=False).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi = 100 - 100 / (1 + rs)

        return ((rsi >= self.min_rsi) & (rsi <= self.max_rsi)).fillna(False)


class GainFilter(BaseFilter):
    """
    N 日涨幅区间过滤（防追高 / 防深跌）。

    gain_N = close / close.shift(N) - 1

    参数
    ----
    period   : 计算涨幅的周期（默认 10）
    max_gain : 涨幅上限（默认 1.0 = 100%，即不限制）
    min_gain : 涨幅下限（默认 -1.0 = -100%，即不限制）

    例：GainFilter(period=10, max_gain=0.6) → 10日涨幅 < 60%（防追高）
    """

    def __init__(self, period: int = 10, max_gain: float = 1.0, min_gain: float = -1.0):
        self.period = period
        self.max_gain = max_gain
        self.min_gain = min_gain

    def apply(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        gain = close / close.shift(self.period) - 1
        return ((gain >= self.min_gain) & (gain <= self.max_gain)).fillna(False)


class LimitMoveFilter(BaseFilter):
    """
    排除涨停/跌停（当日无法正常买卖）。

    以涨跌幅 ±9.5% 为阈值（适配 A 股主板 ±10% 和科创/创业板 ±20%）。
    若需科创板/创业板 ±20%，调整 threshold=0.195。

    参数
    ----
    exclude_up   : 排除涨停日（当日涨幅接近上限）
    exclude_down : 排除跌停日（当日跌幅接近下限）
    threshold    : 判断涨跌停的绝对涨幅阈值（默认 0.095）
    """

    def __init__(
        self,
        exclude_up: bool = True,
        exclude_down: bool = True,
        threshold: float = 0.095,
    ):
        self.exclude_up = exclude_up
        self.exclude_down = exclude_down
        self.threshold = threshold

    def apply(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        prev_close = close.shift(1)
        daily_return = (close - prev_close) / prev_close

        mask = pd.Series(True, index=df.index)
        if self.exclude_up:
            mask &= daily_return < self.threshold
        if self.exclude_down:
            mask &= daily_return > -self.threshold

        return mask.fillna(False)


class VolumeFilter(BaseFilter):
    """
    成交量相对均量倍数过滤。

    vol_ratio = volume / rolling_mean(volume, vol_ma_days)

    参数
    ----
    vol_ma_days      : 均量计算窗口（默认 20）
    min_multiplier   : 成交量最小倍数（默认 0 = 不限制下限）
    max_multiplier   : 成交量最大倍数（默认 inf = 不限制上限）

    例：VolumeFilter(min_multiplier=2.0) → 成交量 >= 均量 2 倍（放量）
    """

    def __init__(
        self,
        vol_ma_days: int = 20,
        min_multiplier: float = 0.0,
        max_multiplier: float = float("inf"),
    ):
        self.vol_ma_days = vol_ma_days
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier

    def apply(self, df: pd.DataFrame) -> pd.Series:
        volume = df["volume"]
        vol_ma = volume.rolling(self.vol_ma_days, min_periods=5).mean()
        vol_ratio = volume / vol_ma.replace(0, float("nan"))

        return (
            (vol_ratio >= self.min_multiplier) & (vol_ratio <= self.max_multiplier)
        ).fillna(False)


class BoardFilter(BaseFilter):
    """
    按上市板块过滤。

    A 股板块与代码对应关系：
      主板   : 600xxx 601xxx 603xxx 605xxx（沪市）/ 000xxx 001xxx 002xxx 003xxx（深市）
      创业板 : 300xxx 301xxx
      科创板 : 688xxx 689xxx

    参数
    ----
    exclude_boards : 需要排除的板块列表。可选值: "cyb"（创业板）、"kcb"（科创板）
                     例: exclude_boards=["cyb", "kcb"] 只保留主板

    注意：该过滤器需要 DataFrame 的 attrs["symbol"] 字段（由 StrategyEvaluator.load_data 注入）。
          若未注入，则不过滤（返回全 True）。
    """

    _CYB_PREFIXES = ("300", "301")
    _KCB_PREFIXES = ("688", "689")

    def __init__(self, exclude_boards: list[str] = None):
        self.exclude_boards = [b.lower() for b in (exclude_boards or [])]

    def apply(self, df: pd.DataFrame) -> pd.Series:
        symbol = df.attrs.get("symbol", "")
        if not symbol:
            return pd.Series(True, index=df.index)

        excluded = False
        if "cyb" in self.exclude_boards and symbol.startswith(self._CYB_PREFIXES):
            excluded = True
        if "kcb" in self.exclude_boards and symbol.startswith(self._KCB_PREFIXES):
            excluded = True

        return pd.Series(not excluded, index=df.index)

    def __repr__(self) -> str:
        return f"BoardFilter(exclude_boards={self.exclude_boards})"


class MAAlignmentFilter(BaseFilter):
    """
    均线多头/空头排列过滤。

    多头排列（direction="bull"）：MA(periods[0]) > MA(periods[1]) > ... > MA(periods[-1])
    即短期均线 > 中期均线 > 长期均线，表示上升趋势。

    空头排列（direction="bear"）：MA(periods[0]) < MA(periods[1]) < ... < MA(periods[-1])
    即短期均线 < 长期均线，表示下降趋势。

    参数
    ----
    periods   : 均线周期列表，按从短到长排列，如 [5, 10, 20, 60]
    direction : "bull" 多头排列（短线在上）| "bear" 空头排列（短线在下）

    例：
        MAAlignmentFilter([5, 10, 20, 60], direction="bull")
        → MA5 > MA10 > MA20 > MA60，均线多头排列
    """

    def __init__(
        self,
        periods: list[int] = None,
        direction: str = "bull",
    ):
        self.periods = periods if periods is not None else [5, 10, 20, 60]
        self.direction = direction
        if len(self.periods) < 2:
            raise ValueError("periods 至少需要 2 个周期")

    def apply(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]

        # 计算各周期均线
        mas = [
            close.rolling(p, min_periods=p).mean()
            for p in self.periods
        ]

        # 逐对检查相邻均线的大小关系
        mask = pd.Series(True, index=df.index)
        for i in range(len(mas) - 1):
            if self.direction == "bull":
                mask &= mas[i] > mas[i + 1]   # 短线 > 长线
            else:
                mask &= mas[i] < mas[i + 1]   # 短线 < 长线

        return mask.fillna(False)

    def __repr__(self) -> str:
        return (
            f"MAAlignmentFilter(periods={self.periods}, direction='{self.direction}')"
        )


class RegimeFilter(BaseFilter):
    """
    市场状态过滤。

    根据当日大盘状态（趋势/震荡/危机/过渡期）过滤信号。
    需要预先计算市场状态序列（由 RegimeDetector 产生）。

    参数
    ----
    regime_series : pd.Series，index=trade_date，值为
                    'trend'|'range'|'crisis'|'transition'
    allow         : 允许入场的状态列表，如 ['trend'] 或 ['trend','range']

    例：
        from strategies.regime import get_regime
        regime = get_regime(db)
        RegimeFilter(regime_series=regime, allow=['trend'])
        → 只在趋势市允许入场
    """

    def __init__(self, regime_series: pd.Series, allow: list[str] = None):
        self._regime = regime_series
        self._allow = set(allow) if allow else {'trend', 'range', 'transition'}

    def apply(self, df: pd.DataFrame) -> pd.Series:
        if self._regime.empty:
            return pd.Series(True, index=df.index)

        # 对齐市场状态到个股日期
        regime_aligned = self._regime.reindex(df.index)
        mask = regime_aligned.isin(self._allow)
        return mask.fillna(False)

    def __repr__(self) -> str:
        return f"RegimeFilter(allow={sorted(self._allow)})"


# ------------------------------------------------------------------
# Screener 组合类
# ------------------------------------------------------------------

class Screener:
    """
    选股器：将多个 Filter 串联（AND 逻辑）。

    用法：
        sc = Screener([
            MarketCapFilter(max_yi=200),
            TurnoverFilter(min_pct=0.10, max_pct=0.30),
            LimitMoveFilter(),
        ])
        eligible = sc.apply(df)   # bool Series
    """

    def __init__(self, filters: list[BaseFilter]):
        self.filters = filters

    def apply(self, df: pd.DataFrame) -> pd.Series:
        """串联所有过滤器，返回 AND 组合后的 bool Series"""
        mask = pd.Series(True, index=df.index)
        for f in self.filters:
            mask = mask & f.apply(df)
        return mask.fillna(False)

    def describe(self) -> str:
        """可读描述，用于日志和报告"""
        return " & ".join(repr(f) for f in self.filters)

    def __repr__(self) -> str:
        return f"Screener([{', '.join(repr(f) for f in self.filters)}])"


# ------------------------------------------------------------------
# 预设选股器
# ------------------------------------------------------------------

SCREENER_PRESETS: dict[str, Screener] = {
    "全市场基础": Screener([
        LimitMoveFilter(),
    ]),
    "小市值活跃股": Screener([
        MarketCapFilter(max_yi=200),
        TurnoverFilter(min_pct=0.10, max_pct=0.30),
        LimitMoveFilter(),
    ]),
    "小市值宽松": Screener([
        MarketCapFilter(max_yi=200),
        TurnoverFilter(min_pct=0.05, max_pct=0.35),
        LimitMoveFilter(),
    ]),
    "超卖小市值": Screener([
        MarketCapFilter(max_yi=200),
        TurnoverFilter(min_pct=0.05, max_pct=0.30),
        RSIFilter(max_rsi=35),
        LimitMoveFilter(),
    ]),
    "大市值稳健": Screener([
        MarketCapFilter(min_yi=200),
        TurnoverFilter(min_pct=0.02, max_pct=0.15),
        LimitMoveFilter(),
    ]),
    "趋势向上": Screener([
        MAFilter(period=20, direction="above", tolerance_pct=0.0),
        MAFilter(period=60, direction="above", tolerance_pct=-0.05),
        LimitMoveFilter(),
    ]),
    "放量活跃": Screener([
        VolumeFilter(vol_ma_days=20, min_multiplier=1.5),
        LimitMoveFilter(),
    ]),
    "均线多头排列": Screener([
        MAAlignmentFilter(periods=[5, 10, 20, 60], direction="bull"),
        LimitMoveFilter(),
    ]),
    "均线多头排列_小市值": Screener([
        MarketCapFilter(max_yi=200),
        TurnoverFilter(min_pct=0.05, max_pct=0.35),
        MAAlignmentFilter(periods=[5, 10, 20, 60], direction="bull"),
        LimitMoveFilter(),
    ]),
    "均线空头排列": Screener([
        MAAlignmentFilter(periods=[5, 10, 20, 60], direction="bear"),
        LimitMoveFilter(),
    ]),
    "主板": Screener([
        BoardFilter(exclude_boards=["cyb", "kcb"]),
        LimitMoveFilter(),
    ]),
    "主板_涨停9.9": Screener([
        BoardFilter(exclude_boards=["cyb", "kcb"]),
        LimitMoveFilter(threshold=0.099),
    ]),
    "主板小市值": Screener([
        BoardFilter(exclude_boards=["cyb", "kcb"]),
        MarketCapFilter(max_yi=200),
        TurnoverFilter(min_pct=0.05, max_pct=0.35),
        LimitMoveFilter(),
    ]),
}


# ------------------------------------------------------------------
# 带 Regime 过滤的动态预设（运行时注入 regime_series）
# ------------------------------------------------------------------

def make_regime_screener(
    regime_series: pd.Series,
    allow: list[str],
    base_screener: str = "全市场基础",
) -> Screener:
    """
    在已有预设上叠加 Regime 过滤。

    用法：
        from strategies.regime import get_regime
        regime = get_regime(db)
        screener = make_regime_screener(regime, ['trend'])
    """
    base = SCREENER_PRESETS.get(base_screener, SCREENER_PRESETS["全市场基础"])
    return Screener(base.filters + [RegimeFilter(regime_series=regime_series, allow=allow)])

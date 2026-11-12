"""
策略抽象基类。

用户自定义策略只需继承 BaseStrategy 并实现 on_bar() 方法：

    class MyStrategy(BaseStrategy):
        name = "我的策略"

        def on_bar(self, ctx: BarContext) -> list[Order]:
            orders = []
            for symbol in ctx.universe:
                df = ctx.data[symbol]
                if len(df) < 20:
                    continue
                ma20 = df['close'].rolling(20).mean().iloc[-1]
                if df['close'].iloc[-1] > ma20:
                    qty = self.calc_qty(ctx.portfolio.cash, df['close'].iloc[-1])
                    if qty > 0:
                        orders.append(Order(symbol, 'buy', qty))
            return orders
"""

from abc import ABC, abstractmethod

import pandas as pd

from engine.broker import Order
from engine.context import BarContext


class BaseStrategy(ABC):
    """
    策略接口。

    prepare(data) 在回测开始前调用一次，可预计算技术指标。
    on_bar(ctx)  每个交易日收盘后调用，返回次日待执行订单。
    """

    name: str = "BaseStrategy"
    params: dict = {}

    def __init__(self, **params):
        if params:
            self.params = params
            self.name = f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k,v in params.items())})"
        else:
            self.name = self.__class__.__name__

    def prepare(self, data: dict[str, pd.DataFrame]) -> None:
        """
        [可选覆盖] 回测开始前的预计算钩子。
        data 包含完整历史数据，适合预先计算技术指标矩阵（向量化加速）。

        警告：在 on_bar 中访问预计算指标时，必须用 ctx.current_date 索引，
              不能用 .iloc[-1]（否则引入未来数据）：
              正确：self._ma[symbol].loc[ctx.current_date]
              错误：self._ma[symbol].iloc[-1]
        """
        pass

    @abstractmethod
    def on_bar(self, ctx: BarContext) -> list[Order]:
        """
        每个交易日收盘后调用。
        ctx.data[symbol] 只包含截至当日的历史数据（含当日收盘）。
        返回次日开盘执行的订单列表。
        """
        ...

    def on_fill(self, fill) -> None:
        """[可选] 成交回调"""
        pass

    # ------------------------------------------------------------------
    # 工具方法（子类可直接使用）
    # ------------------------------------------------------------------

    @staticmethod
    def calc_qty(
        cash: float,
        price: float,
        fraction: float = 0.1,
        lot_size: int = 100,
    ) -> int:
        """
        按可用资金的 fraction 比例计算买入手数（向下取整到 lot_size 的整数倍）。
        A 股最小交易单位 100 股（1手）。
        """
        if price <= 0 or cash <= 0:
            return 0
        max_qty = cash * fraction / price
        return int(max_qty / lot_size) * lot_size

    @staticmethod
    def is_crossover(fast: pd.Series, slow: pd.Series) -> bool:
        """判断最近两根 bar 是否发生金叉（fast 上穿 slow）"""
        if len(fast) < 2 or len(slow) < 2:
            return False
        return (fast.iloc[-2] <= slow.iloc[-2]) and (fast.iloc[-1] > slow.iloc[-1])

    @staticmethod
    def is_crossunder(fast: pd.Series, slow: pd.Series) -> bool:
        """判断最近两根 bar 是否发生死叉（fast 下穿 slow）"""
        if len(fast) < 2 or len(slow) < 2:
            return False
        return (fast.iloc[-2] >= slow.iloc[-2]) and (fast.iloc[-1] < slow.iloc[-1])

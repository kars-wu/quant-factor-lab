"""
Broker：模拟撮合引擎。

交易成本模型（A股）：
  - 佣金：max(成交额 × 万三, 最低5元)，买卖双向
  - 印花税：卖出成交额 × 千一（仅卖出）
  - 滑点：成交价 × 滑点率（买入加滑点，卖出减滑点）

成交时机（防未来函数）：
  T 日收盘后 on_bar 产生信号 → 订单放入 pending
  T+1 日开盘价（加滑点）成交
"""

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from engine.portfolio import Fill, Portfolio
from strategies.cost_model import CostModel


@dataclass
class Order:
    symbol: str
    direction: str          # 'buy' / 'sell'
    quantity: int           # 股数（必须是100的倍数，A股最小单位）
    order_date: str         # 信号产生日期（T日）
    order_type: str = "market"


class Broker:

    def __init__(self, cost: CostModel = None):
        self._cost = cost or CostModel()

    def execute(
        self,
        order: Order,
        next_open: float,
        fill_date: str,
        portfolio: Portfolio,
    ) -> Optional[Fill]:
        """
        执行订单，返回 Fill；资金不足或数量为0返回 None。

        next_open : T+1 日开盘价（不复权，用于计算实际成本）
        fill_date : T+1 日日期
        """
        if order.quantity <= 0:
            return None

        if next_open <= 0:
            logger.warning("{}@{} 开盘价无效 ({}), 跳过", order.symbol, fill_date, next_open)
            return None

        # 含滑点的成交价
        if order.direction == "buy":
            fill_price = round(self._cost.apply_slippage_buy(next_open), 4)
        else:
            fill_price = round(self._cost.apply_slippage_sell(next_open), 4)

        commission, stamp_duty, slippage_cost = self._calc_cost(
            order.direction, fill_price, order.quantity
        )

        fill = Fill(
            symbol=order.symbol,
            direction=order.direction,
            trade_date=fill_date,
            price=fill_price,
            quantity=order.quantity,
            commission=commission,
            stamp_duty=stamp_duty,
            slippage_cost=slippage_cost,
            order_date=order.order_date,
        )

        portfolio.apply_fill(fill)
        return fill

    def _calc_cost(
        self,
        direction: str,
        price: float,
        quantity: int,
    ) -> tuple[float, float, float]:
        """返回 (commission, stamp_duty, slippage_cost)"""
        trade_amount = price * quantity

        commission = max(trade_amount * self._cost.commission_rate, self._cost.min_commission)
        stamp_duty = trade_amount * self._cost.stamp_duty_rate if direction == "sell" else 0.0
        # 滑点成本已内嵌到成交价中，此处记录以供报告分析
        slippage_cost = trade_amount * self._cost.slippage_rate

        return round(commission, 4), round(stamp_duty, 4), round(slippage_cost, 4)

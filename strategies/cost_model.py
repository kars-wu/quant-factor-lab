"""
统一交易成本模型。

━━━ 问题背景 ━━━

此前成本参数在三处分别定义且不一致：
  TradeSimulator  — 滑点 10bps，无最低佣金
  PortfolioManager — 滑点 10bps，无最低佣金
  engine/Broker   — 滑点 2bps，最低佣金 5 元

统一为单一来源（config/settings.py），所有消费者引用同一个 CostModel。

━━━ 成本构成（A股）━━━

  佣金    = max(成交额 × 万一, 最低5元)  — 买卖双向
  印花税  = 卖出成交额 × 千一             — 仅卖出
  滑点    = 成交价 × slippage_rate        — 买入加，卖出减

━━━ 用法 ━━━

    cost = CostModel()  # 使用默认参数
    buy_cost  = cost.buy_cost(amount)    # 买入总成本（佣金+滑点）
    sell_cost = cost.sell_cost(amount)   # 卖出总成本（佣金+印花税+滑点）
    buy_price  = cost.apply_slippage_buy(raw_price)
    sell_price = cost.apply_slippage_sell(raw_price)

也可自定义：
    cost = CostModel(slippage_rate=0.005)  # 50bps 滑点（保守估计）
"""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import (
    DEFAULT_COMMISSION_RATE,
    DEFAULT_MIN_COMMISSION,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_STAMP_DUTY_RATE,
)


@dataclass
class CostModel:
    """统一交易成本模型（A股）。"""

    commission_rate: float = DEFAULT_COMMISSION_RATE    # 万一
    min_commission: float = DEFAULT_MIN_COMMISSION      # 最低5元
    stamp_duty_rate: float = DEFAULT_STAMP_DUTY_RATE     # 千一（卖出）
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE         # 竞价策略默认0（可配置）

    # ------ 成本计算 ------

    def buy_cost(self, amount: float) -> float:
        """买入时的总费用（佣金 + 滑点成本）。"""
        commission = max(amount * self.commission_rate, self.min_commission)
        slippage = amount * self.slippage_rate
        return commission + slippage

    def sell_cost(self, amount: float) -> float:
        """卖出时的总费用（佣金 + 印花税 + 滑点成本）。"""
        commission = max(amount * self.commission_rate, self.min_commission)
        stamp_duty = amount * self.stamp_duty_rate
        slippage = amount * self.slippage_rate
        return commission + stamp_duty + slippage

    def total_round_trip_cost(self, amount: float) -> float:
        """一买一卖的总费用。"""
        return self.buy_cost(amount) + self.sell_cost(amount)

    # ------ 价格调整 ------

    def apply_slippage_buy(self, price: float) -> float:
        """买入实际成交价（加滑点）。"""
        return price * (1 + self.slippage_rate)

    def apply_slippage_sell(self, price: float) -> float:
        """卖出实际成交价（减滑点）。"""
        return price * (1 - self.slippage_rate)

    # ------ 成本率（用于向量化快速估算）------

    @property
    def buy_cost_rate(self) -> float:
        """买入成本占成交额的比例（不含最低佣金的近似值）。"""
        return self.commission_rate + self.slippage_rate

    @property
    def sell_cost_rate(self) -> float:
        """卖出成本占成交额的比例。"""
        return self.commission_rate + self.stamp_duty_rate + self.slippage_rate

    @property
    def round_trip_cost_rate(self) -> float:
        """一买一卖的总成本率（用于 TradeSimulator 的快速估算）。"""
        return self.buy_cost_rate + self.sell_cost_rate

"""
Portfolio：持仓与资金管理。

所有状态变更通过 apply_fill() 触发（由 Broker 调用），
外部只能通过 snapshot() 触发每日市值记录。
"""

from dataclasses import dataclass

import pandas as pd
from loguru import logger

from engine.context import PortfolioView


@dataclass
class Position:
    symbol: str
    quantity: int           # 持有股数
    avg_cost: float         # 平均成本（含手续费，每股）
    open_date: str          # 建仓日期
    high_watermark: float   # 持仓期间最高收盘价（用于跟踪止损）


@dataclass
class Fill:
    """成交回报"""
    symbol: str
    direction: str          # 'buy' / 'sell'
    trade_date: str
    price: float            # 成交价（含滑点）
    quantity: int
    commission: float
    stamp_duty: float
    slippage_cost: float
    order_date: str         # 信号产生日期（T日）
    pnl: float = 0.0        # 本笔净盈亏（卖出时填入）


class Portfolio:

    def __init__(self, initial_capital: float):
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._positions: dict[str, Position] = {}
        self._nav_series: dict[str, float] = {}   # {trade_date: nav}
        self._fills: list[Fill] = []

    # ------------------------------------------------------------------
    # 只读属性
    # ------------------------------------------------------------------

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    @property
    def initial_capital(self) -> float:
        return self._initial_capital

    def market_value(self, prices: dict[str, float]) -> float:
        """当前持仓总市值（按给定价格）"""
        mv = 0.0
        for symbol, pos in self._positions.items():
            price = prices.get(symbol, pos.avg_cost)
            mv += pos.quantity * price
        return mv

    def total_value(self, prices: dict[str, float]) -> float:
        return self._cash + self.market_value(prices)

    def nav(self, prices: dict[str, float]) -> float:
        return self.total_value(prices) / self._initial_capital

    def get_readonly_view(self, prices: dict[str, float]) -> PortfolioView:
        pos_view = {
            s: {
                "quantity": p.quantity,
                "avg_cost": p.avg_cost,
                "open_date": p.open_date,
                "high_watermark": p.high_watermark,
            }
            for s, p in self._positions.items()
        }
        return PortfolioView(
            cash=self._cash,
            total_value=self.total_value(prices),
            positions=pos_view,
            nav=self.nav(prices),
        )

    # ------------------------------------------------------------------
    # 状态变更（由 Broker 调用）
    # ------------------------------------------------------------------

    def apply_fill(self, fill: Fill):
        """处理一笔成交，更新现金和持仓"""
        total_cost = fill.commission + fill.stamp_duty + fill.slippage_cost
        self._fills.append(fill)

        if fill.direction == "buy":
            trade_amount = fill.price * fill.quantity
            cash_out = trade_amount + total_cost

            if cash_out > self._cash + 1e-6:
                logger.warning(
                    "资金不足：买入 {} {}股@{:.2f} 需要 {:.2f}，可用 {:.2f}，已调整数量",
                    fill.symbol, fill.quantity, fill.price, cash_out, self._cash
                )
                # 根据可用资金调整数量
                fill.quantity = max(
                    0,
                    int((self._cash - total_cost) / fill.price / 100) * 100
                )
                if fill.quantity == 0:
                    return
                trade_amount = fill.price * fill.quantity
                cash_out = trade_amount + total_cost

            self._cash -= cash_out

            if fill.symbol in self._positions:
                pos = self._positions[fill.symbol]
                total_qty = pos.quantity + fill.quantity
                pos.avg_cost = (pos.avg_cost * pos.quantity + fill.price * fill.quantity) / total_qty
                pos.quantity = total_qty
                pos.high_watermark = max(pos.high_watermark, fill.price)
            else:
                self._positions[fill.symbol] = Position(
                    symbol=fill.symbol,
                    quantity=fill.quantity,
                    avg_cost=fill.price,
                    open_date=fill.trade_date,
                    high_watermark=fill.price,
                )

        elif fill.direction == "sell":
            if fill.symbol not in self._positions:
                logger.warning("尝试卖出未持有的标的 {}", fill.symbol)
                return

            pos = self._positions[fill.symbol]
            sell_qty = min(fill.quantity, pos.quantity)
            fill.quantity = sell_qty

            trade_amount = fill.price * sell_qty
            cash_in = trade_amount - total_cost
            self._cash += cash_in

            # 计算本笔盈亏
            fill.pnl = (fill.price - pos.avg_cost) * sell_qty - total_cost

            pos.quantity -= sell_qty
            if pos.quantity <= 0:
                del self._positions[fill.symbol]

    # ------------------------------------------------------------------
    # 每日快照
    # ------------------------------------------------------------------

    def snapshot(self, trade_date: str, close_prices: dict[str, float]):
        """记录当日净值"""
        # 更新持仓最高价
        for symbol, pos in self._positions.items():
            price = close_prices.get(symbol, pos.avg_cost)
            pos.high_watermark = max(pos.high_watermark, price)

        nav = self.total_value(close_prices) / self._initial_capital
        self._nav_series[trade_date] = nav

    # ------------------------------------------------------------------
    # 结果导出
    # ------------------------------------------------------------------

    def get_nav_series(self) -> pd.Series:
        s = pd.Series(self._nav_series)
        s.index.name = "trade_date"
        return s.sort_index()

    def get_fills(self) -> list[Fill]:
        return list(self._fills)

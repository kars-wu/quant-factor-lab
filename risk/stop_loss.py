"""
止损止盈规则（事件驱动模式）。

注意：此模块为 engine/ 事件驱动架构的遗留组件。
新的向量化架构使用 strategies/stops.py + strategies/trade_simulator.py
实现止损止盈，本模块保留但不再推荐使用。

在策略的 on_bar 中检查持仓，生成卖出信号。
"""


class StopLossRule:

    def __init__(
        self,
        stop_loss_pct: float = 0.08,
        take_profit_pct: float = None,
        trailing_stop_pct: float = None,
    ):
        """
        stop_loss_pct    : 固定止损（从成本价跌幅超过此比例触发）
        take_profit_pct  : 固定止盈（从成本价涨幅超过此比例触发）
        trailing_stop_pct: 跟踪止损（从最高价回撤超过此比例触发）
        """
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct

    def should_exit(
        self,
        avg_cost: float,
        current_price: float,
        high_watermark: float = None,
    ) -> tuple[bool, str]:
        """
        返回 (是否应该退出, 原因)
        """
        if avg_cost <= 0:
            return False, ""

        pnl_pct = (current_price - avg_cost) / avg_cost

        # 固定止损
        if self.stop_loss_pct and pnl_pct <= -self.stop_loss_pct:
            return True, f"止损({pnl_pct:.2%})"

        # 固定止盈
        if self.take_profit_pct and pnl_pct >= self.take_profit_pct:
            return True, f"止盈({pnl_pct:.2%})"

        # 跟踪止损
        if self.trailing_stop_pct and high_watermark and high_watermark > 0:
            drawdown = (current_price - high_watermark) / high_watermark
            if drawdown <= -self.trailing_stop_pct:
                return True, f"跟踪止损(从高点回撤{drawdown:.2%})"

        return False, ""

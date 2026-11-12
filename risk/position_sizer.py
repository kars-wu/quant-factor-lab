"""
仓位计算器。
"""


class PositionSizer:

    @staticmethod
    def fixed_fraction(
        cash: float,
        price: float,
        fraction: float = 0.1,
        lot_size: int = 100,
    ) -> int:
        """固定比例仓位，返回股数（向下取整到手）"""
        if price <= 0 or cash <= 0:
            return 0
        return int(cash * fraction / price / lot_size) * lot_size

    @staticmethod
    def equal_weight(
        total_value: float,
        price: float,
        n_total: int,
        n_current: int = 0,
        lot_size: int = 100,
    ) -> int:
        """等权重仓位（按最大持仓数均分总资产）"""
        if n_total <= n_current or price <= 0:
            return 0
        per = total_value / n_total
        return int(per / price / lot_size) * lot_size

    @staticmethod
    def volatility_target(
        total_value: float,
        price: float,
        atr_value: float,
        target_risk: float = 0.01,
        atr_multiplier: float = 2.0,
        lot_size: int = 100,
    ) -> int:
        """
        波动率目标仓位：单次止损不超过 total_value × target_risk。
        止损幅度 = atr_multiplier × ATR
        """
        if atr_value <= 0 or price <= 0:
            return 0
        stop_distance = atr_multiplier * atr_value
        risk_per_share = stop_distance
        max_risk = total_value * target_risk
        qty = int(max_risk / risk_per_share / lot_size) * lot_size
        return qty

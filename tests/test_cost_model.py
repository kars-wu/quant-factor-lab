"""统一成本模型测试。

断言方式刻意从 ``config.settings`` 取值，避免默认参数调整后测试变成
"记得改数字" 的负担；这里真正要守住的是**公式与最低佣金规则**。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    DEFAULT_COMMISSION_RATE,
    DEFAULT_MIN_COMMISSION,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_STAMP_DUTY_RATE,
)
from strategies.cost_model import CostModel


class TestCostModel:

    def test_defaults_match_settings(self):
        cost = CostModel()
        assert cost.commission_rate == DEFAULT_COMMISSION_RATE
        assert cost.min_commission == DEFAULT_MIN_COMMISSION
        assert cost.stamp_duty_rate == DEFAULT_STAMP_DUTY_RATE
        assert cost.slippage_rate == DEFAULT_SLIPPAGE_RATE

    def test_buy_cost_uses_rate_above_minimum(self):
        cost = CostModel()
        amount = 1_000_000
        assert cost.commission_rate * amount > cost.min_commission
        assert cost.buy_cost(amount) == pytest.approx(
            amount * cost.commission_rate + amount * cost.slippage_rate)

    def test_buy_cost_hits_minimum_commission(self):
        cost = CostModel()
        amount = 1000
        assert cost.commission_rate * amount < cost.min_commission
        assert cost.buy_cost(amount) == pytest.approx(
            cost.min_commission + amount * cost.slippage_rate)

    def test_sell_cost_includes_stamp_duty(self):
        cost = CostModel()
        amount = 1_000_000
        assert cost.sell_cost(amount) == pytest.approx(
            amount * cost.commission_rate
            + amount * cost.stamp_duty_rate
            + amount * cost.slippage_rate)
        assert cost.sell_cost(amount) > cost.buy_cost(amount)

    def test_sell_cost_small_amount_hits_minimum(self):
        cost = CostModel()
        amount = 1000
        assert cost.sell_cost(amount) == pytest.approx(
            cost.min_commission + amount * cost.stamp_duty_rate
            + amount * cost.slippage_rate)

    def test_round_trip_is_sum_of_legs(self):
        cost = CostModel()
        amount = 200_000
        assert cost.total_round_trip_cost(amount) == pytest.approx(
            cost.buy_cost(amount) + cost.sell_cost(amount))

    def test_round_trip_rate_matches_formula(self):
        cost = CostModel()
        expected = 2 * cost.commission_rate + cost.stamp_duty_rate + 2 * cost.slippage_rate
        assert cost.round_trip_cost_rate == pytest.approx(expected)

    def test_slippage_price_adjustment(self):
        cost = CostModel(slippage_rate=0.01)
        assert cost.apply_slippage_buy(10.0) == pytest.approx(10.1)
        assert cost.apply_slippage_sell(10.0) == pytest.approx(9.9)

    def test_custom_slippage_increases_cost(self):
        conservative = CostModel(slippage_rate=0.005)
        assert conservative.slippage_rate == 0.005
        assert conservative.buy_cost(100_000) > CostModel().buy_cost(100_000)

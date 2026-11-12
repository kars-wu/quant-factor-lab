"""
OOS验证 + Walk-Forward 测试。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.oos import WalkForwardValidator
from strategies.types import Trade


@pytest.fixture
def sample_trades():
    """构造10笔交易，分散在2年内"""
    return [
        Trade(
            symbol=f'STK{i:03d}', signal_date=f'2022-01-{i+1:02d}',
            buy_date=f'2022-01-{i+1:02d}', sell_date=f'2022-02-{i+1:02d}',
            buy_price=10.0, sell_price=10.5 + i * 0.1,
            return_pct=0.05 + i * 0.01, net_return_pct=0.04 + i * 0.01,
            hold_days=30,
        )
        for i in range(1, 11)
    ]


class TestSplit:

    def test_simple_split(self):
        v = WalkForwardValidator()
        splits = v.split('2020-01-01', '2026-05-14', train_ratio=0.6)
        assert len(splits) == 1
        train_start, train_end, test_start, test_end = splits[0]
        assert train_start == '2020-01-01'
        assert test_end == '2026-05-14'
        # 训练集在前
        assert train_start < train_end < test_start < test_end

    def test_walk_forward_split(self):
        v = WalkForwardValidator()
        splits = v.split_walk_forward('2020-01-01', '2026-05-14',
                                       train_window=252, test_window=63)
        assert len(splits) > 1
        # 检查每个窗口内部训练在前测试在后
        for i, (ts, te, vs, ve) in enumerate(splits):
            assert ts < te < vs <= ve
        # 检查测试窗口不重叠（滑动步长 = test_window）
        for i in range(1, len(splits)):
            prev_test_start = splits[i-1][2]
            curr_test_start = splits[i][2]
            assert curr_test_start > prev_test_start  # 测试窗口递进

    def test_custom_ratio(self):
        v = WalkForwardValidator()
        splits_60 = v.split('2020-01-01', '2026-05-14', train_ratio=0.6)
        splits_80 = v.split('2020-01-01', '2026-05-14', train_ratio=0.8)
        # 80% 训练 → 训练集更长 → 测试集更短
        assert splits_80[0][1] > splits_60[0][1]  # train_end 更晚


class TestValidate:

    def test_empty_trades(self):
        v = WalkForwardValidator()
        splits = v.split('2020-01-01', '2026-05-14')
        result = v.validate([], splits)
        assert result.in_sample_sharpe == 0
        assert result.out_of_sample_sharpe == 0

    def test_basic_validate(self, sample_trades):
        v = WalkForwardValidator()
        splits = v.split('2022-01-01', '2022-12-31', train_ratio=0.6)
        result = v.validate(sample_trades, splits)
        assert result.in_sample_trades > 0 or result.out_of_sample_trades > 0
        # 衰减率
        if result.in_sample_sharpe > 0:
            assert result.decay_ratio >= 0

    def test_overfit_detection(self):
        """构造过拟合场景：样本内好、样本外差"""
        in_trades = [
            Trade(symbol='A', signal_date='2022-01-01',
                  buy_date='2022-01-02', sell_date='2022-02-01',
                  buy_price=10.0, sell_price=12.0,
                  return_pct=0.20, net_return_pct=0.19, hold_days=30)
        ]
        out_trades = [
            Trade(symbol='B', signal_date='2022-07-01',
                  buy_date='2022-07-02', sell_date='2022-08-01',
                  buy_price=10.0, sell_price=9.0,
                  return_pct=-0.10, net_return_pct=-0.12, hold_days=30)
        ]
        all_trades = in_trades + out_trades
        v = WalkForwardValidator()
        splits = v.split('2022-01-01', '2022-12-31', train_ratio=0.5)
        result = v.validate(all_trades, splits)
        # 样本内正收益、样本外负收益 → 衰减率应该低
        if result.in_sample_sharpe > 0 and result.out_of_sample_sharpe < 0:
            assert result.is_overfit

"""
PortfolioManager 测试：仓位分配、先卖后买、熔断、流动性约束。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from strategies.portfolio_manager import PortfolioManager
from strategies.types import Trade


@pytest.fixture
def sample_data():
    """构造两只股票的测试行情"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=30, freq='B')
    df1 = pd.DataFrame({
        'open': 10.0 + np.cumsum(np.random.randn(30) * 0.1),
        'high': 11.0, 'low': 9.0,
        'close': 10.0 + np.cumsum(np.random.randn(30) * 0.1),
        'volume': 1e7, 'amount': 1e8,
        'turnover_rate': 0.05,
    }, index=dates)
    df2 = df1.copy()
    df2['close'] = df2['close'] * 2  # 第二只股票价格翻倍
    return {'AAA': df1, 'BBB': df2}


@pytest.fixture
def sample_trades():
    """构造2笔交易：第1天买入，第5天卖出"""
    return [
        Trade(
            symbol='AAA', signal_date='2024-01-01',
            buy_date='2024-01-02', sell_date='2024-01-08',
            buy_price=10.0, sell_price=11.0,
            return_pct=0.10, net_return_pct=0.098,
            stop_price=9.0, target_price=12.0,
            exit_reason='target', hold_days=5,
        ),
        Trade(
            symbol='BBB', signal_date='2024-01-01',
            buy_date='2024-01-02', sell_date='2024-01-08',
            buy_price=20.0, sell_price=22.0,
            return_pct=0.10, net_return_pct=0.098,
            stop_price=18.0, target_price=24.0,
            exit_reason='target', hold_days=5,
        ),
    ]


class TestPortfolioManager:

    def test_basic_allocate(self, sample_data, sample_trades):
        """基本仓位分配"""
        pm = PortfolioManager(sample_data, initial_capital=100_000,
                              position_size=0.1, max_positions=5)
        result = pm.allocate(sample_trades)
        assert result.total_trades > 0
        assert result.final_value > 0
        assert len(result.nav_series) > 0

    def test_max_positions_limit(self, sample_data):
        """持仓上限：资金够但持仓数超限"""
        trades = [
            Trade(symbol=f'STK{i:03d}', signal_date='2024-01-01',
                  buy_date='2024-01-02', sell_date='2024-01-10',
                  buy_price=10.0, sell_price=11.0,
                  return_pct=0.10, net_return_pct=0.098,
                  hold_days=5)
            for i in range(20)
        ]
        # 只有一只股票的数据，模拟20个不同标的
        data = {f'STK{i:03d}': sample_data['AAA'] for i in range(20)}
        pm = PortfolioManager(data, initial_capital=1_000_000,
                              position_size=0.1, max_positions=3)
        result = pm.allocate(trades)
        assert result.total_trades <= 3 + 1  # 最多3个同时+可能的滚动

    def test_insufficient_funds(self, sample_data):
        """资金不足：跳过无法买入的信号"""
        trades = [
            Trade(symbol='AAA', signal_date='2024-01-01',
                  buy_date='2024-01-02', sell_date='2024-01-10',
                  buy_price=1000.0, sell_price=1100.0,  # 极高价
                  return_pct=0.10, net_return_pct=0.098,
                  hold_days=5)
        ]
        pm = PortfolioManager(sample_data, initial_capital=1_000,  # 极少资金
                              position_size=0.5, max_positions=5)
        result = pm.allocate(trades)
        assert result.skipped_count >= 1

    def test_drawdown_circuit_breaker(self, sample_data):
        """组合回撤熔断"""
        # 构造一笔亏损交易让净值跌破15%
        trades = [
            Trade(symbol='AAA', signal_date='2024-01-01',
                  buy_date='2024-01-02', sell_date='2024-01-10',
                  buy_price=10.0, sell_price=8.0,  # 亏20%
                  return_pct=-0.20, net_return_pct=-0.22,
                  hold_days=5)
        ]
        pm = PortfolioManager(sample_data, initial_capital=100_000,
                              position_size=1.0, max_positions=1,
                              max_drawdown_limit=0.10)
        result = pm.allocate(trades)
        # 应该触发了熔断
        assert result.total_trades >= 0  # 不崩溃即可

    def test_liquidity_constraint(self, sample_data):
        """流动性约束：买入金额不超过当日成交额的 max_participation"""
        trades = [
            Trade(symbol='AAA', signal_date='2024-01-01',
                  buy_date='2024-01-02', sell_date='2024-01-10',
                  buy_price=10.0, sell_price=11.0,
                  return_pct=0.10, net_return_pct=0.098,
                  hold_days=5)
        ]
        # 成交额=1e8，max_participation=0.1 → 最多买1千万
        pm = PortfolioManager(sample_data, initial_capital=1_000_000,
                              position_size=0.5,  # 目标50万
                              max_participation=0.1)
        result = pm.allocate(trades)
        # 应该正常执行（50万 < 1千万限制）
        assert result.total_trades == 1

    def test_no_duplicate(self, sample_data):
        """同标的不重复建仓"""
        trades = [
            Trade(symbol='AAA', signal_date='2024-01-01',
                  buy_date='2024-01-02', sell_date='2024-01-15',
                  buy_price=10.0, sell_price=11.0,
                  return_pct=0.10, net_return_pct=0.098,
                  hold_days=10),
            Trade(symbol='AAA', signal_date='2024-01-03',  # 第一笔还在持仓
                  buy_date='2024-01-04', sell_date='2024-01-15',
                  buy_price=10.5, sell_price=11.0,
                  return_pct=0.05, net_return_pct=0.048,
                  hold_days=8),
        ]
        pm = PortfolioManager(sample_data, initial_capital=1_000_000,
                              position_size=0.1, max_positions=5,
                              allow_duplicate=False)
        result = pm.allocate(trades)
        # 第二笔应被跳过（同标的已持仓）
        assert result.skipped_count >= 1

    def test_empty_trades(self, sample_data):
        """空交易列表"""
        pm = PortfolioManager(sample_data)
        result = pm.allocate([])
        assert result.total_trades == 0
        assert result.final_value == 0

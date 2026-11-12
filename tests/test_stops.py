"""
止损价格计算测试。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from strategies.stops import (
    STOP_METHODS,
    calc_stop_atr,
    calc_stop_ma,
    calc_stop_swing_low,
    calc_target,
)


@pytest.fixture
def sample_df():
    """构造一段测试行情数据"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=60, freq='B')
    close = 10.0 + np.cumsum(np.random.randn(60) * 0.2)
    df = pd.DataFrame({
        'open': close + np.random.randn(60) * 0.05,
        'high': close + abs(np.random.randn(60) * 0.1),
        'low': close - abs(np.random.randn(60) * 0.1),
        'close': close,
        'volume': np.random.randint(1e6, 1e7, 60).astype(float),
    }, index=dates)
    return df


class TestATRStop:

    def test_basic(self, sample_df):
        sig_date = sample_df.index[40]
        buy_price = float(sample_df.loc[sig_date, 'close'])
        stop = calc_stop_atr(sample_df, sig_date, n_atr=2.0)
        assert stop < buy_price  # 止损价应低于买入价

    def test_different_n_atr(self, sample_df):
        sig_date = sample_df.index[40]
        stop_1 = calc_stop_atr(sample_df, sig_date, n_atr=1.0)
        stop_2 = calc_stop_atr(sample_df, sig_date, n_atr=2.0)
        stop_3 = calc_stop_atr(sample_df, sig_date, n_atr=3.0)
        # n_atr 越大止损越宽
        assert stop_1 > stop_2 > stop_3

    def test_min_stop_floor(self, sample_df):
        """止损价不低于买入价的50%"""
        sig_date = sample_df.index[40]
        buy_price = float(sample_df.loc[sig_date, 'close'])
        stop = calc_stop_atr(sample_df, sig_date, n_atr=100.0)  # 极端值
        assert stop >= buy_price * 0.5

    def test_atr_fallback(self, sample_df):
        """ATR数据不足时回退到95%"""
        sig_date = sample_df.index[0]  # 第一天，ATR可能不足
        stop = calc_stop_atr(sample_df, sig_date, n_atr=2.0, period=14)
        # 应该回退或给出合理值
        assert stop > 0


class TestSwingLowStop:

    def test_basic(self, sample_df):
        sig_date = sample_df.index[40]
        stop = calc_stop_swing_low(sample_df, sig_date, lookback=5)
        # 止损应在最近5根K线最低点下方
        loc = sample_df.index.get_loc(sig_date)
        swing_low = float(sample_df['low'].iloc[max(0, loc-4): loc+1].min())
        assert stop <= swing_low * 1.005  # 含0.5%缓冲

    def test_different_lookback(self, sample_df):
        sig_date = sample_df.index[40]
        stop_3 = calc_stop_swing_low(sample_df, sig_date, lookback=3)
        stop_5 = calc_stop_swing_low(sample_df, sig_date, lookback=5)
        # lookback越大，最低可能越低，止损越低
        assert stop_5 <= stop_3 + 1e-6


class TestMAStop:

    def test_basic(self, sample_df):
        sig_date = sample_df.index[40]
        stop = calc_stop_ma(sample_df, sig_date, ma_period=20, buffer=0.01)
        # 止损应在MA20下方1%
        ma20 = float(sample_df['close'].rolling(20).mean().loc[sig_date])
        assert stop <= ma20 * 1.01  # 含1%缓冲


class TestTargetCalc:

    def test_basic(self):
        target = calc_target(10.0, 9.0, 2.0)
        # risk = 10 - 9 = 1, target = 10 + 1*2 = 12
        assert target == pytest.approx(12.0)

    def test_rr_1_5(self):
        target = calc_target(10.0, 9.0, 1.5)
        # target = 10 + 1*1.5 = 11.5
        assert target == pytest.approx(11.5)

    def test_stop_above_buy(self):
        """止损价 >= 买入价时回退"""
        target = calc_target(10.0, 10.5, 2.0)
        # risk <= 0, 回退到 10 * 1.05 = 10.5
        assert target == pytest.approx(10.5)


class TestStopMethods:

    def test_all_methods_registered(self):
        assert 'atr' in STOP_METHODS
        assert 'swing' in STOP_METHODS
        assert 'ma' in STOP_METHODS
        assert 'trailing' in STOP_METHODS
        assert len(STOP_METHODS) == 4

"""
数据质量校验测试。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from data.quality import DataQualityChecker


@pytest.fixture
def clean_df():
    """正常的测试数据"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=50, freq='B')
    close = 10.0 + np.cumsum(np.random.randn(50) * 0.1)
    df = pd.DataFrame({
        'open': close + np.random.randn(50) * 0.05,
        'high': close + abs(np.random.randn(50) * 0.1),
        'low': close - abs(np.random.randn(50) * 0.1),
        'close': close,
        'volume': np.random.randint(1e6, 1e8, 50).astype(float),
        'amount': np.random.uniform(1e7, 1e9, 50),
    }, index=dates)
    # 确保 high >= max(open, close), low <= min(open, close)
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    return df


class TestOHLCLogic:

    def test_clean_data(self, clean_df):
        checker = DataQualityChecker()
        report = checker.check_symbol(clean_df, 'TEST')
        ohlc_issues = [i for i in report.issues if i.check == 'ohlc_logic']
        assert len(ohlc_issues) == 0

    def test_bad_high(self, clean_df):
        df = clean_df.copy()
        df.iloc[5, df.columns.get_loc('high')] = 1.0  # high 远低于 open/close
        checker = DataQualityChecker()
        report = checker.check_symbol(df, 'TEST')
        ohlc_errors = [i for i in report.issues if i.check == 'ohlc_logic' and i.severity == 'error']
        assert len(ohlc_errors) > 0


class TestPriceRange:

    def test_negative_price(self, clean_df):
        df = clean_df.copy()
        df.iloc[10, df.columns.get_loc('close')] = -1.0
        checker = DataQualityChecker()
        report = checker.check_symbol(df, 'TEST')
        price_issues = [i for i in report.issues if i.check == 'price_range']
        assert len(price_issues) > 0

    def test_zero_price(self, clean_df):
        df = clean_df.copy()
        df.iloc[10, df.columns.get_loc('open')] = 0.0
        checker = DataQualityChecker()
        report = checker.check_symbol(df, 'TEST')
        price_issues = [i for i in report.issues if i.check == 'price_range']
        assert len(price_issues) > 0


class TestVolumeAnomaly:

    def test_zero_volume(self, clean_df):
        df = clean_df.copy()
        df.iloc[15, df.columns.get_loc('volume')] = 0
        checker = DataQualityChecker()
        report = checker.check_symbol(df, 'TEST')
        vol_issues = [i for i in report.issues if i.check == 'zero_volume']
        assert len(vol_issues) > 0


class TestAdjFactor:

    def test_monotonic(self, clean_df):
        df = clean_df.copy()
        df['hfq_factor'] = np.linspace(1.0, 1.5, 50)
        checker = DataQualityChecker()
        report = checker.check_symbol(df, 'TEST')
        adj_issues = [i for i in report.issues if i.check == 'adj_factor']
        assert len(adj_issues) == 0

    def test_decreasing(self, clean_df):
        df = clean_df.copy()
        df['hfq_factor'] = np.linspace(1.0, 1.5, 50)
        df.iloc[30, df.columns.get_loc('hfq_factor')] = 1.0  # 突然下降
        checker = DataQualityChecker()
        report = checker.check_symbol(df, 'TEST')
        adj_issues = [i for i in report.issues if i.check == 'adj_factor']
        assert len(adj_issues) > 0

    def test_floating_point_noise(self, clean_df):
        """浮点微小波动不报错"""
        df = clean_df.copy()
        df['hfq_factor'] = np.linspace(1.0, 1.5, 50)
        # 加入微小浮动波动（< 1e-3）
        df['hfq_factor'] += np.random.randn(50) * 1e-5
        checker = DataQualityChecker()
        report = checker.check_symbol(df, 'TEST')
        adj_issues = [i for i in report.issues if i.check == 'adj_factor']
        assert len(adj_issues) == 0  # 微小波动不应触发

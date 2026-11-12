"""
多因子合成测试。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from strategies.factor_synthesis import MultiFactorSynthesizer


@pytest.fixture
def mock_factor_analyzer():
    """构造一个 mock FactorAnalyzer（不依赖数据库）。"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='B')
    symbols = [f'STK{i:03d}' for i in range(20)]

    # 构造因子数据
    records = []
    for sym in symbols:
        for date in dates:
            records.append((date, sym))
    idx = pd.MultiIndex.from_tuples(records, names=['date', 'symbol'])

    factors = {}
    # 3个独立因子 + 1个噪音因子 + 1个重复因子
    factors['momentum_5d'] = pd.Series(np.random.randn(len(idx)), index=idx)
    factors['rsi_14'] = pd.Series(np.random.randn(len(idx)), index=idx)
    factors['volume_ratio'] = pd.Series(np.random.randn(len(idx)), index=idx)
    factors['noise'] = pd.Series(np.random.randn(len(idx)), index=idx)
    factors['momentum_dup'] = factors['momentum_5d'] * 0.95  # 高相关

    # 构造目标：有信号的 = 0.3*momentum + 0.2*rsi + 0.1*volume + noise
    y = (0.3 * factors['momentum_5d']
         + 0.2 * factors['rsi_14']
         + 0.1 * factors['volume_ratio']
         + 0.5 * np.random.randn(len(idx)))
    y.name = 'forward_return'

    # 构造 mock analyzer
    class MockFA:
        def __init__(self):
            self._factors = factors
            self._returns = {5: y}

        def get_factor_matrix(self, names=None):
            if names is None:
                names = [n for n in self._factors if not n.endswith('_neutral')]
            return pd.DataFrame({n: self._factors[n] for n in names if n in self._factors}).sort_index()

        def get_forward_returns(self, days):
            return self._returns.get(days, pd.Series(dtype=float))

    return MockFA()


class TestSynthesis:

    def test_linear_synthesis(self, mock_factor_analyzer):
        """线性模型合成"""
        synth = MultiFactorSynthesizer(mock_factor_analyzer)
        result = synth.synthesize(model='linear', forward_days=5, train_ratio=0.6)
        assert result.model_name == 'linear'
        assert result.n_factors > 0
        assert result.train_ic != 0 or result.test_ic != 0
        assert len(result.alpha) > 0
        assert len(result.factor_weights) > 0

    def test_factor_deduplication(self, mock_factor_analyzer):
        """因子去重：高相关因子应被移除"""
        synth = MultiFactorSynthesizer(mock_factor_analyzer)
        result = synth.synthesize(model='linear', forward_days=5, corr_threshold=0.8)
        # momentum_dup 和 momentum_5d 相关性 > 0.8，应有一个被移除
        assert len(result.removed_factors) >= 1

    def test_alpha_not_empty(self, mock_factor_analyzer):
        """合成因子 alpha 不为空"""
        synth = MultiFactorSynthesizer(mock_factor_analyzer)
        result = synth.synthesize(model='linear', forward_days=5)
        assert isinstance(result.alpha, pd.Series)
        assert result.alpha.name == 'composite_alpha'

    def test_layer_backtest(self, mock_factor_analyzer):
        """分层回测结果"""
        synth = MultiFactorSynthesizer(mock_factor_analyzer)
        result = synth.synthesize(model='linear', forward_days=5)
        assert not result.layer_returns.empty
        assert len(result.layer_returns) == 5  # 5 层

    def test_overfit_detection(self, mock_factor_analyzer):
        """过拟合检测"""
        synth = MultiFactorSynthesizer(mock_factor_analyzer)
        result = synth.synthesize(model='linear', forward_days=5)
        # 由于有信号因子，应该不是过拟合
        assert isinstance(result.is_overfit, bool)
        assert isinstance(result.decay_ratio, float)

    def test_factor_weights(self, mock_factor_analyzer):
        """线性模型因子权重"""
        synth = MultiFactorSynthesizer(mock_factor_analyzer)
        result = synth.synthesize(model='linear', forward_days=5)
        # momentum_5d 应有权重（目标中有信号）
        assert 'momentum_5d' in result.factor_weights

    def test_summary(self, mock_factor_analyzer):
        """摘要格式"""
        synth = MultiFactorSynthesizer(mock_factor_analyzer)
        result = synth.synthesize(model='linear', forward_days=5)
        s = result.summary()
        assert 'linear' in s
        assert 'IC' in s

    def test_default_ic_prefilter_disabled(self, mock_factor_analyzer):
        """默认关闭 IC 预筛选，不应因阈值过滤删除低 IC 因子。"""
        synth = MultiFactorSynthesizer(mock_factor_analyzer)
        result = synth.synthesize(model='linear', forward_days=5)
        removed_text = " | ".join(result.removed_factors)
        assert "<0.0" not in removed_text

    def test_empty_factors(self):
        """无因子时不崩溃"""
        class EmptyFA:
            _factors = {}
            def get_factor_matrix(self, names=None):
                return pd.DataFrame()
            def get_forward_returns(self, days):
                return pd.Series(dtype=float)

        synth = MultiFactorSynthesizer(EmptyFA())
        result = synth.synthesize(model='linear', forward_days=5)
        assert result.n_factors == 0

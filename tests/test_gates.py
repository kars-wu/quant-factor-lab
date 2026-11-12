"""Gate0 工程不变量：能不能真的挡住已知缺陷。"""
import numpy as np
import pandas as pd

from multi_agent import dsl
from multi_agent.contracts import EvaluationSpec, ExecutionSpec
from multi_agent.evaluation import gates

SPEC = {'op': 'ts_pct_change', 'n': 5, 'inputs': [{'field': 'close'}]}


def test_prefix_invariance_passes_for_clean_factor(dev_view):
    item = gates.check_prefix_invariance(SPEC, dev_view)
    assert item.passed, item.detail


def test_prefix_invariance_catches_future_leak(dev_view):
    """用一个故意看未来的编译函数，验证检查器本身有效。"""
    leaky = lambda panels: panels['close'].pct_change(-1)     # noqa: E731
    item = gates.check_prefix_invariance(None, dev_view, compiler=leaky)
    assert not item.passed


def test_cross_section_invariance(dev_view):
    item = gates.check_cross_section_invariance(SPEC, dev_view, seed=3)
    assert item.passed, item.detail


def test_blend_skips_zero_weight_and_keeps_nan():
    idx = pd.date_range('2020-01-01', periods=3)
    base = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [2.0, 2.0, 2.0]}, index=idx)
    sparse = pd.DataFrame({'a': [np.nan, np.nan, 5.0], 'b': [np.nan] * 3}, index=idx)

    zero = gates.blend({'main': base, 'ev': sparse}, {'main': 1.0, 'ev': 0.0})
    pd.testing.assert_frame_equal(zero, base)

    both = gates.blend({'main': base, 'ev': sparse}, {'main': 1.0, 'ev': 1.0})
    assert both.isna().loc[idx[0], 'a'], '非零权重时缺失应继续传播'


def test_zero_weight_invariance_detects_nan_propagation(dev_view):
    item = gates.check_zero_weight_invariance(SPEC, dev_view)
    assert item.passed, item.detail
    base = dsl.compile_factor(SPEC, dev_view.panels)
    naive = base + 0.0 * base.where(base > base.stack().quantile(0.99))
    assert naive.isna().sum().sum() > base.isna().sum().sum(), '夹具本身要能触发 bug'


def test_coverage_rejects_big_cap_only_panel(dev_view):
    mv = dev_view.panels['mv']
    big_only = dev_view.panels['close'].where(mv.rank(axis=1, pct=True) > 0.95)
    limited = dev_view.with_panel('close', big_only)
    item = gates.check_coverage(SPEC, limited, min_coverage=0.6)
    assert not item.passed


def test_cash_identity_keeps_all_calendar_days(dev_view):
    spec = ExecutionSpec(hold_days=5)
    ev = EvaluationSpec(top_n=10)
    factor = dsl.compile_factor(SPEC, dev_view.panels)
    item = gates.check_cash_identity(factor, dev_view, spec, ev)
    assert item.passed, item.detail


def test_run_gate0_blocks_invalid_expression(dev_view):
    spec = ExecutionSpec()
    ev = EvaluationSpec(top_n=10)
    report = gates.run_gate0({'op': 'ts_shift', 'n': -3, 'inputs': [{'field': 'close'}]},
                             dev_view, spec, ev)
    assert not report.passed
    assert report.failures()[0].name == 'dsl_valid'
    assert 'Gate0 失败' in report.summary()


def test_run_gate0_passes_clean_candidate(dev_view):
    report = gates.run_gate0(SPEC, dev_view, ExecutionSpec(hold_days=5),
                             EvaluationSpec(top_n=10))
    assert report.passed, report.to_dict()


def test_audit_regression_suite_detects_all_known_defects(dev_view):
    suite = gates.audit_regression_suite(dev_view, ExecutionSpec(hold_days=5),
                                         EvaluationSpec(top_n=10))
    missing = [k for k, v in suite.items() if not v]
    assert not missing, f'这些已知缺陷没被检出: {missing}'


def test_design_rules_are_soft_by_default(dev_view):
    report = gates.run_gate0(SPEC, dev_view, ExecutionSpec(hold_days=5),
                             EvaluationSpec(top_n=10))
    item = next(i for i in report.items if i.name == 'design_richness')
    assert item.passed
    assert '建议' in item.detail


def test_design_rules_can_be_strict(dev_view):
    ev = EvaluationSpec(top_n=10, design_strict=True)
    report = gates.run_gate0(SPEC, dev_view, ExecutionSpec(hold_days=5), ev,
                             design_strict=True)
    item = next(i for i in report.items if i.name == 'design_richness')
    assert not item.passed
    assert not report.passed

    rich = {'op': 'mul', 'inputs': [
        {'op': 'cs_rank', 'inputs': [{'op': 'ts_pct_change', 'n': 20,
                                      'inputs': [{'field': 'close'}]}]},
        {'op': 'ts_corr', 'n': 20, 'inputs': [{'field': 'close'}, {'field': 'volume'}]},
    ]}
    ok = gates.run_gate0(rich, dev_view, ExecutionSpec(hold_days=5), ev, design_strict=True)
    assert next(i for i in ok.items if i.name == 'design_richness').passed

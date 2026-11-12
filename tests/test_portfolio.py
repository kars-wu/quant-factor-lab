"""tranche 回测：不加杠杆、成本可回溯、空仓日不丢、结果可复现。"""
import pandas as pd

from multi_agent import dsl
from multi_agent.contracts import EvaluationSpec, ExecutionSpec
from multi_agent.evaluation import portfolio

SPEC = {'op': 'ts_pct_change', 'n': 10, 'inputs': [{'field': 'close'}]}


def _run(view, hold_days=5, top_n=10, cost=18.0):
    factor = dsl.compile_factor(SPEC, view.panels)
    score = factor.where(view.mask).rank(axis=1, pct=True)
    return portfolio.tranche_backtest(score, view, ExecutionSpec(
        hold_days=hold_days, cost_bps_round_trip=cost), top_n=top_n, direction='high')


def test_no_leverage(dev_view):
    res = _run(dev_view, hold_days=10, top_n=30)
    assert res.turnover.max() <= 1.0 / 10 + 1e-9, '每日新开仓权重不得超过 1/H'
    assert res.daily_net.abs().max() < 0.2, '日收益异常放大说明权重没按笔均分'


def test_cost_is_turnover_times_rate(dev_view):
    res = _run(dev_view, hold_days=20, cost=18.0)
    expected = res.turnover * 18.0 / 1e4
    pd.testing.assert_series_equal(res.daily_cost, expected, check_names=False,
                                   rtol=1e-9, atol=1e-12)


def test_all_calendar_days_kept(dev_view):
    res = _run(dev_view)
    assert len(res.daily_net) == len(dev_view.dates)
    assert not res.daily_net.isna().any()


def test_empty_days_are_zero_not_dropped(dev_view):
    """前 H 天还没有持仓，收益应为 0 而不是缺失。"""
    res = _run(dev_view, hold_days=10)
    assert (res.daily_net.iloc[:2] == 0).all()


def test_reproducible(dev_view):
    a = _run(dev_view, hold_days=5)
    b = _run(dev_view, hold_days=5)
    pd.testing.assert_series_equal(a.daily_net, b.daily_net)


def test_top_n_clipped_when_universe_too_small(dev_view):
    """top_n 超过当日可选标的 20% 时按 20% 截断，避免"买整池"被当成选股。"""
    res = _run(dev_view, top_n=10_000)
    assert res.avg_positions <= 0.2 * dev_view.mask.sum(axis=1).max() * 2


def test_untradable_symbols_are_skipped(dev_view):
    mask = dev_view.mask.copy()
    mask.iloc[:] = False
    view = type(dev_view)(dev_view.snapshot_id + '|none', dev_view.panels, mask,
                          dev_view.benchmark, dev_view.meta)
    res = _run(view)
    assert res.n_trades == 0
    assert (res.daily_net == 0).all()


def test_direction_flips_selection(dev_view):
    factor = dsl.compile_factor(SPEC, dev_view.panels)
    high = portfolio.evaluate_candidate(factor, dev_view, ExecutionSpec(hold_days=5),
                                        EvaluationSpec(top_n=10), direction='high')
    low = portfolio.evaluate_candidate(factor, dev_view, ExecutionSpec(hold_days=5),
                                       EvaluationSpec(top_n=10), direction='low')
    assert high['summary']['sharpe'] != low['summary']['sharpe']


def test_judge_reports_reasons(dev_view):
    factor = dsl.compile_factor(SPEC, dev_view.panels)
    strict = EvaluationSpec(top_n=10, pass_rules={'excess_sharpe': 99.0},
                            min_days=10_000)
    result = portfolio.evaluate_candidate(factor, dev_view, ExecutionSpec(hold_days=5),
                                          strict, direction='high')
    assert result['verdict'] == 'reject'
    assert any('有效天数' in r for r in result['reasons'])
    assert any('excess_sharpe' in r for r in result['reasons'])

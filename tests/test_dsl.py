"""DSL 白名单、编译、去重与时间安全检查。"""
import numpy as np
import pandas as pd
import pytest

from multi_agent import dsl

CLOSE_5D = {'op': 'ts_pct_change', 'n': 5, 'inputs': [{'field': 'close'}]}


def test_validate_accepts_simple_expression():
    assert dsl.validate(CLOSE_5D) == []


@pytest.mark.parametrize('spec', [
    {'op': 'ts_pct_change', 'n': 5, 'inputs': [{'field': 'not_a_field'}]},
    {'op': 'not_an_op', 'inputs': [{'field': 'close'}]},
    {'op': 'ts_pct_change', 'n': 0, 'inputs': [{'field': 'close'}]},
    {'op': 'ts_pct_change', 'n': 999, 'inputs': [{'field': 'close'}]},
    {'op': 'ts_pct_change', 'n': 5.5, 'inputs': [{'field': 'close'}]},
    {'op': 'mul', 'inputs': [{'field': 'close'}]},               # 参数数量不够
    {'op': 'ts_rolling', 'n': 20, 'agg': 'nope', 'inputs': [{'field': 'close'}]},
    {'field': 'close', 'n': 5},                                   # 字段节点带多余键
])
def test_validate_rejects_invalid(spec):
    assert dsl.validate(spec), f'应判为非法: {spec}'


def test_negative_shift_is_impossible():
    """shift 只接受正窗口，负值直接抛错（不存在"把未来搬到今天"的写法）。"""
    with pytest.raises(ValueError):
        dsl._ts_shift(pd.DataFrame({'a': [1.0, 2.0, 3.0]}), -1)


def test_compile_matches_pandas(view):
    spec = {'op': 'ts_rolling', 'n': 20, 'agg': 'std', 'inputs': [{'field': 'close'}]}
    got = dsl.compile_factor(spec, view.panels)
    expect = view.panels['close'].rolling(20, min_periods=10).std()
    pd.testing.assert_frame_equal(got, expect)


def test_warmup_grows_with_nesting():
    simple = dsl.warmup(CLOSE_5D)
    nested = dsl.warmup({'op': 'ts_rolling', 'n': 20, 'agg': 'mean',
                         'inputs': [CLOSE_5D]})
    assert nested > simple


def test_family_ignores_window_but_not_field():
    other_window = {'op': 'ts_pct_change', 'n': 60, 'inputs': [{'field': 'close'}]}
    other_field = {'op': 'ts_pct_change', 'n': 5, 'inputs': [{'field': 'amount'}]}
    assert dsl.family_of(CLOSE_5D) == dsl.family_of(other_window)
    assert dsl.family_of(CLOSE_5D) != dsl.family_of(other_field)
    assert dsl.exact_key(CLOSE_5D) != dsl.exact_key(other_window)


def test_canonical_is_stable():
    shuffled = {'inputs': [{'field': 'close'}], 'n': 5, 'op': 'ts_pct_change'}
    assert dsl.canonical(CLOSE_5D) == dsl.canonical(shuffled)


def test_time_safety_flags_fundamental_fields():
    spec = {'op': 'ts_pct_change', 'n': 4, 'inputs': [{'field': 'net_profit_ttm'}]}
    assert dsl.check_time_safety(spec), '基本面字段应提示需要 PIT 对齐'
    assert dsl.check_time_safety(CLOSE_5D) == []


def test_complexity_and_depth():
    deep = {'op': 'neg', 'inputs': [{'op': 'neg', 'inputs': [
        {'op': 'neg', 'inputs': [{'field': 'close'}]}]}]}
    assert dsl.complexity(deep) == 4
    assert dsl.max_depth(deep) == 4


def test_compiled_factor_coverage(view):
    factor = dsl.compile_factor(CLOSE_5D, view.panels)
    compiled = dsl.CompiledFactor('x', factor, dsl.warmup(CLOSE_5D),
                                  dsl.complexity(CLOSE_5D), ('close',))
    cov = compiled.coverage(view.mask)
    assert 0.0 <= cov <= 1.0


RICH = {'op': 'mul', 'inputs': [
    {'op': 'cs_rank', 'inputs': [{'op': 'ts_pct_change', 'n': 20,
                                  'inputs': [{'field': 'close'}]}]},
    {'op': 'ts_corr', 'n': 20, 'inputs': [{'field': 'close'}, {'field': 'volume'}]},
]}


def test_design_report_accepts_rich_expression():
    report = dsl.design_report(RICH)
    assert report.ok, report.failures
    assert 'cross_section' in report.categories
    assert report.has_relative


def test_design_report_rejects_single_op_expression():
    report = dsl.design_report(CLOSE_5D)
    assert not report.ok
    assert any('算子类别' in f for f in report.failures)
    assert any('相对算子' in f for f in report.failures)


def test_design_report_flags_root_linear_blend():
    blend = {'op': 'add', 'inputs': [
        RICH,
        {'op': 'mul', 'inputs': [{'field': 'close'},
                                 {'op': 'ts_rolling', 'n': 20, 'agg': 'mean',
                                  'inputs': [{'field': 'volume'}]}]},
    ]}
    report = dsl.design_report(blend)
    assert report.is_linear_blend
    assert any('线性叠加' in f for f in report.failures)


def test_design_report_allows_product_of_two_trees():
    """相乘/相除是交互，不算线性叠加。"""
    product = {'op': 'div', 'inputs': [RICH, {'op': 'ts_pct_change', 'n': 10,
                                              'inputs': [{'field': 'amount'}]}]}
    assert not dsl.design_report(product).is_linear_blend


# ─────────────────── 新增算子 ───────────────────

def _panel(seed=0, rows=40, cols=2):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(rows, cols)),
                        index=pd.date_range('2020-01-01', periods=rows),
                        columns=['a', 'b'])


def test_ts_slope_matches_polyfit():
    x = _panel()
    got = dsl.compile_factor({'op': 'ts_slope', 'n': 10, 'inputs': [{'field': 'close'}]},
                             {'close': x})
    w = x['a'].iloc[21:31].values
    assert np.isclose(got['a'].iloc[30], np.polyfit(np.arange(10), w, 1)[0])


def test_decay_linear_weights_recent_more():
    x = _panel(seed=1)
    got = dsl.compile_factor({'op': 'decay_linear', 'n': 5,
                              'inputs': [{'field': 'close'}]}, {'close': x})
    w = x['a'].iloc[30:35].values
    assert np.isclose(got['a'].iloc[34], (w * np.arange(1, 6)).sum() / 15)
    # 近端更高权重：把最后一天拉大，结果应高于等权均值
    spiked = x.copy()
    spiked.iloc[34, 0] += 10
    spiked_got = dsl.compile_factor({'op': 'decay_linear', 'n': 5,
                                     'inputs': [{'field': 'close'}]}, {'close': spiked})
    plain = spiked['a'].iloc[30:35].mean()
    assert spiked_got['a'].iloc[34] > plain


def test_ts_slope_requires_full_window():
    x = _panel()
    x.iloc[32, 0] = np.nan
    got = dsl.compile_factor({'op': 'ts_slope', 'n': 10, 'inputs': [{'field': 'close'}]},
                             {'close': x})
    assert np.isnan(got['a'].iloc[40 - 3]) or np.isnan(got['a'].iloc[33])


def test_ts_quantile_matches_pandas():
    x = _panel(seed=2)
    got = dsl.compile_factor({'op': 'ts_quantile', 'n': 10, 'q': 0.3,
                              'inputs': [{'field': 'close'}]}, {'close': x})
    assert np.isclose(got['a'].iloc[30], x['a'].iloc[21:31].quantile(0.3))


def test_ts_zscore_uses_trailing_window():
    x = _panel(seed=3)
    got = dsl.compile_factor({'op': 'ts_zscore', 'n': 10, 'inputs': [{'field': 'close'}]},
                             {'close': x})
    w = x['a'].iloc[21:31]
    assert np.isclose(got['a'].iloc[30], (x['a'].iloc[30] - w.mean()) / w.std())


def test_cs_winsorize_clips_per_date():
    x = _panel(seed=4)
    x.iloc[0, 0] = 100.0
    got = dsl.compile_factor({'op': 'cs_winsorize', 'q': 0.1,
                              'inputs': [{'field': 'close'}]}, {'close': x})
    assert got.iloc[0, 0] < 100.0
    assert got.abs().max(axis=1).le(x.abs().max(axis=1) + 1e-12).all()


def test_inverse_and_cs_scale():
    x = _panel(seed=5) + 3.0
    inv = dsl.compile_factor({'op': 'inverse', 'inputs': [{'field': 'close'}]}, {'close': x})
    assert np.isclose(inv['a'].iloc[0], 1 / x['a'].iloc[0])
    scaled = dsl.compile_factor({'op': 'cs_scale', 'inputs': [{'field': 'close'}]},
                                {'close': x})
    assert np.isclose(scaled.iloc[0].abs().sum(), 1.0)


@pytest.mark.parametrize('spec', [
    {'op': 'ts_quantile', 'n': 10, 'q': 1.5, 'inputs': [{'field': 'close'}]},
    {'op': 'ts_quantile', 'n': 10, 'inputs': [{'field': 'close'}]},
    {'op': 'cs_winsorize', 'q': 0.7, 'inputs': [{'field': 'close'}]},
    {'op': 'ts_slope', 'n': 2, 'inputs': [{'field': 'close'}]},
])
def test_new_operator_validation(spec):
    assert dsl.validate(spec)


def test_rolling_accepts_kurt(view):
    spec = {'op': 'ts_rolling', 'n': 20, 'agg': 'kurt', 'inputs': [{'field': 'close'}]}
    assert dsl.validate(spec) == []
    dsl.compile_factor(spec, view.panels)


# ─────────────────── 极值位置 / 分组中性化 / 条件算子 ───────────────────

def _group_panel(n_symbols=60, rows=30, n_industries=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2020-01-01', periods=rows)
    cols = [f'S{i:02d}' for i in range(n_symbols)]
    close = pd.DataFrame(rng.normal(size=(rows, n_symbols)).cumsum(0), index=idx, columns=cols)
    mv = pd.DataFrame(np.exp(rng.normal(23, 1, size=(rows, n_symbols))), index=idx, columns=cols)
    industry = pd.DataFrame(
        np.tile([i % n_industries for i in range(n_symbols)], (rows, 1)).astype(float),
        index=idx, columns=cols)
    amount = pd.DataFrame(rng.random((rows, n_symbols)), index=idx, columns=cols)
    return {'close': close, 'mv': mv, 'industry_code': industry, 'amount': amount,
            'volume': amount * 10}


def test_ts_argmax_argmin_positions():
    panels = _group_panel(rows=20)
    close = panels['close']
    got = dsl.compile_factor({'op': 'ts_argmax', 'n': 10, 'inputs': [{'field': 'close'}]}, panels)
    w = close.iloc[10:20, 0].values
    assert np.isclose(got.iloc[19, 0], w.argmax() / 9)
    low = dsl.compile_factor({'op': 'ts_argmin', 'n': 10, 'inputs': [{'field': 'close'}]}, panels)
    assert np.isclose(low.iloc[19, 0], w.argmin() / 9)
    assert got.iloc[:9].isna().all().all(), '窗口不足时应为 NaN'


def test_ts_argmax_requires_full_window():
    panels = _group_panel(rows=20)
    panels['close'].iloc[15, 1] = np.nan
    got = dsl.compile_factor({'op': 'ts_argmax', 'n': 10, 'inputs': [{'field': 'close'}]}, panels)
    assert np.isnan(got.iloc[19, 1])


def test_group_demean_is_within_group():
    panels = _group_panel()
    got = dsl.compile_factor(
        {'op': 'cs_group_demean', 'by': 'industry_code', 'inputs': [{'field': 'close'}]}, panels)
    row, ind = got.iloc[20], panels['industry_code'].iloc[20]
    for code in ind.unique():
        assert abs(row[ind == code].mean()) < 1e-12


def test_group_rank_is_bounded_and_grouped():
    panels = _group_panel()
    got = dsl.compile_factor(
        {'op': 'cs_group_rank', 'by': 'industry_code', 'inputs': [{'field': 'close'}]}, panels)
    assert got.min().min() >= 0 and got.max().max() <= 1
    row, ind = got.iloc[20], panels['industry_code'].iloc[20]
    # 组内最大值应为 1
    for code in ind.unique():
        assert np.isclose(row[ind == code].max(), 1.0)


def test_regress_multi_orthogonalizes_industry_and_style():
    panels = _group_panel()
    got = dsl.compile_factor(
        {'op': 'cs_regress_multi', 'by': ['industry_code', 'mv'],
         'inputs': [{'field': 'close'}]}, panels)
    row, ind = got.iloc[20], panels['industry_code'].iloc[20]
    assert row.notna().all()
    for code in ind.unique():
        assert abs(row[ind == code].mean()) < 1e-8
    assert abs(np.corrcoef(row, panels['mv'].iloc[20])[0, 1]) < 1e-9


def test_regress_multi_needs_enough_names():
    panels = _group_panel(n_symbols=6)
    got = dsl.compile_factor(
        {'op': 'cs_regress_multi', 'by': ['industry_code'], 'inputs': [{'field': 'close'}]}, panels)
    assert got.isna().all().all(), '标的太少时应放弃回归而不是给假残差'


def test_where_filters_by_condition():
    """where(cond, x) 的语义是 cond > 0，所以这里把阈值折进条件里。"""
    panels = _group_panel()
    cond_spec = {'op': 'sub', 'inputs': [{'field': 'amount'},
                                        {'op': 'mul', 'inputs': [
                                            {'op': 'cs_demean', 'inputs': [{'field': 'amount'}]},
                                            {'op': 'sign', 'inputs': [
                                                {'op': 'cs_demean', 'inputs': [{'field': 'amount'}]}]}]}]}
    cond_values = dsl.compile_factor(cond_spec, panels)
    got = dsl.compile_factor({'op': 'where', 'inputs': [cond_spec, {'field': 'close'}]}, panels)
    assert (got.notna() == (cond_values > 0)).all().all()
    assert got.notna().any().any() and not got.notna().all().all()


def test_trade_when_forward_fills():
    """trigger>0 时更新为当日信号，否则沿用最近一次的值。"""
    panels = _group_panel(rows=12)
    trigger = {'op': 'sub', 'inputs': [
        {'field': 'amount'},
        {'op': 'ts_rolling', 'n': 5, 'agg': 'mean', 'inputs': [{'field': 'amount'}]},
    ]}
    cond = dsl.compile_factor(trigger, panels)
    got = dsl.compile_factor({'op': 'trade_when', 'inputs': [trigger, {'field': 'close'}]},
                             panels)
    updates = cond > 0
    assert np.allclose(got[updates].values, panels['close'][updates].values, equal_nan=True)
    # 非触发日必须是前向填充（等于最近一次触发日的值），而不是当日收盘价
    stale = updates.shift(1, fill_value=False) & ~updates
    assert got.notna().sum().sum() >= updates.sum().sum()
    assert not np.allclose(got[stale].values, panels['close'][stale].values, equal_nan=True)


@pytest.mark.parametrize('spec', [
    {'op': 'cs_group_demean', 'by': 'not_a_field', 'inputs': [{'field': 'close'}]},
    {'op': 'cs_group_demean', 'inputs': [{'field': 'close'}]},
    {'op': 'cs_regress_multi', 'by': 'mv', 'inputs': [{'field': 'close'}]},
    {'op': 'cs_group_rank', 'by': ['industry_code'], 'inputs': [{'field': 'close'}]},
])
def test_group_op_validation(spec):
    assert dsl.validate(spec)


def test_fields_of_includes_group_reference():
    spec = {'op': 'cs_group_rank', 'by': 'industry_code', 'inputs': [{'field': 'close'}]}
    assert dsl.fields_of(spec) == ['close', 'industry_code']


def test_conditional_category_counts_as_diversity():
    spec = {'op': 'mul', 'inputs': [
        {'op': 'where', 'inputs': [{'op': 'where_pos', 'inputs': [{'field': 'amount'}]},
                                   {'op': 'cs_rank', 'inputs': [
                                       {'op': 'ts_pct_change', 'n': 10,
                                        'inputs': [{'field': 'close'}]}]}]},
        {'op': 'ts_corr', 'n': 20, 'inputs': [{'field': 'close'}, {'field': 'volume'}]},
    ]}
    report = dsl.design_report(spec)
    assert 'conditional' in report.categories
    assert report.has_relative


def test_synthetic_view_carries_industry_codes(view):
    assert 'industry_code' in view.panels
    spec = {'op': 'cs_group_demean', 'by': 'industry_code',
            'inputs': [{'op': 'ts_pct_change', 'n': 20, 'inputs': [{'field': 'close'}]}]}
    got = dsl.compile_factor(spec, view.panels)
    assert got.notna().any().any()

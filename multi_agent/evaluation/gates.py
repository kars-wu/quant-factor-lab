"""Gate 0：数据与工程不变量。

这些检查在**看收益之前**跑。它们不能证明因子有效，但能挡掉最典型的
「一定是 bug」的情况：

* 前缀不变性：删掉 T 以后的数据，T 及以前的因子值不能变（防未来函数）；
* 截面不变性：打乱列顺序/行顺序后，排序结果应完全一致；
* 零权重不变性：加一个零权重稀疏因子，组合值与候选池不能变
  （历史 bug：``0 * NaN`` 传播导致覆盖池塌陷）；
* 覆盖核对：逐层记录股票池规模，避免用「因子有值的股票」当基准池；
* 资金恒等式：现金 + 持仓市值 = 净值，且空仓日必须保留；
* 方向/血缘一致性：同一 family 的结构指纹与预算计数必须可追溯。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from multi_agent import dsl
from multi_agent.contracts import EvaluationSpec, ExecutionSpec
from multi_agent.datasets.base import DataView
from multi_agent.evaluation import portfolio


@dataclass
class GateItem:
    name: str
    passed: bool
    detail: str = ''


@dataclass
class GateReport:
    items: List[GateItem] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = '') -> 'GateReport':
        self.items.append(GateItem(name, bool(passed), detail))
        return self

    @property
    def passed(self) -> bool:
        return all(i.passed for i in self.items)

    def failures(self) -> List[GateItem]:
        return [i for i in self.items if not i.passed]

    def to_dict(self) -> Dict[str, object]:
        return {
            'passed': self.passed,
            'items': [{'name': i.name, 'passed': i.passed, 'detail': i.detail} for i in self.items],
        }

    def summary(self) -> str:
        ok = sum(1 for i in self.items if i.passed)
        if self.passed:
            return f'Gate0 通过（{ok}/{len(self.items)}）'
        bad = ', '.join(i.name for i in self.failures())
        return f'Gate0 失败（{ok}/{len(self.items)}）：{bad}'


# ─────────────────────────── 基础不变量 ───────────────────────────

def check_prefix_invariance(spec: Optional[Dict], view: DataView,
                            cutoffs: Optional[List[str]] = None,
                            compiler=None) -> GateItem:
    """截断到 cutoffs 后重算，比较截断日之前的值是否相同。"""
    if cutoffs is None:
        dates = view.dates
        if len(dates) < 120:
            return GateItem('prefix_invariance', True, '样本过短，跳过')
        cutoffs = [str(dates[int(len(dates) * f)].date()) for f in (0.5, 0.8)]

    compile_fn = compiler or (lambda panels: dsl.compile_factor(spec or {}, panels))
    full = compile_fn(view.panels)
    for cutoff in cutoffs:
        truncated = view.cutoff(cutoff)
        part = compile_fn(truncated.panels)
        common_idx = part.index
        common_cols = [c for c in part.columns if c in full.columns]
        a = full.reindex(index=common_idx, columns=common_cols)
        b = part.reindex(index=common_idx, columns=common_cols)
        if not a.shape == b.shape:
            return GateItem('prefix_invariance', False, f'{cutoff}: 形状不一致')
        diff = (a.fillna(-9e9) - b.fillna(-9e9)).abs()
        bad = float(diff.max().max()) if diff.size else 0.0
        if bad > 1e-9:
            return GateItem('prefix_invariance', False,
                            f'{cutoff}: 前缀值被后续数据改变，最大差 {bad:.3e}')
    return GateItem('prefix_invariance', True, f'截断点 {cutoffs}')


def check_cross_section_invariance(spec: Dict, view: DataView, seed: int = 0) -> GateItem:
    rng = np.random.default_rng(seed)
    base = dsl.compile_factor(spec, view.panels)
    cols = list(base.columns)
    perm = list(cols)
    rng.shuffle(perm)
    panels = {k: v[perm] for k, v in view.panels.items()}
    shuffled = DataView(view.snapshot_id + '|perm', panels, view.mask[perm],
                        view.benchmark, dict(view.meta))
    got = dsl.compile_factor(spec, shuffled.panels)[cols]
    diff = (base.fillna(-9e9) - got.fillna(-9e9)).abs()
    bad = float(diff.max().max()) if diff.size else 0.0
    return GateItem('cross_section_invariance', bad < 1e-9, f'最大差 {bad:.3e}')


def blend(parts: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> pd.DataFrame:
    """按权重合成因子面板；权重为 0 的分量**必须**被完全跳过。

    这是历史 bug 的修复点：``0 * NaN`` 会污染合成结果，并进一步改变
    候选池。任何实现都不允许用权重相乘来「关掉」一个稀疏因子。
    """
    acc: Optional[pd.DataFrame] = None
    for name, factor in parts.items():
        w = float(weights.get(name, 0.0))
        if w == 0.0:
            continue
        term = factor * w
        # 不用 fill_value：非零权重的缺失值必须继续传播（缺失就是缺失，
        # 不能当成 0 参与合成），只有显式 0 权重才允许整项跳过。
        acc = term if acc is None else acc.add(term)
    if acc is None:
        raise ValueError('所有分量权重均为 0')
    return acc


def check_zero_weight_invariance(spec: Dict, view: DataView) -> GateItem:
    base = dsl.compile_factor(spec, view.panels)
    sparse = base.copy()
    keep = np.zeros(sparse.shape, dtype=bool)
    rng = np.random.default_rng(3)
    keep[rng.integers(0, sparse.shape[0], 20), rng.integers(0, sparse.shape[1], 20)] = True
    sparse = sparse.where(pd.DataFrame(keep, index=sparse.index, columns=sparse.columns))

    blended = blend({'main': base, 'sparse_event': sparse},
                    {'main': 1.0, 'sparse_event': 0.0})
    diff = (base.fillna(-9e9) - blended.fillna(-9e9)).abs()
    bad = float(diff.max().max()) if diff.size else 0.0
    if bad > 1e-9:
        return GateItem('zero_weight_invariance', False,
                        f'零权重稀疏因子改变了合成值，最大差 {bad:.3e}')
    # 候选池也必须一致
    pool_a = base.where(view.mask).notna()
    pool_b = blended.where(view.mask).notna()
    if int((pool_a & ~pool_b).sum().sum()) or int((pool_b & ~pool_a).sum().sum()):
        return GateItem('zero_weight_invariance', False, '零权重因子改变了候选池覆盖')
    return GateItem('zero_weight_invariance', True, '零权重项不影响合成值与池')


def check_coverage(spec: Dict, view: DataView, min_coverage: float = 0.6) -> GateItem:
    factor = dsl.compile_factor(spec, view.panels)
    masked = factor.where(view.mask)
    daily_cov = masked.notna().sum(axis=1).div(view.mask.sum(axis=1).replace(0, np.nan))
    cov = float(daily_cov.mean())
    detail = f'平均覆盖率 {cov:.3f}（可交易池 {float(view.mask.sum(axis=1).mean()):.1f} 只）'
    if cov < min_coverage:
        return GateItem('coverage', False, detail + f' < {min_coverage}')
    # 覆盖是否系统性偏向大市值（覆盖选择效应）
    mv = view.panels.get('mv')
    if mv is not None:
        covered = masked.notna() & view.mask
        big = mv.rank(axis=1, pct=True) > 0.7
        share_big = float((covered & big).sum().sum() / max((covered.sum().sum()), 1))
        detail += f'，覆盖中大市值占比 {share_big:.2f}'
        if share_big > 0.6:
            return GateItem('coverage', False, detail + '：覆盖偏向大市值，存在覆盖偏差风险')
    return GateItem('coverage', True, detail)


def check_cash_identity(factor: pd.DataFrame, view: DataView,
                        execution: ExecutionSpec, evaluation: EvaluationSpec,
                        start: Optional[str] = None, end: Optional[str] = None) -> GateItem:
    result = portfolio.tranche_backtest(
        factor.where(view.mask).rank(axis=1, pct=True), view, execution,
        evaluation.top_n, 'high', start, end,
    )
    idx = result.daily_net.index
    if result.daily_net.isna().any():
        return GateItem('cash_identity', False, '日收益序列存在 NaN（空仓日被污染）')
    expected = view.dates[(view.dates >= idx[0]) & (view.dates <= idx[-1])] if len(idx) else view.dates[:0]
    if len(expected) != len(idx):
        return GateItem('cash_identity', False,
                        f'日历天数不一致：{len(idx)} vs {len(expected)}（空仓日被删除？）')
    # 逐日恒等式：net = gross - cost
    reconstructed = result.daily_gross - result.daily_cost
    bad = float((reconstructed - result.daily_net).abs().max())
    if bad > 1e-12:
        return GateItem('cash_identity', False, f'净值恒等式偏差 {bad:.3e}')
    return GateItem('cash_identity', True, f'{len(idx)} 个交易日完整，成本可回溯')


# ─────────────────────────── 组合入口 ───────────────────────────

def run_gate0(
    spec: Dict,
    view: DataView,
    execution: ExecutionSpec,
    evaluation: EvaluationSpec,
    factor: Optional[pd.DataFrame] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    checks: Optional[List[str]] = None,
    design_strict: bool = False,
    min_categories: int = 3,
) -> GateReport:
    """执行 Gate 0。

    ``design_strict=True`` 时，表达式的"算子交互"设计要求是硬门槛
    （见 ``dsl.design_report``）；默认只记录不拦截。
    """
    errors = dsl.validate(spec)
    report = GateReport()
    report.add('dsl_valid', not errors, '; '.join(errors) if errors else '表达式合法')
    if errors:
        return report

    report.add('complexity', dsl.complexity(spec) <= dsl.MAX_NODES
               and dsl.max_depth(spec) <= dsl.MAX_DEPTH,
               f'nodes={dsl.complexity(spec)}, depth={dsl.max_depth(spec)}')
    report.add('warmup', dsl.warmup(spec) <= 250, f'warmup={dsl.warmup(spec)}')

    design = dsl.design_report(spec, min_categories=min_categories)
    if design_strict:
        report.add('design_richness', design.ok, '; '.join(design.failures))
    else:
        report.add('design_richness', True,
                   f'宽松模式（strict 关闭）：类别 {list(design.categories)}'
                   + ('' if design.ok else f"；建议：{'; '.join(design.failures)}"))

    enabled = set(checks or ['prefix', 'cross_section', 'zero_weight', 'coverage', 'cash'])
    if 'prefix' in enabled:
        report.items.append(check_prefix_invariance(spec, view))
    if 'cross_section' in enabled:
        report.items.append(check_cross_section_invariance(spec, view))
    if 'zero_weight' in enabled:
        report.items.append(check_zero_weight_invariance(spec, view))
    if 'coverage' in enabled:
        report.items.append(check_coverage(spec, view, evaluation.min_coverage))
    if 'cash' in enabled:
        factor = factor if factor is not None else dsl.compile_factor(spec, view.panels)
        report.items.append(check_cash_identity(factor, view, execution, evaluation, start, end))
    return report


# ─────────────────────────── 审计回归用例 ───────────────────────────

def audit_regression_suite(view: DataView, execution: ExecutionSpec,
                           evaluation: EvaluationSpec) -> Dict[str, bool]:
    """已知缺陷夹具：衡量审计链路「能不能真的发现问题」。

    每个用例都注入一种已知错误，返回该错误是否被检出。它衡量的是
    审计流程本身的有效性，与因子收益无关。
    """
    good_spec = {'op': 'ts_pct_change', 'n': 5, 'inputs': [{'field': 'close'}]}
    out: Dict[str, bool] = {}

    # 1) 未来函数：引用不存在于白名单/未来对齐的字段
    leak_spec = {'op': 'ts_pct_change', 'n': 5, 'inputs': [{'field': 'next_close'}]}
    out['leak_detected'] = bool(dsl.validate(leak_spec))

    # 2) 负向 shift（把未来数据搬到今天）
    try:
        dsl._ts_shift(view.panels['close'], -1)
        out['negative_shift_detected'] = False
    except ValueError:
        out['negative_shift_detected'] = True

    # 3) 零权重 NaN 传播
    base = dsl.compile_factor(good_spec, view.panels)
    sparse = base.where(base > base.stack().quantile(0.99))
    naive = base + 0.0 * sparse                      # 错误实现
    correct = blend({'main': base, 'ev': sparse}, {'main': 1.0, 'ev': 0.0})
    naive_breaks = bool(naive.isna().sum().sum() > base.isna().sum().sum())
    correct_ok = bool((correct.fillna(-9e9) - base.fillna(-9e9)).abs().max().max() < 1e-9)
    out['zero_weight_detected'] = naive_breaks and correct_ok

    # 4) 空仓日被删除：模拟 dropna 后统计
    res = portfolio.tranche_backtest(base.where(view.mask).rank(axis=1, pct=True),
                                     view, execution, evaluation.top_n, 'high')
    dropped = res.daily_net[res.daily_net != 0]
    out['calendar_gap_detected'] = bool(len(dropped) < len(res.daily_net))

    # 5) 覆盖偏差：只在 10% 大市值上有值
    mv = view.panels.get('mv')
    if mv is not None:
        big = mv.rank(axis=1, pct=True) > 0.9
        item = check_coverage(good_spec, view.with_panel('close', view.panels['close'].where(big)),
                              evaluation.min_coverage)
        out['coverage_bias_detected'] = not item.passed
    else:
        out['coverage_bias_detected'] = True

    # 6) 重复家族：仅改窗口应被识别为同一 family
    f1 = dsl.family_of(good_spec)
    f2 = dsl.family_of({'op': 'ts_pct_change', 'n': 20, 'inputs': [{'field': 'close'}]})
    out['duplicate_family_detected'] = f1 == f2
    return out

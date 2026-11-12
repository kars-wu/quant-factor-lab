"""执行与组合模拟：H 日持仓重叠（tranche）口径。

时序（与 ``ExecutionSpec`` 一致）：

    T 收盘  得到信号（只使用 T 及以前的数据）
    T+1 开盘 买入，资金权重 1/hold_days
    T+2..T+H 继续持有
    T+1+H 开盘 卖出该笔

每笔 tranche 独立记账，日收益 = Σ(权重 × 标的 open-to-open 日收益)，
未投入的权重按现金处理（收益 0，不做再归一化），因此不会把空仓日从
净值序列里删掉。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from multi_agent.contracts import EvaluationSpec, ExecutionSpec
from multi_agent.datasets.base import DataView
from multi_agent.evaluation import metrics


@dataclass
class BacktestResult:
    daily_net: pd.Series
    daily_gross: pd.Series
    daily_cost: pd.Series
    turnover: pd.Series
    held_ratio: pd.Series
    n_trades: int
    avg_positions: float
    avg_signal_participation: Optional[float] = None
    n_untradable: int = 0
    extra: Dict[str, float] = field(default_factory=dict)

    def summary(self, benchmark: Optional[pd.Series] = None,
                evaluation: Optional[EvaluationSpec] = None) -> Dict[str, float]:
        out = dict(metrics.stats(self.daily_net))
        if benchmark is not None:
            out.update(metrics.excess_stats(self.daily_net, benchmark))
            out['positive_years'] = float(
                sum(1 for v in metrics.yearly_active(self.daily_net, benchmark).values()
                    if v['active_annual_return'] > 0)
            )
            out['total_years'] = float(len(metrics.yearly_active(self.daily_net, benchmark)))
        out['turnover_annual'] = round(float(self.turnover.mean() * 244), 4)
        out['avg_held_ratio'] = round(float(self.held_ratio.mean()), 4)
        out['n_trades'] = int(self.n_trades)
        out['avg_positions'] = round(float(self.avg_positions), 2)
        if self.avg_signal_participation is not None:
            out['avg_signal_participation'] = round(float(self.avg_signal_participation), 5)
        out.update(self.extra)
        return out


def _rank_picks(row: pd.Series, tradable: pd.Series, top_n: int,
                direction: str) -> pd.Index:
    scores = row.reindex(tradable.index).dropna()
    if scores.empty:
        return pd.Index([])
    scores = scores[tradable.reindex(scores.index).fillna(False).astype(bool)]
    if scores.empty:
        return pd.Index([])
    # 若 top_n 超过当日可选标的的 20%，实际上就变成"买整池"，
    # 截面选股意义消失。这里按 20% 截断，避免用错池子还当成选股结果。
    k = max(1, min(int(top_n), int(len(scores) * 0.2)))
    ascending = direction == 'low'
    return scores.sort_values(ascending=ascending).head(k).index


def tranche_backtest(
    score: pd.DataFrame,
    view: DataView,
    execution: ExecutionSpec,
    top_n: int = 100,
    direction: str = 'high',
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> BacktestResult:
    """按 H 日 tranche 口径回测一个打分面板。"""
    open_w = view.panels['open']
    close_w = view.panels['close']
    amount_w = view.amount()

    dates = view.dates
    n_dates, n_syms = open_w.shape
    score_w = score.reindex(index=dates, columns=open_w.columns).astype(float)
    mask = view.mask.reindex(index=dates, columns=open_w.columns).fillna(False).astype(bool)

    # open-to-open 日收益：ret[d] = open[d+1] / open[d] - 1
    ret_oo = (open_w.shift(-1) / open_w - 1)

    H = max(int(execution.hold_days), 1)
    cost_rate = execution.cost_bps_round_trip / 1e4

    weights = np.zeros((n_dates, n_syms))
    new_weight = np.zeros(n_dates)      # 当日新开仓的名义权重 = 换手
    cost_vec = np.zeros(n_dates)
    trade_count = 0
    untradable = 0
    participation = []

    # 每日新开一笔 tranche，占总资金 1/H；这笔钱在该 tranche 内等权分给入选标的
    tranche_weight = 1.0 / H

    for t in range(n_dates - 1):
        entry = t + 1
        if entry >= n_dates:
            break
        # 只允许使用 T 收盘及以前的信息；T+1 是否可交易属于执行事实
        tradable_t = mask.iloc[t]
        tradable_entry = mask.iloc[entry] & (open_w.iloc[entry] > 0)
        picks = _rank_picks(score_w.iloc[t], tradable_t, top_n, direction)
        if len(picks) == 0:
            continue

        closed_gap = (open_w.iloc[entry].reindex(picks) /
                      close_w.iloc[t].reindex(picks) - 1)
        if execution.buy_limit_gap is not None:
            ok = closed_gap < execution.buy_limit_gap
            untradable += int((~ok).sum())
            picks = picks[ok.reindex(picks).fillna(False).values]
            if len(picks) == 0:
                continue

        is_tradable = tradable_entry.reindex(picks).fillna(False).astype(bool).values
        untradable += int((~is_tradable).sum())
        picks = picks[is_tradable]
        if len(picks) == 0:
            continue

        cols = open_w.columns.get_indexer(picks)
        rows = np.arange(entry, min(entry + H, n_dates))
        weight_per_pick = tranche_weight / len(picks)
        weights[np.ix_(rows, cols)] += weight_per_pick
        new_weight[entry] += tranche_weight
        cost_vec[entry] += cost_rate * tranche_weight
        trade_count += len(picks)

        adv = amount_w.iloc[t].reindex(picks).replace(0, np.nan)
        notional = execution.capital * weight_per_pick
        est = (notional / adv).dropna()
        if len(est):
            participation.append(float(est.mean()))

    gross = pd.Series(np.nansum(weights * ret_oo.fillna(0).values, axis=1),
                      index=dates, name='gross')
    cost = pd.Series(cost_vec, index=dates, name='cost')
    net = gross - cost
    # 换手 = 当日新开仓名义权重（不是持仓总权重）
    turnover = pd.Series(new_weight, index=dates)
    held_ratio = pd.Series(np.sum(weights > 0, axis=1) > 0, index=dates).astype(float)

    avg_positions = float((weights > 0).sum(axis=1).mean())

    if start:
        sel = dates >= pd.Timestamp(start)
    else:
        sel = np.ones(n_dates, dtype=bool)
    if end:
        sel = sel & (dates <= pd.Timestamp(end))
    idx = dates[sel]

    # 空仓日保留 0 收益，不 dropna
    return BacktestResult(
        daily_net=net.loc[idx].fillna(0.0),
        daily_gross=gross.loc[idx].fillna(0.0),
        daily_cost=cost.loc[idx].fillna(0.0),
        turnover=turnover.loc[idx].fillna(0.0),
        held_ratio=held_ratio.loc[idx],
        n_trades=trade_count,
        avg_positions=avg_positions,
        avg_signal_participation=float(np.mean(participation)) if participation else None,
        n_untradable=untradable,
    )


def evaluate_candidate(
    factor: pd.DataFrame,
    view: DataView,
    execution: ExecutionSpec,
    evaluation: EvaluationSpec,
    direction: str = 'high',
    start: Optional[str] = None,
    end: Optional[str] = None,
    benchmark: Optional[pd.Series] = None,
) -> Dict[str, object]:
    """标准化评估：返回指标、逐年结果与门槛判定。

    打分口径：因子值先做截面 rank（降低极值影响），再乘方向作为排序键。
    """
    score = factor.where(view.mask)
    score = score.rank(axis=1, pct=True)
    if direction == 'low':
        score = -score

    result = tranche_backtest(score, view, execution, evaluation.top_n,
                              direction='high', start=start, end=end)
    bench = (benchmark if benchmark is not None else view.benchmark)
    bench = bench[(bench.index >= result.daily_net.index[0])
                  & (bench.index <= result.daily_net.index[-1])]
    summary = result.summary(bench, evaluation)

    coverage = float(factor.where(view.mask).notna().sum(axis=1)
                     .div(view.mask.sum(axis=1).replace(0, np.nan)).mean())
    summary['coverage'] = round(coverage, 4)

    yearly = metrics.yearly_active(result.daily_net, bench)
    verdict, reasons = judge(summary, evaluation)
    return {
        'summary': summary,
        'yearly': yearly,
        'verdict': verdict,
        'reasons': reasons,
        'backtest': result,
        'benchmark': bench,
    }


def judge(summary: Dict[str, float], spec: EvaluationSpec) -> tuple[str, list[str]]:
    """按 ``EvaluationSpec.pass_rules`` 判定；所有规则都满足才 pass。"""
    reasons: list[str] = []
    if summary.get('n_days', 0) < spec.min_days:
        reasons.append(f"有效天数 {summary.get('n_days')} < {spec.min_days}")
    if summary.get('coverage', 0) < spec.min_coverage:
        reasons.append(f"覆盖率 {summary.get('coverage')} < {spec.min_coverage}")
    for metric, threshold in spec.pass_rules.items():
        value = summary.get(metric)
        if value is None:
            reasons.append(f'缺少指标 {metric}')
            continue
        if threshold >= 0 and value < threshold:
            reasons.append(f'{metric}={value} < {threshold}')
        if threshold < 0 and value < threshold:
            reasons.append(f'{metric}={value} < {threshold}')
    if spec.yearly_consistency:
        pos = summary.get('positive_years', 0)
        total = summary.get('total_years', 0)
        ratio = (pos / total) if total else 1.0
        if ratio < spec.min_positive_year_ratio:
            reasons.append(f'逐年正超额 {int(pos)}/{int(total)} < '
                           f'{spec.min_positive_year_ratio:.0%}')
    return ('pass' if not reasons else 'reject'), reasons

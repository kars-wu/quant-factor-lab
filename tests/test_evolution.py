"""外层进化：过程指标打分、配对评测、接受/回滚。"""
import pytest

from multi_agent.contracts import EvaluationSpec, ExecutionSpec
from multi_agent.evolution import evaluate_policy, evolve, paired_compare
from multi_agent.policy import ALL_AUDIT_CHECKS, PolicyState


def test_full_checklist_detects_more_than_partial(view):
    full = PolicyState(version='v0', audit_checks=list(ALL_AUDIT_CHECKS))
    partial = PolicyState(version='v0', audit_checks=['coverage'])
    ex, ev = ExecutionSpec(hold_days=5), EvaluationSpec(top_n=10)
    a = evaluate_policy(full, view, ex, ev)
    b = evaluate_policy(partial, view, ex, ev)
    assert a.detection > b.detection
    assert a.score > b.score
    assert a.false_positive == 0.0


def test_paired_compare_accepts_gap_fix(view):
    partial = PolicyState(version='v0', audit_checks=['prefix', 'coverage'])
    fixed = partial.bump('audit_checklist', {'audit_checks': list(ALL_AUDIT_CHECKS)},
                         reason='audit_gap')
    cmp = paired_compare(partial, fixed, view, ExecutionSpec(hold_days=5),
                         EvaluationSpec(top_n=10))
    assert cmp['improved'] is True
    assert cmp['delta'] > 0


def test_paired_compare_rejects_noop(view):
    policy = PolicyState(version='v0', audit_checks=list(ALL_AUDIT_CHECKS))
    noop = policy.bump('proposal', {'tracks': list(reversed(policy.tracks))}, reason='noop')
    cmp = paired_compare(policy, noop, view, ExecutionSpec(hold_days=5),
                         EvaluationSpec(top_n=10))
    assert cmp['improved'] is False
    assert cmp['delta'] == 0


def test_bump_only_allows_one_module_and_bumps_version():
    p = PolicyState(version='v0')
    v1 = p.bump('budget_allocation', {'proposal_quota': 2}, reason='减配额')
    assert v1.version == 'v1'
    assert v1.proposal_quota == 2
    assert v1.history[-1]['module'] == 'budget_allocation'
    with pytest.raises(ValueError):
        p.bump('not_a_module', {'proposal_quota': 1})
    with pytest.raises(ValueError):
        p.bump('proposal', {'no_such_field': 1})


def test_evolve_accepts_then_rolls_back(view, tmp_path):
    traces = [{'candidate_id': 'c1', 'verdict': 'reject',
               'failure_types': ['no_incremental_signal']}]
    result = evolve(traces, view, PolicyState(audit_checks=['prefix']),
                    ExecutionSpec(hold_days=5), EvaluationSpec(top_n=10), rounds=2,
                    out_path=str(tmp_path / 'evo.json'))
    assert result['champion']['audit_checks'] == list(ALL_AUDIT_CHECKS)
    assert result['history'][0]['accepted'] is True
    assert (tmp_path / 'evo.json').exists()


def test_evolve_without_traces_returns_rejected(view):
    result = evolve([], view, PolicyState(), ExecutionSpec(hold_days=5),
                    EvaluationSpec(top_n=10), rounds=1)
    assert result['champion']['version'] == 'v0'
    assert result['history'] == []

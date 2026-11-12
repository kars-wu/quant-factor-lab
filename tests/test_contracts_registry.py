"""契约哈希与台账（不可覆盖、去重、预算、失败记忆）。"""
import json

import pytest

from multi_agent.contracts import (
    Budget,
    CandidateSpec,
    DataSpec,
    EvaluationSpec,
    ExecutionSpec,
    FactorSpec,
    HypothesisCard,
    ResearchProtocol,
)
from multi_agent.registry import MemoryRecord, Registry

SPEC = {'op': 'ts_pct_change', 'n': 5, 'inputs': [{'field': 'close'}]}


def _candidate(**kwargs):
    base = dict(
        candidate_id=kwargs.pop('candidate_id', 'cand_x'),
        experiment_id=kwargs.pop('experiment_id', 'run_1'),
        hypothesis_id='h1',
        family='fam',
        factor=FactorSpec(ast=SPEC),
        direction='high',
        data=DataSpec(snapshot_id='snap-1', dev_start='2020-01-01', dev_end='2021-01-01'),
        execution=ExecutionSpec(hold_days=5),
        evaluation=EvaluationSpec(top_n=10),
    )
    base.update(kwargs)
    return CandidateSpec(**base)


def test_key_ignores_identity_but_tracks_口径():
    a = _candidate(candidate_id='a', experiment_id='run_1')
    b = _candidate(candidate_id='b', experiment_id='run_2')
    assert a.key() == b.key(), '同公式同口径应共享 key（用于跨 run 复现识别）'
    assert a.fingerprint() != b.fingerprint()

    assert _candidate(direction='low').key() != a.key()
    assert _candidate(execution=ExecutionSpec(hold_days=20)).key() != a.key()
    assert _candidate(data=DataSpec(snapshot_id='snap-2')).key() != a.key()


def test_protocol_hash_changes_with_protocol():
    p1 = ResearchProtocol(run_id='r1', objective='x', data=DataSpec(snapshot_id='s'),
                          execution=ExecutionSpec(), evaluation=EvaluationSpec())
    p2 = ResearchProtocol(run_id='r1', objective='y', data=DataSpec(snapshot_id='s'),
                          execution=ExecutionSpec(), evaluation=EvaluationSpec())
    assert p1.protocol_hash() != p2.protocol_hash()


def test_budget_accounting():
    b = Budget(max_proposals=2, max_train_evaluations=1)
    assert b.can_propose()
    b.spend(proposals=2)
    assert not b.can_propose()
    assert b.can_evaluate()
    b.spend(train_evaluations=1)
    assert not b.can_evaluate()
    with pytest.raises(AttributeError):
        b.spend(unknown=1)


def test_registry_dedupe_within_run_and_prior_runs(tmp_path):
    reg = Registry(tmp_path / 'registry.db')
    proto = ResearchProtocol(run_id='run_a', objective='x', data=DataSpec(snapshot_id='s'),
                             execution=ExecutionSpec(), evaluation=EvaluationSpec())
    reg.create_run(proto)
    cand = _candidate(experiment_id='run_a', candidate_id='cand_a')
    assert reg.add_candidate(cand) is True
    assert reg.add_candidate(_candidate(experiment_id='run_a', candidate_id='cand_a2')) is False

    reg.create_run(ResearchProtocol(run_id='run_b', objective='x', data=DataSpec(snapshot_id='s'),
                                    execution=ExecutionSpec(), evaluation=EvaluationSpec()))
    assert reg.add_candidate(_candidate(experiment_id='run_b', candidate_id='cand_b')) is True
    assert reg.prior_runs_of_key(cand.key(), 'run_b') == ['run_a']


def test_registry_records_state_and_report(tmp_path):
    reg = Registry(tmp_path / 'r.db')
    reg.create_run(ResearchProtocol(run_id='r1', objective='x', data=DataSpec(snapshot_id='s'),
                                    execution=ExecutionSpec(), evaluation=EvaluationSpec()))
    reg.add_candidate(_candidate(experiment_id='r1', candidate_id='c1'))
    reg.add_attempt('c1', 'gate0', 'failed', 'boom')
    reg.add_evaluation('c1', 'dev', {'passed': False}, {}, 'reject', ['coverage'])
    reg.add_decision('c1', 'repair', '修实现')
    reg.set_state('c1', 'REPAIR_REQUIRED')
    reg.log_budget('r1', 'train_evaluation', 1)

    report = reg.report('r1')
    assert report['n_candidates'] == 1
    assert report['states'] == {'REPAIR_REQUIRED': 1}
    assert report['verdicts'] == {'reject': 1}
    assert report['budget'] == {'train_evaluation': 1}


def test_memory_search(tmp_path):
    reg = Registry(tmp_path / 'r.db')
    reg.add_memory(MemoryRecord(family='fam_a', failure_type='coverage_bias',
                                evidence='零权重 NaN 传播', rule='零权重项不得改变池'))
    reg.add_memory(MemoryRecord(family='fam_b', failure_type='cost_dominated',
                                evidence='换手过高', rule='看压力成本'))
    assert len(reg.search_memory(keyword='零权重')) == 1
    assert len(reg.search_memory(failure_type='cost_dominated')) == 1
    assert len(reg.search_memory()) == 2


def test_hypothesis_roundtrip():
    card = HypothesisCard(hypothesis_id='h', author='a', family='f', mechanism='m',
                          falsifiable_prediction='p', rejection_conditions=['r'],
                          direction='low', spec=SPEC)
    again = HypothesisCard.from_dict(json.loads(json.dumps(card.to_dict())))
    assert again.spec == SPEC
    assert again.direction == 'low'

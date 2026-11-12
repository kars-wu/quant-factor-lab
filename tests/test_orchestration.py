"""研究循环：产物落盘、预算扣减、跨 run 复现识别、保留段访问登记。"""
import json

from multi_agent.contracts import Budget
from multi_agent.orchestration import ResearchLoop, RunConfig


def _cfg(tmp_path, **overrides):
    params = dict(
        dataset='synthetic',
        synthetic_symbols=40,
        synthetic_dates=320,
        synthetic_strength=1.0,
        selection_top=8,
        hold_days=5,
        budget=Budget(max_proposals=6, max_variants=2, max_train_evaluations=6,
                      max_full_validations=1),
        out_dir=str(tmp_path),
        registry_path=str(tmp_path / 'registry.db'),
        verbose=False,
        seed=13,
    )
    params.update(overrides)
    return RunConfig(**params)


def test_run_writes_artifacts_and_respects_budget(tmp_path):
    loop = ResearchLoop(_cfg(tmp_path))
    report = loop.run()

    assert report['candidates'], '至少应产出候选'
    assert report['ledger']['budget'].get('proposal', 0) <= 6
    assert report['llm']['used'] is False

    run_dir = tmp_path / report['run_id']
    assert (run_dir / 'plan.json').exists()
    assert (run_dir / 'report.json').exists()
    for cand in report['candidates']:
        facet = run_dir / cand['candidate_id']
        for name in ('spec.json', 'gate.json', 'audit.json'):
            assert (facet / name).exists(), f'缺少 {name}'

    payload = json.loads((run_dir / 'report.json').read_text())
    assert payload['run_id'] == report['run_id']


def test_budget_exhaustion_stops_proposals(tmp_path):
    cfg = _cfg(tmp_path, budget=Budget(max_proposals=2, max_train_evaluations=2,
                                       max_full_validations=0))
    report = ResearchLoop(cfg).run()
    assert report['ledger']['budget'].get('proposal', 0) <= 2
    assert len(report['candidates']) <= 2


def test_second_run_is_recognised_as_replication(tmp_path):
    first = ResearchLoop(_cfg(tmp_path)).run()
    second = ResearchLoop(_cfg(tmp_path, run_id='run_second')).run()
    replicated = [c for c in second['candidates'] if c.get('replicated_from')]
    assert first['candidates'], '第一次运行应有候选'
    assert replicated, '第二次运行同公式应被标记为复现'


def test_holdout_without_pass_does_nothing(tmp_path):
    report = ResearchLoop(_cfg(tmp_path, holdout_start=None)).run()
    assert report['holdout_validation'] == []


def test_holdout_access_is_logged_when_used(tmp_path):
    cfg = _cfg(tmp_path, synthetic_dates=420, synthetic_strength=3.0,
               holdout_start='2019-04-01')
    loop = ResearchLoop(cfg)
    report = loop.run()
    validated = report['holdout_validation']
    if validated:
        rows = loop.registry._conn().execute(
            'SELECT * FROM access_events').fetchall()
        assert len(rows) >= len(validated)
        assert report['ledger']['budget'].get('full_validation', 0) <= 1


def test_roles_fall_back_to_heuristics_without_llm(tmp_path):
    loop = ResearchLoop(_cfg(tmp_path))
    report = loop.run()
    assert 'policy' in report and report['policy']['version'] == 'v0'
    # 启发式产出 6 个候选上限内，且都通过了 Gate0（干净表达式）
    assert all(c['gate_passed'] for c in report['candidates'])

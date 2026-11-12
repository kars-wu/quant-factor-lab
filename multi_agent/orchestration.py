"""编排器：把六个角色、Gate0、评估器、台账串成一个有界研究循环。

一次 ``ResearchLoop.run()`` 做的事：

    Lead 出计划 → Researcher 出假设 → Engineer 冻表达式 → 注册（去重）
    → Gate0 不变量 → 开发窗口评估 → 成本压力情景 → 审计/决策 → 失败写入记忆
    → （可选）冻结后对保留段做一次验证

所有预算（提案、变体、评估、验证、LLM 调用）都在这里扣减并落台账。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from multi_agent import dsl
from multi_agent.agents import prompts, roles
from multi_agent.agents.base import LLMClient, get_client, is_llm
from multi_agent.contracts import (
    Budget,
    CandidateSpec,
    DataSpec,
    EvaluationSpec,
    ExecutionSpec,
    HypothesisCard,
    ResearchProtocol,
)
from multi_agent.datasets.base import DataView
from multi_agent.evaluation import gates, portfolio
from multi_agent.policy import PolicyState
from multi_agent.registry import MemoryRecord, Registry


@dataclass
class RunConfig:
    objective: str = '在给定研究池内寻找扣成本后有增量信息的短周期因子'
    dataset: str = 'synthetic'          # synthetic | sqlite
    db_path: Optional[str] = None
    start: str = '2018-01-01'
    end: Optional[str] = None
    universe: str = 'top_n_amount'
    top_n: int = 500               # 股票池规模
    selection_top: int = 100       # 每次选股数量（截面 top_k）
    hold_days: int = 20            # 每笔持仓天数；每日新开 1/hold_days 资金的一笔
    dev_start: Optional[str] = None
    dev_end: Optional[str] = None
    holdout_start: Optional[str] = None
    budget: Budget = field(default_factory=Budget)
    policy: PolicyState = field(default_factory=PolicyState)
    out_dir: str = 'outputs/multi_agent'
    registry_path: Optional[str] = None
    seed: int = 0
    cost_stress_bps: float = 50.0
    design_strict: bool = False
    synthetic_symbols: int = 120
    synthetic_dates: int = 700
    synthetic_strength: float = 0.35
    run_id: Optional[str] = None
    verbose: bool = True


def build_view(cfg: RunConfig) -> DataView:
    if cfg.dataset == 'synthetic':
        from multi_agent.datasets.synthetic import make_synthetic_view
        return make_synthetic_view(cfg.synthetic_dates, cfg.synthetic_symbols, cfg.seed,
                                   signal_strength=cfg.synthetic_strength)
    if cfg.dataset == 'sqlite':
        from multi_agent.datasets.sqlite_view import load_sqlite_view
        return load_sqlite_view(cfg.db_path, start=cfg.start, end=cfg.end,
                               universe=cfg.universe, top_n=cfg.top_n)
    raise ValueError(f'未知 dataset: {cfg.dataset}')


def _default_dev_window(view: DataView, dev_start: Optional[str],
                        dev_end: Optional[str]) -> tuple[str, str]:
    dates = view.dates
    start = dev_start or str(dates[int(len(dates) * 0.35)].date())
    end = dev_end or str(dates[-1].date())
    return start, end


class ResearchLoop:
    def __init__(self, cfg: RunConfig, client: Optional[LLMClient] = None):
        self.cfg = cfg
        self.view = build_view(cfg)
        dev_start, dev_end = _default_dev_window(self.view, cfg.dev_start, cfg.dev_end)
        self.dev_view = self.view.slice(dev_start, dev_end)
        self.holdout_view: Optional[DataView] = (
            self.view.slice(cfg.holdout_start, None) if cfg.holdout_start else None
        )
        self.client = client or get_client()
        self.policy = cfg.policy
        self.banned = prompts.banned_families()
        self.registry = Registry(cfg.registry_path or Path(cfg.out_dir) / 'registry.db')

        run_id = cfg.run_id or f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        data_spec = DataSpec(
            snapshot_id=self.dev_view.snapshot_id,
            universe=cfg.universe,
            universe_params={'top_n': cfg.top_n},
            start=cfg.start,
            end=dev_end,
            dev_start=dev_start,
            dev_end=dev_end,
            holdout_start=cfg.holdout_start,
        )
        self.protocol = ResearchProtocol(
            run_id=run_id,
            objective=cfg.objective,
            data=data_spec,
            execution=ExecutionSpec(hold_days=cfg.hold_days),
            evaluation=EvaluationSpec(top_n=cfg.selection_top,
                                      design_strict=cfg.design_strict),
            budget=cfg.budget,
            policy_version=self.policy.version,
        )
        self.out_dir = Path(cfg.out_dir) / run_id
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.traces: List[Dict[str, Any]] = []

    # ── 对外入口 ─────────────────────────────────────────────
    def run(self) -> Dict[str, Any]:
        cfg = self.cfg
        self.registry.create_run(self.protocol)
        if cfg.verbose:
            logger.info('run {} | 数据 {} | dev {} ~ {} | policy {}',
                        self.protocol.run_id, self.dev_view.snapshot_id,
                        self.protocol.data.dev_start, self.protocol.data.dev_end,
                        self.policy.version)
            logger.info('视图: {}', self.dev_view.info())

        memories = self.registry.search_memory(limit=20)
        plan = roles.research_lead(self.client, self.protocol, self.policy, memories)
        self._save('plan.json', plan.to_dict())
        self._spend_llm()

        candidates: List[Dict[str, Any]] = []
        for track in plan.payload.get('research_tracks', []):
            name = track.get('local_ref', 'track').replace('track_', '')
            quota = min(int(track.get('proposal_quota') or self.policy.proposal_quota),
                        max(self.protocol.budget.max_proposals
                            - self.protocol.budget.proposals_used, 0))
            if quota <= 0 or self.protocol.budget.exhausted():
                break
            env = roles.propose_hypotheses(self.client, name, self.protocol, self.policy,
                                           self.banned, memories, quota)
            self._spend_llm()
            self._save(f'hypotheses_{name}.json', env.to_dict())
            if env.status != 'ready':
                continue
            for item in env.payload.get('hypotheses', []):
                if self.protocol.budget.exhausted():
                    break
                record = self._process_hypothesis(HypothesisCard.from_dict(item))
                if record:
                    candidates.append(record)

        validated = self._final_validation(candidates)
        self.registry.close_run(self.protocol.run_id)
        report = self.report(candidates, validated)
        self._save('report.json', report)
        if self.cfg.verbose:
            print_report(report)
        return report

    # ── 单个假设的处理链 ────────────────────────────────────
    def _process_hypothesis(self, card: HypothesisCard) -> Optional[Dict[str, Any]]:
        budget = self.protocol.budget
        spec = roles.card_spec(card)
        if spec is None:
            self._memory(card, 'implementation_bug', '假设没有可编译的表达式草案')
            return None

        self.registry.add_hypothesis(self.protocol.run_id, card)
        budget.spend(proposals=1)
        self.registry.log_budget(self.protocol.run_id, 'proposal', 1)

        impl = roles.implement(self.client, card, spec, self.protocol, self.policy)
        budget.spend(train_evaluations=1)
        self.registry.log_budget(self.protocol.run_id, 'train_evaluation', 1)
        if impl.status != 'ready':
            self._memory(card, 'implementation_bug', impl.summary)
            return None

        candidate = CandidateSpec.from_dict(impl.payload['candidate'])
        if not self.registry.add_candidate(candidate):
            self.registry.log_budget(self.protocol.run_id, 'duplicate', 1)
            self._memory(card, 'duplicate_family',
                         f'同 key 候选已存在：{dsl.canonical(spec)}')
            return None

        prior = self.registry.prior_runs_of_key(candidate.key(), self.protocol.run_id)
        if prior:
            logger.warning('{} 在其他 run 出现过（{}），本次视为复现', candidate.candidate_id,
                           ','.join(prior[:3]))

        facet = self.out_dir / candidate.candidate_id
        facet.mkdir(parents=True, exist_ok=True)
        (facet / 'spec.json').write_text(json.dumps(candidate.to_dict(), ensure_ascii=False,
                                                    indent=2, default=str))

        factor = dsl.compile_factor(spec, self.dev_view.panels)
        gate = gates.run_gate0(spec, self.dev_view, candidate.execution, candidate.evaluation,
                               factor=factor, start=candidate.data.dev_start,
                               end=candidate.data.dev_end,
                               checks=self.policy.audit_checks,
                               design_strict=candidate.evaluation.design_strict,
                               min_categories=candidate.evaluation.min_operator_categories)
        (facet / 'gate.json').write_text(json.dumps(gate.to_dict(), ensure_ascii=False, indent=2))

        audit = roles.audit(self.client, candidate, gate.to_dict())
        self._spend_llm()
        self._save(str(facet.relative_to(self.out_dir) / 'audit.json'), audit.to_dict())

        if not gate.passed:
            self.registry.set_state(candidate.candidate_id, 'REPAIR_REQUIRED')
            self.registry.add_attempt(candidate.candidate_id, 'gate0', 'failed', gate.summary(),
                                      str(facet))
            reasons = [f"{i.name}: {i.detail}" for i in gate.failures()]
            self.registry.add_evaluation(candidate.candidate_id, 'dev', gate.to_dict(), {},
                                         'reject', reasons)
            self.registry.add_decision(candidate.candidate_id, 'repair',
                                       'Gate0 硬失败，需先修实现')
            for ftype in roles.classify_reasons(reasons):
                self._memory(card, ftype, '; '.join(reasons[:2]))
            self.traces.append({'candidate_id': candidate.candidate_id, 'status': 'gate_failed',
                                'failure_types': roles.classify_reasons(reasons)})
            return self._record(candidate, gate, None, audit, None, 'repair')

        self.registry.set_state(candidate.candidate_id, 'AUDITED')
        result = portfolio.evaluate_candidate(
            factor, self.dev_view, candidate.execution, candidate.evaluation,
            direction=candidate.direction, start=candidate.data.dev_start,
            end=candidate.data.dev_end,
        )
        summary = result['summary']
        stress = self._cost_stress(factor, candidate, result['verdict'])
        decision = roles.decide(self.client, candidate, summary, result['verdict'],
                                result['reasons'], stress)
        self._spend_llm()

        metrics = {k: v for k, v in summary.items()}
        metrics['yearly'] = result['yearly']
        metrics['cost_stress'] = stress
        (facet / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False,
                                                       indent=2, default=str))
        self._save(str(facet.relative_to(self.out_dir) / 'decision.json'), decision.to_dict())

        self.registry.add_evaluation(candidate.candidate_id, 'dev', gate.to_dict(),
                                     metrics, result['verdict'], result['reasons'])
        self.registry.add_decision(candidate.candidate_id,
                                   decision.payload.get('decision', result['verdict']),
                                   decision.payload.get('rationale', ''))
        keep = decision.payload.get('decision') in ('keep', 'watch')
        self.registry.set_state(candidate.candidate_id,
                                'DEV_VALIDATED' if keep else 'REJECTED')
        if result['verdict'] != 'pass':
            for ftype in roles.classify_reasons(result['reasons']):
                self._memory(card, ftype, '; '.join(result['reasons'][:2]))
        self.traces.append({
            'candidate_id': candidate.candidate_id,
            'verdict': result['verdict'],
            'reasons': result['reasons'],
            'failure_types': roles.classify_reasons(result['reasons']),
            'excess_sharpe': summary.get('excess_sharpe'),
            'duplicate': False,
        })
        return self._record(candidate, gate, metrics, audit, decision, result['verdict'])

    def _cost_stress(self, factor, candidate: CandidateSpec,
                     dev_verdict: str = 'pass') -> Dict[str, Any]:
        stress_exec = replace(candidate.execution,
                              cost_bps_round_trip=self.cfg.cost_stress_bps)
        res = portfolio.evaluate_candidate(
            factor, self.dev_view, stress_exec, candidate.evaluation,
            direction=candidate.direction, start=candidate.data.dev_start,
            end=candidate.data.dev_end,
        )
        return {
            'cost_bps': self.cfg.cost_stress_bps,
            'verdict': res['verdict'],
            'excess_sharpe': res['summary'].get('excess_sharpe'),
            # 只有当开发阶段通过、压力情景不通过时，才算"结论翻转"
            'flipped': dev_verdict == 'pass' and res['verdict'] != 'pass',
        }

    def _final_validation(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """保留段揭盲：只对开发阶段通过的候选，一次，且不回流到搜索。"""
        if self.holdout_view is None:
            return []
        kept = [c for c in candidates if c.get('verdict') == 'pass']
        if not kept:
            return []
        kept.sort(key=lambda c: c.get('metrics', {}).get('excess_sharpe') or -9, reverse=True)
        out = []
        for entry in kept:
            if not self.protocol.budget.can_validate():
                break
            candidate = CandidateSpec.from_dict(entry['candidate'])
            self.registry.log_access('validator', f'holdout:{candidate.data.holdout_start}',
                                     'read', True)
            factor = dsl.compile_factor(candidate.factor.ast, self.holdout_view.panels)
            res = portfolio.evaluate_candidate(
                factor, self.holdout_view, candidate.execution, candidate.evaluation,
                direction=candidate.direction,
            )
            self.protocol.budget.spend(full_validations=1)
            self.registry.log_budget(self.protocol.run_id, 'full_validation', 1)
            self.registry.add_evaluation(candidate.candidate_id, 'holdout', {},
                                         res['summary'], res['verdict'], res['reasons'])
            self.registry.set_state(candidate.candidate_id,
                                    'FINAL_VALIDATED' if res['verdict'] == 'pass' else 'REJECTED')
            if res['verdict'] != 'pass':
                self._memory_by_id(candidate, 'holdout_rejected',
                                   '; '.join(res['reasons'][:2]))
            out.append({'candidate_id': candidate.candidate_id,
                        'verdict': res['verdict'],
                        'metrics': res['summary'],
                        'reasons': res['reasons']})
            logger.info('保留段验证 {}: {}', candidate.candidate_id, res['verdict'])
        return out

    # ── 辅助 ────────────────────────────────────────────────
    def _spend_llm(self) -> None:
        if is_llm(self.client):
            self.protocol.budget.spend(llm_calls=1)
            self.registry.log_budget(self.protocol.run_id, 'llm_call', 1)

    def _record(self, candidate: CandidateSpec, gate, metrics, audit, decision,
                verdict: str) -> Dict[str, Any]:
        return {
            'candidate': candidate.to_dict(),
            'candidate_id': candidate.candidate_id,
            'family': candidate.family,
            'direction': candidate.direction,
            'spec': dsl.canonical(candidate.factor.ast),
            'gate_passed': gate.passed,
            'metrics': metrics or {},
            'audit': audit.payload.get('classification'),
            'decision': (decision.payload.get('decision') if decision else 'repair'),
            'verdict': verdict,
            'replicated_from': self.registry.prior_runs_of_key(candidate.key(),
                                                               self.protocol.run_id),
        }

    def _memory(self, card: HypothesisCard, failure_type: str, evidence: str) -> None:
        self.registry.add_memory(MemoryRecord(
            family=card.family,
            failure_type=failure_type,
            evidence=evidence[:500],
            rule='同类结构在修复或改变机制前不重复提交',
            source=f'hypothesis:{card.hypothesis_id}',
            evidence_level='implementation_audit',
            run_id=self.protocol.run_id,
            tags=f'{card.family},{card.author}',
        ))

    def _memory_by_id(self, candidate: CandidateSpec, failure_type: str, evidence: str) -> None:
        self.registry.add_memory(MemoryRecord(
            family=candidate.family, failure_type=failure_type, evidence=evidence[:500],
            rule='保留段失败后该保留段降级为开发历史', source=f'candidate:{candidate.candidate_id}',
            evidence_level='holdout', run_id=self.protocol.run_id,
            tags=candidate.family,
        ))

    def _save(self, name: str, payload: Dict[str, Any]) -> None:
        path = self.out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))

    def report(self, candidates: List[Dict[str, Any]],
               validated: List[Dict[str, Any]]) -> Dict[str, Any]:
        ledger = self.registry.report(self.protocol.run_id)
        return {
            'run_id': self.protocol.run_id,
            'protocol': self.protocol.to_dict(),
            'view': self.dev_view.info(),
            'policy': self.policy.to_dict(),
            'llm': {'used': is_llm(self.client), 'client': self.client.name},
            'candidates': candidates,
            'holdout_validation': validated,
            'ledger': ledger,
            'artifacts_dir': str(self.out_dir),
        }


def _fmt(value: Any) -> str:
    return '-' if value is None else f'{float(value):.3f}'


def print_report(report: Dict[str, Any]) -> None:
    print()
    print(f"run {report['run_id']}  policy={report['policy']['version']}  "
          f"llm={'yes' if report['llm']['used'] else 'no'}")
    print('-' * 96)
    rows = report['candidates']
    if not rows:
        print('（无候选产出：预算耗尽 / 全部重复 / 全部实现失败）')
    for r in rows:
        m = r.get('metrics') or {}
        print(f"{r['candidate_id']:<22}{r['family'][:20]:<22}{r['verdict']:<8}"
              f"gate={'ok' if r['gate_passed'] else 'fail':<6} "
              f"sharpe={_fmt(m.get('sharpe')):<10}"
              f"excess={_fmt(m.get('excess_sharpe')):<10}"
              f"decision={r['decision']}"
              f"{' (cost-stress flip)' if (r.get('metrics') or {}).get('cost_stress', {}).get('flipped') else ''}")
    if report.get('holdout_validation'):
        print('保留段：' + ', '.join(
            f"{v['candidate_id']}={v['verdict']}" for v in report['holdout_validation']))
    ledger = report['ledger']
    print('-' * 96)
    print(f"假设 {ledger['n_hypotheses']} | 候选 {ledger['n_candidates']} | "
          f"状态 {ledger['states']} | 判定 {ledger['verdicts']} | "
          f"预算 {ledger['budget']} | 记忆 {ledger['memories']}")

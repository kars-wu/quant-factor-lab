"""GEPA 式外层进化：只改「研究策略」，用配对评测决定是否替换冠军。

这里刻意把外层目标定在**过程指标**（能不能真的抓到已知缺陷、会不会误杀
干净候选、检查成本），而不是「改 prompt 后某个因子更赚钱」——后者是噪声，
而且会把开发集越用越脏。

一次进化循环：

    收集轨迹 → 归因失败 → 选一个模块（proposal / audit_checklist / budget_allocation）
    → 生成挑战者 → 相同视图/种子/预算下配对评测 → 达标则替换冠军，否则回滚
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from multi_agent.agents import roles
from multi_agent.agents.base import LLMClient, get_client, is_llm
from multi_agent.contracts import EvaluationSpec, ExecutionSpec
from multi_agent.datasets.base import DataView
from multi_agent.evaluation import gates
from multi_agent.policy import PolicyState

GOOD_SPEC = {'op': 'ts_pct_change', 'n': 5, 'inputs': [{'field': 'close'}]}
ACCEPT_MARGIN = 0.05


@dataclass
class PolicyScore:
    policy_version: str
    detection: float
    false_positive: float
    check_cost: float
    score: float
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'policy_version': self.policy_version,
            'detection': round(self.detection, 4),
            'false_positive': round(self.false_positive, 4),
            'check_cost': self.check_cost,
            'score': round(self.score, 4),
            'detail': self.detail,
        }


def evaluate_policy(policy: PolicyState, view: DataView,
                    execution: ExecutionSpec, evaluation: EvaluationSpec) -> PolicyScore:
    """用审计回归用例给一个 policy 打分。

    detection      : 已启用的检查项中，真正能抓住对应缺陷的比例
    false_positive : 干净候选是否被误杀（0/1）
    check_cost     : 启用的检查项数量（运行时成本代理）
    score          : detection - 0.5 * false_positive
    """
    suite = gates.audit_regression_suite(view, execution, evaluation)
    effectiveness = {
        'prefix': suite['negative_shift_detected'],
        'cross_section': True,
        'zero_weight': suite['zero_weight_detected'],
        'coverage': suite['coverage_bias_detected'],
        'cash': suite['calendar_gap_detected'],
    }
    enabled = [c for c in policy.audit_checks if c in effectiveness]
    detection = (sum(1.0 for c in enabled if effectiveness[c]) / len(effectiveness)
                 if effectiveness else 0.0)
    good_report = gates.run_gate0(GOOD_SPEC, view, execution, evaluation, checks=enabled)
    false_positive = 0.0 if good_report.passed else 1.0
    score = detection - 0.5 * false_positive
    return PolicyScore(
        policy_version=policy.version,
        detection=detection,
        false_positive=false_positive,
        check_cost=float(len(enabled)),
        score=score,
        detail={'effectiveness': effectiveness, 'enabled': enabled,
                'good_spec_failures': [i.name for i in good_report.failures()]},
    )


def paired_compare(champion: PolicyState, challenger: PolicyState, view: DataView,
                   execution: ExecutionSpec, evaluation: EvaluationSpec) -> Dict[str, Any]:
    """同一视图、同一缺陷集下比较两个 policy —— 这就是"配对评测"。"""
    a = evaluate_policy(champion, view, execution, evaluation)
    b = evaluate_policy(challenger, view, execution, evaluation)
    improved = (b.score > a.score + ACCEPT_MARGIN
                and b.false_positive <= a.false_positive)
    return {
        'champion': a.to_dict(),
        'challenger': b.to_dict(),
        'improved': bool(improved),
        'delta': round(b.score - a.score, 4),
    }


def evolve(traces: List[Dict[str, Any]], view: DataView,
           policy: PolicyState, execution: ExecutionSpec, evaluation: EvaluationSpec,
           rounds: int = 3, client: Optional[LLMClient] = None,
           out_path: Optional[str] = None) -> Dict[str, Any]:
    """执行外层进化，返回冠军 policy 与逐轮记录。"""
    client = client or get_client()
    champion = policy
    history: List[Dict[str, Any]] = []

    for i in range(rounds):
        proposal = roles.reflect(client, traces, champion)
        mutation = proposal.payload or {}
        module = mutation.get('module')
        change = mutation.get('change') or {}
        if not module or not change:
            logger.info('第 {} 轮：没有可执行的 policy 改动，停止进化', i + 1)
            break
        try:
            challenger = champion.bump(module, change,
                                       reason=mutation.get('target_root_cause', ''))
        except ValueError as exc:
            history.append({'round': i + 1, 'skipped': str(exc)})
            break

        cmp = paired_compare(champion, challenger, view, execution, evaluation)
        accepted = cmp['improved']
        history.append({
            'round': i + 1,
            'module': module,
            'change': change,
            'target_root_cause': mutation.get('target_root_cause'),
            'expected_metric': mutation.get('expected_metric'),
            'comparison': cmp,
            'accepted': accepted,
        })
        logger.info('第 {} 轮 {}: {} → delta={} {}', i + 1, module,
                    challenger.version, cmp['delta'], 'accepted' if accepted else 'rolled back')
        if accepted:
            champion = challenger
        # 拒绝的变异只留在 history 里，冠军保持不变（回滚）

    result = {
        'champion': champion.to_dict(),
        'history': history,
        'llm_used': is_llm(client),
    }
    if out_path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return result


def summarize(result: Dict[str, Any]) -> str:
    lines = [f"冠军 policy: {result['champion']['version']} "
             f"(checks={result['champion']['audit_checks']})"]
    for h in result['history']:
        if 'skipped' in h:
            lines.append(f"  第 {h['round']} 轮跳过: {h['skipped']}")
            continue
        c = h['comparison']
        lines.append(
            f"  第 {h['round']} 轮 {h['module']}: {c['champion']['score']} → "
            f"{c['challenger']['score']} ({'接受' if h['accepted'] else '回滚'})"
        )
    return '\n'.join(lines)

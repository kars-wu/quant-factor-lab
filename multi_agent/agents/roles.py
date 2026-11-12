"""六种角色的实现。

每个角色：

1. 若配置了 LLM（``MA_LLM_BASE_URL``），把冻结上下文序列化后交给模型，
   要求按角色 prompt 输出 JSON；解析失败或输出非法时回退到启发式；
2. 否则使用确定性启发式 —— 这保证仓库**离线可跑、可测试**，
   但它不会自动发现新 alpha，只是把流程跑通、把审计与预算机制落实。

硬约束在代码里，不在 prompt 里：DSL 白名单、Gate0、成本、预算、去重、
最终保留段访问，这些都不依赖模型是否听话。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from multi_agent import dsl
from multi_agent.agents import prompts
from multi_agent.agents.base import LLMClient, is_llm, parse_json
from multi_agent.contracts import (
    ArtifactEnvelope,
    CandidateSpec,
    FactorSpec,
    HypothesisCard,
    ResearchProtocol,
    short_hash,
)
from multi_agent.policy import PolicyState

# ─────────────────────────── 启发式机制库 ───────────────────────────
# 这些是教科书级的候选机制，用来跑通流程与做回归测试。
# 它们不是"挖出来的因子"，也不保证有 alpha；真实研究应由 LLM 或人来提新机制。


@dataclass(frozen=True)
class MechanismTemplate:
    family: str
    mechanism: str
    prediction: str
    rejections: List[str]
    spec: Dict[str, Any]
    direction: str
    confounders: List[str] = field(default_factory=list)
    variants: List[Dict[str, Any]] = field(default_factory=list)


def _op(name: str, *inputs: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    node = {'op': name, 'inputs': list(inputs)}
    node.update(kwargs)
    return node


def _f(name: str) -> Dict[str, Any]:
    return {'field': name}


REVERSAL_LIBRARY: List[MechanismTemplate] = [
    MechanismTemplate(
        family='factor_volprice_divergence_rev_5_close',
        mechanism='量价背离式下跌：跌幅大但成交额相对萎缩，说明抛压更多来自流动性而非基本面，易出现修复',
        prediction='跌幅排名靠前且量能收缩排名靠前的组合，扣成本净超额为正，且高于单纯跌幅排序',
        rejections=['与单纯 5 日反转截面相关 > 0.8', '扣成本超额 <= 0', '覆盖偏向大市值'],
        spec=_op('mul',
                 _op('cs_rank', _op('neg', _op('ts_pct_change', _f('close'), n=5))),
                 _op('cs_rank', _op('neg', _op('div', _f('volume'),
                                             _op('ts_rolling', _f('volume'), n=20, agg='mean'))))),
        direction='high',
        confounders=['流动性', '规模'],
    ),
    MechanismTemplate(
        family='factor_liquidity_recovery_rev_10_amount',
        mechanism='流动性恢复：成交额斜率转正而价格仍在低位，说明承接资金开始进场',
        prediction='10 日跌幅靠前且成交额斜率为正的组合，净超额高于单纯跌幅排序',
        rejections=['与 10 日反转截面相关 > 0.8', '扣成本超额 <= 0', '覆盖偏向大市值'],
        spec=_op('mul',
                 _op('cs_rank', _op('neg', _op('ts_pct_change', _f('close'), n=10))),
                 _op('cs_rank', _op('ts_slope', _f('amount'), n=10))),
        direction='high',
        confounders=['流动性', '规模'],
    ),
    MechanismTemplate(
        family='factor_industry_neutral_rev_5_volume',
        mechanism='行业内超跌：剥掉行业共同涨跌后个股自身跌幅越大，修复空间越明确',
        prediction='行业中性后 5 日残差跌幅最大的一档，扣成本净超额为正，且高于未中性版本',
        rejections=['与未中性反转相关 > 0.9（说明中性化没起作用）', '扣成本超额 <= 0'],
        spec=_op('mul',
                 _op('cs_rank', _op('neg', _op('ts_pct_change', _f('close'), n=5))),
                 _op('cs_group_rank', by='industry_code',
                     inputs=[_op('div', _f('volume'),
                                 _op('ts_rolling', _f('volume'), n=20, agg='mean'))])),
        direction='high',
        confounders=['行业', '流动性'],
    ),
    MechanismTemplate(
        family='short_term_reversal',
        mechanism='短期跌幅越大，卖压越可能已经释放，未来数日出现修复',
        prediction='在池内 5 日跌幅最大的一档，扣成本后 T+1 开盘到 T+2 开盘的净超额为正',
        rejections=['扣成本超额 <= 0', '逐年超额不全为正', '覆盖偏向大市值'],
        spec=_op('ts_pct_change', _f('close'), n=5),
        direction='low',
        confounders=['规模', '波动率', '行业'],
        variants=[_op('ts_pct_change', _f('close'), n=10)],
    ),
    MechanismTemplate(
        family='volume_contraction_reversal',
        mechanism='缩量下跌说明抛压衰减，比放量下跌更容易出现反转',
        prediction='跌幅/成交额放大比值最高的一档，净超额高于单纯跌幅排序',
        rejections=['与单纯反转因子高度相关（>0.8）', '扣成本超额 <= 0'],
        spec=_op('div', _op('ts_pct_change', _f('close'), n=3),
                 _op('div', _f('volume'), _op('ts_rolling', _f('volume'), n=20, agg='mean'))),
        direction='low',
        confounders=['流动性', '规模'],
    ),
    MechanismTemplate(
        family='residual_reversal',
        mechanism='剥离中期趋势后，短期残差下跌反映的是流动性冲击而非基本面',
        prediction='残差最低一档的净超额为正，且对中期动量的暴露接近 0',
        rejections=['残差因子与 5 日动量的截面相关 > 0.9', '扣成本超额 <= 0'],
        spec=_op('cs_regress_residual', _op('ts_pct_change', _f('close'), n=5),
                 _op('ts_pct_change', _f('close'), n=60)),
        direction='low',
        confounders=['中期动量'],
    ),
    MechanismTemplate(
        family='close_location_reversal',
        mechanism='收盘价落在当日区间低位，说明尾盘抛压未尽；过度反应后易修复',
        prediction='收盘位置最低一档的次日净超额为正',
        rejections=['扣成本超额 <= 0', '信号对开盘缺口高度敏感'],
        spec=_op('div',
                 _op('sub', _op('sub', _op('add', _f('close'), _f('close')), _f('high')), _f('low')),
                 _op('sub', _f('high'), _f('low'))),
        direction='low',
        confounders=['波动率'],
    ),
]

CONTINUATION_LIBRARY: List[MechanismTemplate] = [
    MechanismTemplate(
        family='factor_volprice_confirm_mom_20_close',
        mechanism='量价同向确认过的上涨更有持续性：20 日量价相关性高的上涨，比单纯涨幅更能延续',
        prediction='量价相关排名与 20 日涨幅排名同时靠前的一档，扣成本净超额高于单纯动量',
        rejections=['与 20 日动量截面相关 > 0.9', '扣成本超额 <= 0', '换手过高导致成本吃掉收益'],
        spec=_op('mul',
                 _op('cs_rank', _op('ts_corr', _f('close'), _f('volume'), n=20)),
                 _op('ts_pct_change', _f('close'), n=20)),
        direction='high',
        confounders=['流动性', '规模'],
    ),
    MechanismTemplate(
        family='factor_slope_decay_mom_20_close',
        mechanism='稳健趋势：回归斜率相对波动更高（信息比式动量），近端涨幅再经衰减加权确认',
        prediction='斜率/波动排名与衰减动量排名同时靠前的一档，扣成本净超额高于单纯 20 日动量',
        rejections=['与 20 日动量截面相关 > 0.9', '扣成本超额 <= 0', '换手过高导致成本吃掉收益'],
        spec=_op('mul',
                 _op('cs_rank', _op('div', _op('ts_slope', _f('close'), n=20),
                                   _op('ts_rolling', _f('close'), n=20, agg='std'))),
                 _op('cs_rank', _op('decay_linear',
                                    _op('ts_pct_change', _f('close'), n=5), n=10))),
        direction='high',
        confounders=['波动率', '规模'],
    ),
    MechanismTemplate(
        family='medium_term_momentum',
        mechanism='中期趋势存在延续性，60 日涨幅高的一档未来继续占优',
        prediction='60 日涨幅最高一档的扣成本净超额为正',
        rejections=['扣成本超额 <= 0', '逐年超额不全为正'],
        spec=_op('ts_pct_change', _f('close'), n=60),
        direction='high',
        confounders=['规模', '行业'],
        variants=[_op('ts_pct_change', _f('close'), n=120)],
    ),
    MechanismTemplate(
        family='price_position_breakout',
        mechanism='价格接近区间高点时，上方套牢盘少，趋势更容易延续',
        prediction='收盘价/近 60 日最高价最高一档的净超额为正',
        rejections=['扣成本超额 <= 0', '换手过高导致成本吃掉收益'],
        spec=_op('div', _f('close'), _op('ts_rolling', _f('high'), n=60, agg='max')),
        direction='high',
        confounders=['波动率', '规模'],
    ),
    MechanismTemplate(
        family='low_turnover_strength',
        mechanism='低换手上涨代表筹码稳定，比高换手上涨更可能延续',
        prediction='动量/换手率最高一档的净超额高于单纯动量',
        rejections=['与动量因子相关 > 0.9', '扣成本超额 <= 0'],
        spec=_op('mul', _op('ts_pct_change', _f('close'), n=20),
                 _op('neg', _f('turnover_rate'))),
        direction='high',
        confounders=['流动性', '规模'],
    ),
    MechanismTemplate(
        family='volume_price_confirmation',
        mechanism='量价同向确认过的上涨更有持续性',
        prediction='量与价 20 日相关性高且近期上涨的一档净超额为正',
        rejections=['扣成本超额 <= 0', '量与价的同期相关存在前视'],
        spec=_op('mul', _op('ts_pct_change', _f('close'), n=20),
                 _op('ts_corr', _f('close'), _f('volume'), n=20)),
        direction='high',
        confounders=['流动性'],
    ),
]

LIBRARY: Dict[str, List[MechanismTemplate]] = {
    'reversal': REVERSAL_LIBRARY,
    'continuation': CONTINUATION_LIBRARY,
}


# ─────────────────────────── 工具 ───────────────────────────

def _operator_pool() -> Dict[str, List[str]]:
    pool: Dict[str, List[str]] = {}
    for op in dsl.OP_REGISTRY:
        pool.setdefault(dsl.OP_CATEGORY[op], []).append(op)
    return {k: sorted(v) for k, v in pool.items()}


def _ctx(protocol: ResearchProtocol, policy: PolicyState,
         extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {
        'run_id': protocol.run_id,
        'policy_version': policy.version,
        'objective': protocol.objective,
        'data_snapshot': protocol.data.snapshot_id,
        'dev_window': [protocol.data.dev_start, protocol.data.dev_end],
        'execution': protocol.execution.to_dict(),
        'evaluation': protocol.evaluation.to_dict(),
        'allowed_fields': dsl.PANEL_FIELDS,
        'allowed_ops': list(dsl.OP_REGISTRY),
        'operator_pool': _operator_pool(),
        'design_rules': {
            'min_categories': 3,
            'must_include_relative': ['cs_rank', 'cs_zscore', 'ts_corr', 'ts_cov'],
            'forbid_root_linear_blend': '根节点不能是两个有内容子树相加减',
            'max_nodes': dsl.MAX_NODES,
            'max_depth': dsl.MAX_DEPTH,
            'max_warmup': 250,
            'min_periods': 'max(2, n//2)',
        },
        'naming_convention': 'factor_<逻辑>_<变换>_<窗口>_<字段>',
        'standardization_note': '不要用全样本均值方差；用 cs_rank / cs_zscore 做逐日截面处理',
        'budget': protocol.budget.to_dict(),
    }
    if extra:
        payload.update(extra)
    return payload


def _ask(client: LLMClient, role: str, context: Dict[str, Any],
         track: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """调用 LLM 并尽力解析 JSON；任何异常都返回 None 交给启发式兜底。"""
    if not is_llm(client):
        return None
    system = prompts.system(role)
    if track:
        system = system.replace('{track}', track)
    try:
        raw = client.complete(system, json.dumps(context, ensure_ascii=False, default=str))
        return parse_json(raw)
    except Exception:      # noqa: BLE001 - 模型/网络失败都走兜底
        return None


def _heuristic_cards(track: str, quota: int, banned: List[str],
                     policy: PolicyState) -> List[HypothesisCard]:
    templates = LIBRARY.get(track, [])
    cards: List[HypothesisCard] = []
    for tpl in templates:
        if len(cards) >= quota:
            break
        if dsl.validate(tpl.spec):
            continue
        if prompts.is_banned(f'{tpl.family} {dsl.canonical(tpl.spec)}', banned):
            continue
        hid = 'h_' + short_hash([track, tpl.family, dsl.exact_key(tpl.spec)], 10)
        cards.append(HypothesisCard(
            hypothesis_id=hid,
            author=f'researcher:{track}',
            family=tpl.family,
            mechanism=tpl.mechanism,
            falsifiable_prediction=tpl.prediction,
            rejection_conditions=list(tpl.rejections),
            direction=tpl.direction,
            expected_horizon=5,
            confounders=list(tpl.confounders),
            variants_allowed=[dsl.canonical(v) for v in tpl.variants],
            spec=tpl.spec,
        ))
        # 每个假设最多一个经济含义明确的变体，且受 policy 的 variant 配额限制
        for i, variant in enumerate(tpl.variants[:policy.variant_quota]):
            if len(cards) >= quota:
                break
            if dsl.validate(variant):
                continue
            cards.append(HypothesisCard(
                hypothesis_id='h_' + short_hash([track, tpl.family, i], 10),
                author=f'researcher:{track}',
                family=tpl.family,
                mechanism=f'{tpl.mechanism}（窗口变体：{dsl.canonical(variant)}）',
                falsifiable_prediction=tpl.prediction,
                rejection_conditions=list(tpl.rejections),
                direction=tpl.direction,
                expected_horizon=5,
                confounders=list(tpl.confounders),
                spec=variant,
            ))
    return cards


def _spec_for_family(track: str, family: str) -> Optional[Dict[str, Any]]:
    for tpl in LIBRARY.get(track, []):
        if tpl.family == family:
            return tpl.spec
    return None


# ─────────────────────────── 角色实现 ───────────────────────────

def research_lead(client: LLMClient, protocol: ResearchProtocol, policy: PolicyState,
                  memories: List[Dict[str, Any]]) -> ArtifactEnvelope:
    context = _ctx(protocol, policy, {
        'failure_memories': [
            {'family': m['family'], 'failure_type': m['failure_type'], 'rule': m['rule']}
            for m in memories[:10]
        ],
    })
    data = _ask(client, 'research_lead', context)
    if data and isinstance(data.get('payload'), dict) and data['payload'].get('research_tracks'):
        payload = data['payload']
        summary = data.get('summary', 'model plan')
    else:
        payload = {
            'objective': protocol.objective,
            'research_tracks': [
                {
                    'local_ref': f'track_{name}',
                    'assigned_role': f'researcher:{name}',
                    'mechanisms': [t.mechanism for t in LIBRARY[name][:2]],
                    'proposal_quota': policy.proposal_quota,
                    'variant_quota': policy.variant_quota,
                    'allowed_changes': ['替换一个已注册算子', '同族窗口变体（受配额限制）'],
                    'stop_conditions': ['Gate0 硬失败', '预算耗尽'],
                    'reopen_conditions': ['实现 bug 修复后新建 attempt'],
                }
                for name in policy.tracks if name in LIBRARY
            ],
            'budget_allocation_proposal': protocol.budget.to_dict(),
            'key_uncertainties': ['覆盖选择效应', '成本敏感性', '执行可成交性'],
        }
        summary = f'{len(payload["research_tracks"])} 条研究流，预算 {protocol.budget.max_proposals} 提案'
    return ArtifactEnvelope(
        artifact_type='research_plan', run_id=protocol.run_id, summary=summary,
        payload=payload, policy_version=policy.version,
    )


def propose_hypotheses(client: LLMClient, track: str, protocol: ResearchProtocol,
                       policy: PolicyState, banned: List[str],
                       memories: List[Dict[str, Any]], quota: int) -> ArtifactEnvelope:
    """一个研究流 = 一个 Researcher 实例（反转 / 动量各一个）。"""
    context = _ctx(protocol, policy, {
        'track': track,
        'quota': quota,
        'banned_families': banned,
        'related_failures': [
            {'family': m['family'], 'failure_type': m['failure_type'], 'evidence': m['evidence']}
            for m in memories[:8]
        ],
        'skeleton_example': {
            'hypotheses': [{
                'local_ref': 'h1', 'family': '...', 'economic_mechanism': '...',
                'falsifiable_prediction': '...', 'rejection_conditions': ['...'],
                'spec': _op('ts_pct_change', _f('close'), n=5),
                'selection_direction': 'high', 'expected_horizon': 5,
            }]
        },
    })
    role = prompts.researcher_role(track)
    data = _ask(client, role, context, track=track)
    cards: List[HypothesisCard] = []
    if data and isinstance(data.get('payload'), dict):
        for i, item in enumerate(data['payload'].get('hypotheses', [])[:quota]):
            spec = item.get('spec')
            if not isinstance(spec, dict) or dsl.validate(spec):
                continue
            text = f"{item.get('family', '')} {dsl.canonical(spec)} {item.get('economic_mechanism', '')}"
            if prompts.is_banned(text, banned):
                continue
            cards.append(HypothesisCard(
                hypothesis_id='h_' + short_hash([track, spec], 10),
                author=f'researcher:{track}',
                family=item.get('family') or f'family_{i}',
                mechanism=item.get('economic_mechanism', ''),
                falsifiable_prediction=item.get('falsifiable_prediction', ''),
                rejection_conditions=item.get('rejection_conditions', []) or ['扣成本超额 <= 0'],
                direction='low' if item.get('selection_direction') == 'low' else 'high',
                expected_horizon=int(item.get('expected_horizon') or 5),
                confounders=item.get('confounders', []) or [],
                spec=dsl.to_ast(spec),
            ))
    if len(cards) < quota:
        seen = {dsl.family_of(_heuristic_spec(c)) for c in cards if _heuristic_spec(c)}
        for card in _heuristic_cards(track, quota, banned, policy):
            spec = _heuristic_spec(card)
            if spec is None or dsl.family_of(spec) in seen:
                continue
            seen.add(dsl.family_of(spec))
            cards.append(card)
            if len(cards) >= quota:
                break
    return ArtifactEnvelope(
        artifact_type='hypothesis_cards', run_id=protocol.run_id,
        summary=f'{track}: {len(cards)} 个假设',
        payload={'track_ref': f'track_{track}',
                 'hypotheses': [c.to_dict() for c in cards]},
        policy_version=policy.version,
        status='ready' if cards else 'budget_exhausted',
    )


def _heuristic_spec(card: HypothesisCard) -> Optional[Dict[str, Any]]:
    """启发式卡片 → AST：优先卡片自带的 spec，其次按 family 反查库。"""
    if isinstance(card.spec, dict):
        return card.spec
    for tpl in REVERSAL_LIBRARY + CONTINUATION_LIBRARY:
        if tpl.family == card.family:
            return tpl.spec
    return None


def card_spec(card: HypothesisCard) -> Optional[Dict[str, Any]]:
    return _heuristic_spec(card)


def implement(client: LLMClient, card: HypothesisCard, spec: Dict[str, Any],
              protocol: ResearchProtocol, policy: PolicyState,
              parent_id: Optional[str] = None) -> ArtifactEnvelope:
    errors = dsl.validate(spec)
    if errors:
        return ArtifactEnvelope(
            artifact_type='candidate_spec', run_id=protocol.run_id, status='rejected',
            summary='表达式非法：' + '; '.join(errors[:3]),
            missing_inputs=['valid_ast'], policy_version=policy.version,
        )
    ast = dsl.to_ast(spec)
    candidate_id = 'cand_' + short_hash([protocol.run_id, dsl.exact_key(ast)], 16)
    candidate = CandidateSpec(
        candidate_id=candidate_id,
        experiment_id=protocol.run_id,
        hypothesis_id=card.hypothesis_id,
        family=card.family,
        factor=FactorSpec(ast=ast, warmup=dsl.warmup(ast)),
        direction=card.direction,
        data=protocol.data,
        execution=protocol.execution,
        evaluation=protocol.evaluation,
        parent_id=parent_id,
        policy_version=policy.version,
        prompt_hash=short_hash([prompts.load('researcher'),
                                prompts.load('factor_engineer')]),
        code_hash=short_hash([dsl.exact_key(ast), dsl.OP_REGISTRY]),
        seed=protocol.budget.train_evaluations_used,
    )
    return ArtifactEnvelope(
        artifact_type='candidate_spec', run_id=protocol.run_id,
        summary=f'{dsl.canonical(ast)} direction={card.direction}',
        payload={
            'candidate': candidate.to_dict(),
            'hypothesis_ref': card.hypothesis_id,
            'family': card.family,
            'warmup': dsl.warmup(ast),
            'complexity': dsl.complexity(ast),
            'fields': dsl.fields_of(ast),
            'design_report': dsl.design_report(ast).to_dict(),
            'implementation_notes': '由受限 DSL 编译，无自定义 Python',
        },
        policy_version=policy.version,
    )


def audit(client: LLMClient, candidate: CandidateSpec, gate: Dict[str, Any],
          coverage: Optional[float] = None) -> ArtifactEnvelope:
    hard = []
    warnings = []
    for item in gate.get('items', []):
        if not item.get('passed'):
            hard.append({'check': item['name'], 'detail': item.get('detail', ''),
                         'evidence': 'Gate0'})
    if coverage is not None and coverage < 0.7:
        warnings.append({'check': 'coverage', 'detail': f'覆盖率 {coverage:.3f} 偏低'})

    data = _ask(client, 'auditor', {
        'candidate': candidate.to_dict(),
        'gate0': gate,
        'coverage': coverage,
    })
    classification = 'pass' if not hard else 'repair_required'
    summary = 'Gate0 全部通过' if not hard else f'{len(hard)} 项硬失败'
    if data and isinstance(data.get('payload'), dict):
        payload = data['payload']
        # 硬门槛不能被模型覆盖：Gate0 失败时只允许 reject/repair
        if hard and payload.get('classification') not in ('reject', 'repair_required'):
            payload['classification'] = 'repair_required'
        if not hard and payload.get('classification') not in ('pass', 'needs_data'):
            payload['classification'] = 'pass'
        summary = data.get('summary', summary)
    else:
        payload = {'candidate_ref': candidate.candidate_id, 'hard_failures': hard,
                   'warnings': warnings, 'not_run': [], 'classification': classification,
                   'reproduction_notes': 'gates.run_gate0'}
    return ArtifactEnvelope(
        artifact_type='audit_report', run_id=candidate.experiment_id,
        status='ready' if payload.get('classification') != 'needs_data' else 'needs_data',
        summary=summary, payload=payload,
        evidence_refs=[{'artifact_id': 'gate0', 'location': 'evaluation/gates.py',
                        'claim': summary}],
        policy_version=candidate.policy_version,
    )


def decide(client: LLMClient, candidate: CandidateSpec, summary: Dict[str, Any],
           verdict: str, reasons: List[str],
           stress: Optional[Dict[str, Any]] = None) -> ArtifactEnvelope:
    stress = stress or {}
    data = _ask(client, 'portfolio_analyst', {
        'candidate': candidate.to_dict(),
        'metrics': summary,
        'verdict': verdict,
        'reasons': reasons,
        'cost_stress': stress,
    })
    if data and isinstance(data.get('payload'), dict):
        payload = data['payload']
        if verdict == 'reject' and payload.get('decision') in ('keep', 'watch'):
            payload['decision'] = 'reject'
            payload['rationale'] = '硬门槛未通过：' + '; '.join(reasons[:3])
        summary_text = data.get('summary', verdict)
    else:
        decision = 'keep' if verdict == 'pass' else 'reject'
        rationale = ('; '.join(reasons[:4]) if reasons
                     else f"excess_sharpe={summary.get('excess_sharpe')}")
        if stress.get('flipped'):
            decision = 'reject'
            rationale += f" | 成本压力情景（{stress.get('cost_bps')}bp）结论翻转"
        payload = {
            'candidate_ref': candidate.candidate_id,
            'decision': decision,
            'metrics_used': {k: summary.get(k) for k in
                             ('excess_sharpe', 'ir', 'active_annual_return',
                              'max_drawdown', 'turnover_annual', 'coverage')},
            'incremental_evidence': f"active_annual={summary.get('active_annual_return')}",
            'coverage_effect': f"coverage={summary.get('coverage')}",
            'cost_sensitivity': ('翻转' if stress.get('flipped') else '未翻转'),
            'rationale': rationale,
            'next_checks': [] if decision == 'keep' else ['新建 attempt 修复后重跑'],
        }
        summary_text = f'{decision}: {rationale[:80]}'
    return ArtifactEnvelope(
        artifact_type='decision_record', run_id=candidate.experiment_id,
        summary=summary_text, payload=payload, policy_version=candidate.policy_version,
    )


_FAILURE_MAP = [
    ('prefix', 'availability_leak'),
    ('dsl', 'implementation_bug'),
    ('warmup', 'implementation_bug'),
    ('complexity', 'implementation_bug'),
    ('cross_section', 'implementation_bug'),
    ('zero_weight', 'implementation_bug'),
    ('coverage', 'coverage_bias'),
    ('cash', 'non_executable'),
    ('逐年', 'unstable_across_blocks'),
    ('有效天数', 'insufficient_evidence'),
    ('excess_sharpe', 'no_incremental_signal'),
    ('active_annual', 'cost_dominated'),
    ('max_drawdown', 'unstable_across_blocks'),
]


def classify_reasons(reasons: List[str]) -> List[str]:
    types: List[str] = []
    for reason in reasons:
        for key, ftype in _FAILURE_MAP:
            if key in reason and ftype not in types:
                types.append(ftype)
    return types or ['no_incremental_signal']


def reflect(client: LLMClient, traces: List[Dict[str, Any]],
            policy: PolicyState) -> ArtifactEnvelope:
    """根据轨迹提出单模块改动。没有痕迹时返回空建议。"""
    if not traces:
        return ArtifactEnvelope(
            artifact_type='policy_mutation', run_id='', status='rejected',
            summary='无轨迹，暂不进化', policy_version=policy.version,
        )

    failure_counts: Dict[str, int] = {}
    for t in traces:
        for f in t.get('failure_types', []):
            failure_counts[f] = failure_counts.get(f, 0) + 1

    data = _ask(client, 'evolution_coach', {
        'policy': policy.to_dict(),
        'traces': traces[-20:],
        'failure_counts': failure_counts,
        'evolvable_modules': ['proposal', 'audit_checklist', 'budget_allocation'],
        'available_audit_checks': ['prefix', 'cross_section', 'zero_weight', 'coverage', 'cash'],
    })

    payload = None
    if data and isinstance(data.get('payload'), dict) and data['payload'].get('module'):
        payload = data['payload']
    else:
        payload = _heuristic_mutation(failure_counts, traces, policy)

    return ArtifactEnvelope(
        artifact_type='policy_mutation', run_id='',
        summary=f"module={payload.get('module')} change={payload.get('change')}",
        payload=payload, policy_version=policy.version,
    )


def _heuristic_mutation(failure_counts: Dict[str, int], traces: List[Dict[str, Any]],
                        policy: PolicyState) -> Dict[str, Any]:
    """启发式：先补审计缺口，再按最高频失败类型选一个模块做最小改动。"""
    top = max(failure_counts, key=failure_counts.get) if failure_counts else None
    all_checks = ['prefix', 'cross_section', 'zero_weight', 'coverage', 'cash']
    missing = [c for c in all_checks if c not in policy.audit_checks]

    # 审计清单有缺口时，补缺口永远优先：未审计的风险比调参更值得修
    if missing:
        return {
            'base_policy_version': policy.version,
            'module': 'audit_checklist',
            'change': {'audit_checks': list(dict.fromkeys(list(policy.audit_checks) + missing))},
            'target_root_cause': 'audit_gap',
            'expected_metric': 'bug_detection_rate',
            'paired_eval_plan': {'tasks': 6, 'seed': 0},
            'rollback_condition': 'false_positive_rate 上升',
        }

    if top in ('duplicate_family', 'no_incremental_signal'):
        tracks = list(reversed(policy.tracks))
        return {
            'base_policy_version': policy.version,
            'module': 'proposal',
            'change': {'tracks': tracks},
            'target_root_cause': top,
            'expected_metric': 'duplicate_rate',
            'paired_eval_plan': {'tasks': 6, 'seed': 0},
            'rollback_condition': '通过率下降',
        }

    return {
        'base_policy_version': policy.version,
        'module': 'budget_allocation',
        'change': {'proposal_quota': max(1, policy.proposal_quota - 1)},
        'target_root_cause': top or 'unknown',
        'expected_metric': 'cost_per_valid_candidate',
        'paired_eval_plan': {'tasks': 6, 'seed': 0},
        'rollback_condition': '单轮有效候选数下降',
    }

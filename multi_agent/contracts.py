"""研究契约：所有候选、数据视图、执行与评估口径的唯一事实来源。

设计要点（详见 docs/multi-agent.md）：

* 一个候选 = 冻结的表达式 + 方向 + 数据快照 + 执行口径 + 评估规则；
* 任何一项变化都是新候选（新 ``candidate_id`` / 新血缘），不能就地改口径后
  复用旧的通过记录；
* 契约里的所有字段都可序列化、可哈希，便于台账对账与去重；
* 这里的结构不依赖具体市场，A 股只是默认参数。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = '1.0'


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def short_hash(payload: Any, n: int = 16) -> str:
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()[:n]


@dataclass(frozen=True)
class DataSpec:
    """数据快照与股票池。

    ``snapshot_id`` 必须指向一个不可变的数据视图。研究 worker 只应拿到这个
    视图，不应直接访问完整数据库文件。
    """

    snapshot_id: str
    universe: str = 'top_n_amount'
    universe_params: Dict[str, Any] = field(default_factory=lambda: {'top_n': 500, 'lookback': 20})
    adjust: str = 'qfq'
    start: str = '2018-01-01'
    end: str = ''
    dev_start: str = '2021-01-01'
    dev_end: str = ''
    # 最终保留段：None 表示尚未切分。研究期不得读取。
    holdout_start: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionSpec:
    """执行口径。默认：T 收盘信号 → T+1 开盘买入 → 持有 hold_days 后开盘卖出。"""

    signal_time: str = 'close'          # close = T 收盘后可用
    order_time: str = 'next_open'       # next_open = T+1 开盘
    exit_time: str = 'open'             # open | close
    hold_days: int = 1
    cost_bps_round_trip: float = 18.0   # 双边总成本（bp），压力情景建议 30 / 50
    capital: float = 1_000_000.0
    max_participation: float = 0.05     # 单日成交额参与率上限
    buy_limit_gap: Optional[float] = None   # 例：0.09 表示开盘涨幅 >= 9% 视为买不进
    allow_partial_fill: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationSpec:
    """评估与晋级规则。规则在开跑前固定，不因结果修改。"""

    top_n: int = 100
    metric: str = 'excess_sharpe'
    min_days: int = 250
    min_coverage: float = 0.6
    yearly_consistency: bool = True
    min_positive_year_ratio: float = 0.6   # 逐年正超额占比下限（1.0 = 必须全正）
    pass_rules: Dict[str, float] = field(default_factory=lambda: {
        'excess_sharpe': 0.3,
        'active_annual_return': 0.0,
        'max_drawdown': -0.45,
    })
    neutralized: bool = True     # 是否同时要求市值中性后仍有效
    fdr_q: float = 0.1
    # 是否强制"算子交互"设计（≥N 类算子、必须含交叉/相对算子、禁止根部线性叠加）
    design_strict: bool = False
    min_operator_categories: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FactorSpec:
    """因子表达式的注册外壳。``ast`` 见 multi_agent.dsl。"""

    ast: Dict[str, Any]
    missing_policy: str = 'exclude'    # exclude | zero | cross_section_median
    rank_scope: str = 'universe'       # universe | covered
    warmup: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HypothesisCard:
    """研究假设卡片：先写机制与可否证预测，再看结果。"""

    hypothesis_id: str
    author: str
    family: str
    mechanism: str
    falsifiable_prediction: str
    rejection_conditions: List[str]
    direction: str = 'high'
    expected_horizon: int = 1
    confounders: List[str] = field(default_factory=list)
    variants_allowed: List[str] = field(default_factory=list)
    related_failures: List[str] = field(default_factory=list)
    # 表达式草案（DSL AST），由 Researcher 提出、Engineer 冻结
    spec: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'HypothesisCard':
        known = {f for f in cls.__dataclass_fields__}   # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass(frozen=True)
class CandidateSpec:
    """一个可执行、可复现、可审计的候选。"""

    candidate_id: str
    experiment_id: str
    hypothesis_id: str
    family: str
    factor: FactorSpec
    direction: str                      # high = 买因子最大端
    data: DataSpec
    execution: ExecutionSpec
    evaluation: EvaluationSpec
    parent_id: Optional[str] = None
    policy_version: str = 'v0'
    prompt_hash: str = ''
    code_hash: str = ''
    seed: int = 0

    # ── 派生标识 ──────────────────────────────────────────────
    def key(self) -> str:
        """去重键：只包含决定结果的字段（表达式/方向/数据/执行/评估）。"""
        return short_hash({
            'ast': self.factor.ast,
            'missing': self.factor.missing_policy,
            'direction': self.direction,
            'data': self.data.to_dict(),
            'execution': self.execution.to_dict(),
            'evaluation': self.evaluation.to_dict(),
        })

    def fingerprint(self) -> str:
        return short_hash({
            'key': self.key(),
            'experiment': self.experiment_id,
            'candidate': self.candidate_id,
            'hypothesis': self.hypothesis_id,
            'family': self.family,
            'parent': self.parent_id,
            'policy': self.policy_version,
            'prompt': self.prompt_hash,
            'code': self.code_hash,
            'seed': self.seed,
        })

    def with_overrides(self, **kwargs: Any) -> 'CandidateSpec':
        return replace(self, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['key'] = self.key()
        d['fingerprint'] = self.fingerprint()
        return d

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'CandidateSpec':
        return cls(
            candidate_id=payload['candidate_id'],
            experiment_id=payload['experiment_id'],
            hypothesis_id=payload['hypothesis_id'],
            family=payload['family'],
            factor=FactorSpec(**payload['factor']),
            direction=payload['direction'],
            data=DataSpec(**payload['data']),
            execution=ExecutionSpec(**payload['execution']),
            evaluation=EvaluationSpec(**payload['evaluation']),
            parent_id=payload.get('parent_id'),
            policy_version=payload.get('policy_version', 'v0'),
            prompt_hash=payload.get('prompt_hash', ''),
            code_hash=payload.get('code_hash', ''),
            seed=payload.get('seed', 0),
        )


@dataclass
class Budget:
    """研究预算。所有计数跨 agent、跨 prompt 版本累计，不重置。"""

    max_proposals: int = 24
    max_variants: int = 24
    max_train_evaluations: int = 48
    max_full_validations: int = 8
    max_llm_calls: int = 400
    proposals_used: int = 0
    variants_used: int = 0
    train_evaluations_used: int = 0
    full_validations_used: int = 0
    llm_calls_used: int = 0

    def can_propose(self, n: int = 1) -> bool:
        return self.proposals_used + n <= self.max_proposals

    def can_evaluate(self) -> bool:
        return self.train_evaluations_used < self.max_train_evaluations

    def can_validate(self) -> bool:
        return self.full_validations_used < self.max_full_validations

    def can_call_llm(self) -> bool:
        return self.llm_calls_used < self.max_llm_calls

    def spend(self, **kwargs: int) -> None:
        for k, v in kwargs.items():
            attr = f'{k}_used'
            if not hasattr(self, attr):
                raise AttributeError(f'未知预算项 {k}')
            setattr(self, attr, getattr(self, attr) + v)

    def exhausted(self) -> bool:
        return not (self.can_propose(0) and self.can_evaluate() and self.can_validate())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchProtocol:
    """一次研究 run 的冻结协议。"""

    run_id: str
    objective: str
    data: DataSpec
    execution: ExecutionSpec
    evaluation: EvaluationSpec
    budget: Budget = field(default_factory=Budget)
    policy_version: str = 'v0'
    schema_version: str = SCHEMA_VERSION

    def protocol_hash(self) -> str:
        return short_hash({
            'objective': self.objective,
            'data': self.data.to_dict(),
            'execution': self.execution.to_dict(),
            'evaluation': self.evaluation.to_dict(),
            'policy': self.policy_version,
            'schema': self.schema_version,
        })

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['protocol_hash'] = self.protocol_hash()
        return d


@dataclass(frozen=True)
class ArtifactEnvelope:
    """agent 之间交接的统一外壳，对应角色 prompt 里的 JSON 契约。"""

    artifact_type: str
    run_id: str
    status: str = 'ready'
    summary: str = ''
    payload: Dict[str, Any] = field(default_factory=dict)
    evidence_refs: List[Dict[str, Any]] = field(default_factory=list)
    missing_inputs: List[str] = field(default_factory=list)
    change_requests: List[Dict[str, Any]] = field(default_factory=list)
    exposure_events: List[Dict[str, Any]] = field(default_factory=list)
    policy_version: str = 'v0'

    VALID_STATUS = ('ready', 'needs_data', 'needs_tool', 'rejected', 'budget_exhausted')

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d['status'] not in self.VALID_STATUS:
            raise ValueError(f'非法 status: {d["status"]}')
        d['context_echo'] = {
            'run_id': self.run_id,
            'policy_version': self.policy_version,
            'input_artifact_ids': [e.get('artifact_id') for e in self.evidence_refs],
        }
        return d

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'ArtifactEnvelope':
        return cls(
            artifact_type=payload.get('artifact_type', 'unknown'),
            run_id=payload.get('run_id', ''),
            status=payload.get('status', 'ready'),
            summary=payload.get('summary', ''),
            payload=payload.get('payload') or {},
            evidence_refs=payload.get('evidence_refs') or [],
            missing_inputs=payload.get('missing_inputs') or [],
            change_requests=payload.get('change_requests') or [],
            exposure_events=payload.get('exposure_events') or [],
            policy_version=payload.get('policy_version', 'v0'),
        )

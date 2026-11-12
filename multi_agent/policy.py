"""研究策略（policy）状态：可被外层进化模块修改，但只能改"研究流程"。

不可进化：数据可用时点、撮合事实、股票池基础政策、purge、成本会计、
最终保留访问、统计口径与最低接受标准。这些都在代码里固定。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List

EVOLVABLE_MODULES = ('proposal', 'audit_checklist', 'budget_allocation')

ALL_AUDIT_CHECKS = ('prefix', 'cross_section', 'zero_weight', 'coverage', 'cash')


@dataclass(frozen=True)
class PolicyState:
    version: str = 'v0'
    # proposal 模块
    tracks: List[str] = field(default_factory=lambda: ['reversal', 'continuation'])
    proposal_quota: int = 4
    variant_quota: int = 1
    # audit_checklist 模块
    audit_checks: List[str] = field(default_factory=lambda: list(ALL_AUDIT_CHECKS))
    # budget_allocation 模块
    max_full_validations: int = 8
    notes: str = ''
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def bump(self, module: str, change: Dict[str, Any], reason: str = '') -> 'PolicyState':
        """生成新版本；每次只允许改一个模块。"""
        if module not in EVOLVABLE_MODULES:
            raise ValueError(f'模块 {module} 不可进化')
        current = self.to_dict()
        updated = dict(current)
        for key, value in change.items():
            if key not in current:
                raise ValueError(f'未知 policy 字段 {key}')
            updated[key] = value
        version = f'v{int(str(self.version).lstrip("v") or 0) + 1}'
        history = list(current['history']) + [{
            'from': self.version, 'to': version, 'module': module,
            'change': change, 'reason': reason,
        }]
        updated['version'] = version
        updated['history'] = history
        return PolicyState(**{k: v for k, v in updated.items()
                              if k in PolicyState.__dataclass_fields__})  # type: ignore[attr-defined]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'PolicyState':
        known = {f for f in cls.__dataclass_fields__}   # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})

    def with_checks(self, checks: List[str]) -> 'PolicyState':
        invalid = [c for c in checks if c not in ALL_AUDIT_CHECKS]
        if invalid:
            raise ValueError(f'未知审计检查项: {invalid}')
        return replace(self, audit_checks=list(dict.fromkeys(checks)))

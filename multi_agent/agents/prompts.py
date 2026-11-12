"""角色 prompt 加载与已占用家族清单。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

PROMPT_DIR = Path(__file__).resolve().parent.parent / 'prompts'
DEFAULT_BANNED_PATH = Path(__file__).resolve().parent.parent / 'banned_families.txt'

PROMPT_FILES = {
    'shared': '00_shared.md',
    'research_lead': '01_research_lead.md',
    'researcher': '02_researcher.md',
    'researcher:reversal': '02a_reversal_researcher.md',
    'researcher:continuation': '02b_continuation_researcher.md',
    'factor_engineer': '03_factor_engineer.md',
    'auditor': '04_auditor.md',
    'portfolio_analyst': '05_portfolio_analyst.md',
    'evolution_coach': '06_evolution_coach.md',
}


@lru_cache(maxsize=None)
def load(role: str) -> str:
    if role not in PROMPT_FILES:
        raise KeyError(f'未知角色 {role}')
    path = PROMPT_DIR / PROMPT_FILES[role]
    if not path.exists():
        raise FileNotFoundError(f'缺少 prompt 文件: {path}')
    return path.read_text(encoding='utf-8')


def system(role: str) -> str:
    """角色 system prompt = 公共约束 + 角色说明。

    两个 researcher 实例（反转 / 动量）共享 `02_researcher.md` 的硬约束，
    再各自追加自己的研究流说明。
    """
    if role.startswith('researcher:'):
        return (load('shared') + '\n\n---\n\n' + load('researcher')
                + '\n\n---\n\n' + load(role))
    return load('shared') + '\n\n---\n\n' + load(role)


def researcher_role(track: str) -> str:
    """track → 角色名；未注册的研究流回退到通用 researcher。"""
    candidate = f'researcher:{track}'
    return candidate if candidate in PROMPT_FILES else 'researcher'


@lru_cache(maxsize=None)
def banned_families(path: str | None = None) -> List[str]:
    """已挖掘/已占用家族关键词。

    默认文件里是模板（全部注释），请填入你自己已经挖过的因子家族，
    命中即拒绝研究，避免把旧结论当新发现重复投稿。
    """
    target = Path(path) if path else DEFAULT_BANNED_PATH
    if not target.exists():
        return []
    out: List[str] = []
    for line in target.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        out.append(line.lower())
    return out


def is_banned(text: str, banned: List[str]) -> bool:
    low = (text or '').lower()
    return any(kw in low for kw in banned)

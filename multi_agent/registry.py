"""研究台账：单写者 SQLite。

原则：

* 产物不可覆盖：重试用新的 ``attempt_id``，失败记录保留；
* 所有状态变化都带时间戳与来源，便于事后对账；
* 去重键 ``candidate_key``（表达式+方向+数据+执行+评估）唯一，
  同一 key 重复注册会被拒绝 —— 重复研究也是要计入成本的；
* 台账同时记录访问事件与预算事件，供报告与审计使用。
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from multi_agent.contracts import CandidateSpec, HypothesisCard, ResearchProtocol

DEFAULT_REGISTRY_PATH = Path('outputs/multi_agent/registry.db')

_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_runs (
    run_id        TEXT PRIMARY KEY,
    objective     TEXT,
    protocol_hash TEXT,
    protocol_json TEXT,
    policy_version TEXT,
    status        TEXT DEFAULT 'running',
    created_at    TEXT
);
CREATE TABLE IF NOT EXISTS hypotheses (
    hypothesis_id TEXT PRIMARY KEY,
    run_id        TEXT,
    author        TEXT,
    family        TEXT,
    direction     TEXT,
    payload_json  TEXT,
    status        TEXT DEFAULT 'registered',
    created_at    TEXT
);
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id  TEXT PRIMARY KEY,
    run_id        TEXT,
    hypothesis_id TEXT,
    family        TEXT,
    candidate_key TEXT,
    fingerprint   TEXT,
    direction     TEXT,
    ast_json      TEXT,
    spec_json     TEXT,
    parent_id     TEXT,
    state         TEXT DEFAULT 'REGISTERED',
    created_at    TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_candidate_run_key
    ON candidates (run_id, candidate_key);
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id    TEXT PRIMARY KEY,
    candidate_id  TEXT,
    stage         TEXT,
    status        TEXT,
    message       TEXT,
    artifact_dir  TEXT,
    created_at    TEXT
);
CREATE TABLE IF NOT EXISTS evaluations (
    eval_id       TEXT PRIMARY KEY,
    candidate_id  TEXT,
    split         TEXT,
    gate_json     TEXT,
    metrics_json  TEXT,
    verdict       TEXT,
    reasons_json  TEXT,
    created_at    TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
    decision_id   TEXT PRIMARY KEY,
    candidate_id  TEXT,
    decision      TEXT,
    rationale     TEXT,
    created_at    TEXT
);
CREATE TABLE IF NOT EXISTS memories (
    memory_id     TEXT PRIMARY KEY,
    run_id        TEXT,
    family        TEXT,
    failure_type  TEXT,
    evidence      TEXT,
    rule          TEXT,
    source        TEXT,
    evidence_level TEXT,
    restart_condition TEXT,
    tags          TEXT,
    created_at    TEXT
);
CREATE TABLE IF NOT EXISTS access_events (
    event_id      TEXT PRIMARY KEY,
    actor         TEXT,
    resource      TEXT,
    mode          TEXT,
    allowed       INTEGER,
    created_at    TEXT
);
CREATE TABLE IF NOT EXISTS budget_events (
    event_id      TEXT PRIMARY KEY,
    run_id        TEXT,
    kind          TEXT,
    n             INTEGER,
    created_at    TEXT
);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec='seconds')


def _uid(prefix: str) -> str:
    return f'{prefix}_{uuid.uuid4().hex[:12]}'


@dataclass
class MemoryRecord:
    family: str
    failure_type: str
    evidence: str
    rule: str
    source: str = ''
    evidence_level: str = 'exploratory'
    restart_condition: str = ''
    tags: str = ''
    run_id: str = ''

    FAILURE_TYPES = (
        'implementation_bug', 'availability_leak', 'coverage_bias', 'non_executable',
        'cost_dominated', 'unstable_across_blocks', 'no_incremental_signal',
        'insufficient_evidence', 'duplicate_family', 'holdout_rejected',
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'memory_id': _uid('mem'),
            'run_id': self.run_id,
            'family': self.family,
            'failure_type': self.failure_type,
            'evidence': self.evidence,
            'rule': self.rule,
            'source': self.source,
            'evidence_level': self.evidence_level,
            'restart_condition': self.restart_condition,
            'tags': self.tags,
            'created_at': _now(),
        }


class Registry:
    """单写者台账。多进程场景请只让编排进程写。"""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or DEFAULT_REGISTRY_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA journal_mode=WAL')
        return conn

    # ── run / hypothesis ─────────────────────────────────────
    def create_run(self, protocol: ResearchProtocol) -> str:
        with self._lock, self._conn() as conn:
            conn.execute(
                'INSERT INTO research_runs (run_id, objective, protocol_hash, protocol_json,'
                ' policy_version, status, created_at) VALUES (?,?,?,?,?,?,?)',
                (protocol.run_id, protocol.objective, protocol.protocol_hash(),
                 json.dumps(protocol.to_dict(), ensure_ascii=False, default=str),
                 protocol.policy_version, 'running', _now()),
            )
        return protocol.run_id

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute('SELECT * FROM research_runs WHERE run_id=?', (run_id,)).fetchone()
        return dict(row) if row else None

    def close_run(self, run_id: str, status: str = 'finished') -> None:
        with self._lock, self._conn() as conn:
            conn.execute('UPDATE research_runs SET status=? WHERE run_id=?', (status, run_id))

    def add_hypothesis(self, run_id: str, card: HypothesisCard) -> str:
        with self._lock, self._conn() as conn:
            conn.execute(
                'INSERT OR REPLACE INTO hypotheses (hypothesis_id, run_id, author, family,'
                ' direction, payload_json, created_at) VALUES (?,?,?,?,?,?,?)',
                (card.hypothesis_id, run_id, card.author, card.family, card.direction,
                 json.dumps(card.to_dict(), ensure_ascii=False), _now()),
            )
        return card.hypothesis_id

    def list_hypotheses(self, run_id: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute('SELECT * FROM hypotheses WHERE run_id=? ORDER BY created_at',
                                (run_id,)).fetchall()
        return [dict(r) for r in rows]

    # ── candidate ────────────────────────────────────────────
    def add_candidate(self, spec: CandidateSpec) -> bool:
        """返回 False 表示同一 run 内已注册过同 key 的候选（重复研究被拒绝）。

        key 只在 run 内唯一：换 run 重跑同一公式属于「复现」，允许但要能追溯
        （见 ``prior_runs_of_key``）。
        """
        payload = json.dumps(spec.to_dict(), ensure_ascii=False, default=str)
        with self._lock, self._conn() as conn:
            exists = conn.execute(
                'SELECT candidate_id FROM candidates WHERE run_id=? AND candidate_key=?',
                (spec.experiment_id, spec.key())).fetchone()
            if exists:
                return False
            conn.execute(
                'INSERT INTO candidates (candidate_id, run_id, hypothesis_id, family,'
                ' candidate_key, fingerprint, direction, ast_json, spec_json, parent_id,'
                ' state, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                (spec.candidate_id, spec.experiment_id, spec.hypothesis_id, spec.family,
                 spec.key(), spec.fingerprint(), spec.direction,
                 json.dumps(spec.factor.ast, ensure_ascii=False), payload,
                 spec.parent_id, 'REGISTERED', _now()),
            )
        return True

    def find_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute('SELECT * FROM candidates WHERE candidate_key=?', (key,)).fetchone()
        return dict(row) if row else None

    def prior_runs_of_key(self, key: str, run_id: str) -> List[str]:
        with self._conn() as conn:
            rows = conn.execute(
                'SELECT DISTINCT run_id FROM candidates WHERE candidate_key=? AND run_id<>?',
                (key, run_id)).fetchall()
        return [r['run_id'] for r in rows]

    def get_candidate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute('SELECT * FROM candidates WHERE candidate_id=?',
                               (candidate_id,)).fetchone()
        return dict(row) if row else None

    def list_candidates(self, run_id: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute('SELECT * FROM candidates WHERE run_id=? ORDER BY created_at',
                                (run_id,)).fetchall()
        return [dict(r) for r in rows]

    def set_state(self, candidate_id: str, state: str) -> None:
        with self._lock, self._conn() as conn:
            conn.execute('UPDATE candidates SET state=? WHERE candidate_id=?',
                         (state, candidate_id))

    # ── attempt / evaluation / decision ──────────────────────
    def add_attempt(self, candidate_id: str, stage: str, status: str,
                    message: str = '', artifact_dir: str = '') -> str:
        attempt_id = _uid('att')
        with self._lock, self._conn() as conn:
            conn.execute(
                'INSERT INTO attempts (attempt_id, candidate_id, stage, status, message,'
                ' artifact_dir, created_at) VALUES (?,?,?,?,?,?,?)',
                (attempt_id, candidate_id, stage, status, message[:2000], artifact_dir, _now()),
            )
        return attempt_id

    def add_evaluation(self, candidate_id: str, split: str, gate: Dict[str, Any],
                       metrics: Dict[str, Any], verdict: str, reasons: List[str]) -> str:
        eval_id = _uid('ev')
        with self._lock, self._conn() as conn:
            conn.execute(
                'INSERT INTO evaluations (eval_id, candidate_id, split, gate_json,'
                ' metrics_json, verdict, reasons_json, created_at) VALUES (?,?,?,?,?,?,?,?)',
                (eval_id, candidate_id, split,
                 json.dumps(gate, ensure_ascii=False, default=str),
                 json.dumps(metrics, ensure_ascii=False, default=str),
                 verdict, json.dumps(reasons, ensure_ascii=False), _now()),
            )
        return eval_id

    def add_decision(self, candidate_id: str, decision: str, rationale: str) -> str:
        decision_id = _uid('dec')
        with self._lock, self._conn() as conn:
            conn.execute(
                'INSERT INTO decisions (decision_id, candidate_id, decision, rationale,'
                ' created_at) VALUES (?,?,?,?,?)',
                (decision_id, candidate_id, decision, rationale, _now()),
            )
        return decision_id

    def list_evaluations(self, candidate_id: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute('SELECT * FROM evaluations WHERE candidate_id=? ORDER BY created_at',
                                (candidate_id,)).fetchall()
        return [dict(r) for r in rows]

    # ── memory ───────────────────────────────────────────────
    def add_memory(self, record: MemoryRecord) -> str:
        payload = record.to_dict()
        with self._lock, self._conn() as conn:
            conn.execute(
                'INSERT INTO memories (memory_id, run_id, family, failure_type, evidence,'
                ' rule, source, evidence_level, restart_condition, tags, created_at)'
                ' VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                (payload['memory_id'], payload['run_id'], payload['family'],
                 payload['failure_type'], payload['evidence'], payload['rule'],
                 payload['source'], payload['evidence_level'],
                 payload['restart_condition'], payload['tags'], payload['created_at']),
            )
        return payload['memory_id']

    def search_memory(self, keyword: str = '', failure_type: str = '',
                      limit: int = 20) -> List[Dict[str, Any]]:
        sql = 'SELECT * FROM memories WHERE 1=1'
        params: List[Any] = []
        if keyword:
            sql += ' AND (family LIKE ? OR evidence LIKE ? OR rule LIKE ? OR tags LIKE ?)'
            params += [f'%{keyword}%'] * 4
        if failure_type:
            sql += ' AND failure_type=?'
            params.append(failure_type)
        sql += ' ORDER BY created_at DESC LIMIT ?'
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [dict(r) for r in rows]

    # ── access / budget ─────────────────────────────────────
    def log_access(self, actor: str, resource: str, mode: str, allowed: bool) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                'INSERT INTO access_events (event_id, actor, resource, mode, allowed, created_at)'
                ' VALUES (?,?,?,?,?,?)',
                (_uid('acc'), actor, resource, mode, int(allowed), _now()),
            )

    def log_budget(self, run_id: str, kind: str, n: int = 1) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                'INSERT INTO budget_events (event_id, run_id, kind, n, created_at)'
                ' VALUES (?,?,?,?,?)', (_uid('bud'), run_id, kind, n, _now()),
            )

    def budget_usage(self, run_id: str) -> Dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute(
                'SELECT kind, SUM(n) AS total FROM budget_events WHERE run_id=? GROUP BY kind',
                (run_id,)).fetchall()
        return {r['kind']: int(r['total'] or 0) for r in rows}

    # ── report ───────────────────────────────────────────────
    def report(self, run_id: str) -> Dict[str, Any]:
        run = self.get_run(run_id) or {}
        hyps = self.list_hypotheses(run_id)
        cands = self.list_candidates(run_id)
        states: Dict[str, int] = {}
        for c in cands:
            states[c['state']] = states.get(c['state'], 0) + 1
        evals = []
        for c in cands:
            evals.extend(self.list_evaluations(c['candidate_id']))
        verdicts: Dict[str, int] = {}
        for e in evals:
            verdicts[e['verdict']] = verdicts.get(e['verdict'], 0) + 1
        return {
            'run': run,
            'n_hypotheses': len(hyps),
            'n_candidates': len(cands),
            'states': states,
            'verdicts': verdicts,
            'budget': self.budget_usage(run_id),
            'memories': len(self.search_memory(limit=1000)),
            'candidates': [
                {'candidate_id': c['candidate_id'], 'family': c['family'],
                 'state': c['state'], 'direction': c['direction'],
                 'hypothesis_id': c['hypothesis_id']}
                for c in cands
            ],
        }

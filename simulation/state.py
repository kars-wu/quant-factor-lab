"""
模拟盘状态存储：SQLite 持久化（strategy 状态 / 净值历史 / 持仓 / 交易记录）。

表结构：
  sim_strategies  — 策略注册与运行状态（当前持仓 JSON、现金、初始资金、股票池配置）
  sim_nav         — 每策略每日净值（date, nav, cash, market_value）
  sim_trades      — 每策略交易记录（信号日/买入日/卖出日/价格/数量/收益/费用）
  sim_runs        — 每日推进日志（每策略最后推进日期、状态、错误信息）
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "sim_state.db"
LOCK_PATH = Path(__file__).parent / "data" / "sim_run.lock"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── 运行进度（进程内共享, 供前端轮询）──
_RUN_PROGRESS: dict = {
    'running': False,
    'stage': '',           # 'idle' | 'fetch' | 'load' | 'signal' | 'advance' | 'done' | 'error'
    'stage_label': '',
    'current': 0,
    'total': 0,
    'percent': 0,
    'phase_current': 0,
    'phase_total': 0,
    'message': '',
    'started_at': '',
    'finished_at': '',
}


def set_progress(stage: str, label: str, current: int = 0, total: int = 0,
                 message: str = '', percent: int | None = None):
    from time import time
    _RUN_PROGRESS.setdefault('_stage_started_at', time())
    if stage != _RUN_PROGRESS.get('stage'):
        _RUN_PROGRESS['_stage_started_at'] = time()
    _RUN_PROGRESS['stage'] = stage
    _RUN_PROGRESS['stage_label'] = label
    _RUN_PROGRESS['current'] = current
    _RUN_PROGRESS['total'] = total
    _RUN_PROGRESS['phase_current'] = current
    _RUN_PROGRESS['phase_total'] = total
    if percent is not None:
        _RUN_PROGRESS['percent'] = percent
    elif total > 0:
        _RUN_PROGRESS['percent'] = int(min(100, current * 100 / total))
    if message:
        _RUN_PROGRESS['message'] = message


def mark_running():
    from datetime import datetime
    _RUN_PROGRESS['running'] = True
    _RUN_PROGRESS['started_at'] = datetime.now().strftime('%H:%M:%S')
    _RUN_PROGRESS['finished_at'] = ''
    _RUN_PROGRESS['percent'] = 0


def mark_done(message: str = '完成'):
    from datetime import datetime
    _RUN_PROGRESS['running'] = False
    _RUN_PROGRESS['stage'] = 'done'
    _RUN_PROGRESS['stage_label'] = message
    _RUN_PROGRESS['percent'] = 100
    _RUN_PROGRESS['finished_at'] = datetime.now().strftime('%H:%M:%S')
    _RUN_PROGRESS['message'] = message


def mark_error(message: str):
    from datetime import datetime
    _RUN_PROGRESS['running'] = False
    _RUN_PROGRESS['stage'] = 'error'
    _RUN_PROGRESS['stage_label'] = '出错'
    _RUN_PROGRESS['finished_at'] = datetime.now().strftime('%H:%M:%S')
    _RUN_PROGRESS['message'] = message


def get_progress() -> dict:
    from time import time
    out = dict(_RUN_PROGRESS)
    t0 = out.get('_stage_started_at')
    if out.get('running') and t0:
        elapsed = max(int(time() - float(t0)), 0)
        out['stage_elapsed_seconds'] = elapsed
    return out


class SimLock:
    """模拟盘运行锁：防止 bootstrap 与 daily/API 并发写库。

    锁文件内容为持有进程 PID；若持有进程已退出（如 API 被 kill 中断），
    残留锁会被判定为陈旧并在下一次 acquire 时自动清除，避免永久卡死。
    """
    def __init__(self):
        self._acquired = False

    @staticmethod
    def _is_stale() -> bool:
        """锁文件存在但持有进程已不存在 → 陈旧锁。"""
        try:
            txt = LOCK_PATH.read_text().strip()
        except (OSError, IOError):
            return False
        if not txt.isdigit():
            return False
        pid = int(txt)
        if pid == os.getpid():
            return False
        try:
            os.kill(pid, 0)
            return False          # 持有进程存活
        except ProcessLookupError:
            return True           # 进程已不存在 → 陈旧
        except PermissionError:
            return False

    def acquire(self) -> bool:
        try:
            fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            self._acquired = True
            return True
        except FileExistsError:
            if self._is_stale():
                try:
                    os.remove(str(LOCK_PATH))
                except FileNotFoundError:
                    pass
                return self.acquire()
            return False

    def release(self):
        if self._acquired:
            try:
                os.remove(str(LOCK_PATH))
            except FileNotFoundError:
                pass
            self._acquired = False

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError(f'模拟盘已在运行（锁文件 {LOCK_PATH} 存在），请稍后重试')
        return self

    def __exit__(self, *exc):
        self.release()


def is_locked() -> bool:
    """检查模拟盘是否已有其他进程持有运行锁。"""
    if not LOCK_PATH.exists():
        return False
    if SimLock._is_stale():
        try:
            LOCK_PATH.unlink()
        except FileNotFoundError:
            pass
        return False
    return True


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS sim_strategies (
            strategy_id   TEXT PRIMARY KEY,
            name          TEXT NOT NULL,
            description   TEXT,
            initial_capital REAL NOT NULL,
            pool          TEXT NOT NULL,
            positions     TEXT NOT NULL DEFAULT '{}',   -- {symbol: {qty, avg_cost, open_date}}
            pending_buys  TEXT NOT NULL DEFAULT '[]',  -- open-open原子换仓的次日开盘买入计划
            cash          REAL NOT NULL,
            extra_state   TEXT NOT NULL DEFAULT '{}',
            created_at    TEXT DEFAULT (datetime('now','localtime')),
            last_run_date TEXT
        );
        CREATE TABLE IF NOT EXISTS sim_nav (
            strategy_id TEXT NOT NULL,
            date        TEXT NOT NULL,
            nav         REAL NOT NULL,
            cash        REAL NOT NULL,
            market_value REAL NOT NULL,
            PRIMARY KEY (strategy_id, date)
        );
        CREATE TABLE IF NOT EXISTS sim_trades (
            strategy_id TEXT NOT NULL,
            trade_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT NOT NULL,
            signal_date TEXT,
            buy_date    TEXT NOT NULL,
            sell_date   TEXT,
            buy_price   REAL,
            sell_price  REAL,
            quantity    INTEGER,
            buy_cost    REAL,
            sell_cost   REAL,
            pnl         REAL,
            return_pct  REAL,
            direction   TEXT NOT NULL,   -- 'long' | 'short'(多空腿)
            exit_reason TEXT
        );
        CREATE TABLE IF NOT EXISTS sim_runs (
            strategy_id TEXT NOT NULL,
            run_date    TEXT NOT NULL,
            status      TEXT NOT NULL,
            message     TEXT,
            PRIMARY KEY (strategy_id, run_date)
        );
        CREATE TABLE IF NOT EXISTS sim_preselect (
            strategy_id TEXT NOT NULL,
            signal_date TEXT NOT NULL,
            rank        INTEGER NOT NULL,
            symbol      TEXT NOT NULL,
            score       REAL,
            buy_date    TEXT,
            PRIMARY KEY (strategy_id, signal_date, symbol)
        );
        CREATE TABLE IF NOT EXISTS sim_benchmarks (
            benchmark_id TEXT NOT NULL,
            date        TEXT NOT NULL,
            nav         REAL NOT NULL,
            PRIMARY KEY (benchmark_id, date)
        );
        CREATE TABLE IF NOT EXISTS sim_timing (
            strategy_id TEXT NOT NULL,
            signal_date TEXT NOT NULL,
            breadth     REAL,
            allowed     INTEGER NOT NULL,
            PRIMARY KEY (strategy_id, signal_date)
        );
        """)
        # 兼容已有库：为旧 schema 增加 pending_buys 字段
    cols = {r[1] for r in conn.execute("PRAGMA table_info(sim_strategies)").fetchall()}
    if 'extra_state' not in cols:
        conn.execute("ALTER TABLE sim_strategies ADD COLUMN extra_state TEXT NOT NULL DEFAULT '{}'")
    if 'pending_buys' not in cols:
        conn.execute("ALTER TABLE sim_strategies ADD COLUMN pending_buys TEXT NOT NULL DEFAULT '[]'")



def record_preselect(strategy_id: str, signal_date: str, buy_date: str,
                     picks: list[tuple[str, float]]) -> None:
    """记录某策略某信号日的预选 topK（T 日信号 → T+1 买入目标）。"""
    init_db()
    with get_conn() as conn:
        conn.execute("DELETE FROM sim_preselect WHERE strategy_id=? AND signal_date=?", (strategy_id, signal_date))
        for i, (sym, score) in enumerate(picks, 1):
            conn.execute(
                "INSERT OR REPLACE INTO sim_preselect (strategy_id, signal_date, rank, symbol, score, buy_date) "
                "VALUES (?,?,?,?,?,?)",
                (strategy_id, signal_date, i, sym, score, buy_date),
            )


def get_preselect_dates(strategy_id: str, limit: int = 2) -> list[str]:
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT signal_date FROM sim_preselect WHERE strategy_id=? ORDER BY signal_date DESC LIMIT ?",
            (strategy_id, limit)).fetchall()
    return [r[0] for r in rows]


def get_preselect(strategy_id: str, signal_date: str | None = None, limit: int = 20) -> list[dict]:
    """读取某策略预选；signal_date=None 取最新一日。"""
    init_db()
    with get_conn() as conn:
        if signal_date is None:
            row = conn.execute(
                "SELECT MAX(signal_date) FROM sim_preselect WHERE strategy_id=?", (strategy_id,)).fetchone()
            if row is None or row[0] is None:
                return []
            signal_date = row[0]
        rows = conn.execute(
            "SELECT * FROM sim_preselect WHERE strategy_id=? AND signal_date=? ORDER BY rank LIMIT ?",
            (strategy_id, signal_date, limit)).fetchall()
    return [dict(r) for r in rows]


def get_held_symbols_as_of(strategy_id: str, date: str) -> set[str]:
    """按交易流水还原某信号日收盘后的持仓 symbol 集合。

    sim_trades 中开仓和平仓是两条记录（开仓记录不回填 sell_date），
    因此不能用“买入数量-卖出数量”简单累加；这里取每只股票截至该日
    最新一笔有效交易，若最新记录仍为 open，则视为当时持有。
    """
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            """
            WITH latest AS (
                SELECT symbol, MAX(trade_id) AS trade_id
                  FROM sim_trades
                 WHERE strategy_id=?
                   AND buy_date<=?
                   AND (sell_date IS NULL OR sell_date<=?)
                 GROUP BY symbol
            )
            SELECT t.symbol
              FROM latest l
              JOIN sim_trades t ON t.trade_id=l.trade_id
             WHERE t.sell_date IS NULL
            """,
            (strategy_id, date, date),
        ).fetchall()
    return {str(r[0]) for r in rows}


# ───────────────────────────── 策略注册表 ─────────────────────────────

def register_strategy(strategy_id: str, name: str, description: str,
                      initial_capital: float, pool: str,
                      positions: dict | None = None, cash: float | None = None) -> None:
    init_db()
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO sim_strategies
               (strategy_id, name, description, initial_capital, pool, positions, cash,
                extra_state, created_at)
               VALUES (?,?,?,?,?,?,?,?, datetime('now','localtime'))""",
            (strategy_id, name, description, initial_capital, pool,
             json.dumps(positions or {}), cash if cash is not None else initial_capital,
             '{}'),
        )


def get_strategy(strategy_id: str) -> dict | None:
    init_db()
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM sim_strategies WHERE strategy_id=?", (strategy_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d['positions'] = json.loads(d['positions'])
    d['pending_buys'] = json.loads(d.get('pending_buys') or '[]')
    d['extra_state'] = json.loads(d.get('extra_state') or '{}')
    return d


def list_strategies() -> list[dict]:
    init_db()
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM sim_strategies ORDER BY created_at").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d['positions'] = json.loads(d['positions'])
        out.append(d)
    return out


def update_strategy_state(strategy_id: str, positions: dict, cash: float, last_run_date: str) -> None:
    update_strategy_state_with_pending(strategy_id, positions, cash, last_run_date, [])


def update_strategy_state_with_pending(strategy_id: str, positions: dict, cash: float,
                                       last_run_date: str,
                                       pending_buys: list | None = None,
                                       extra_state: dict | None = None) -> None:
    """Persist strategy state, including optional extra_state."""
    if extra_state is None:
        row = get_strategy(strategy_id)
        extra_state = row.get('extra_state', {}) if row else {}
    update_strategy_state_extra(strategy_id, last_run_date,
                                {'extra_state': extra_state})
    with get_conn() as conn:
        conn.execute(
            "UPDATE sim_strategies SET positions=?, pending_buys=?, cash=?, last_run_date=? WHERE strategy_id=?",
            (json.dumps(positions), json.dumps(pending_buys or []), cash,
             last_run_date, strategy_id),
        )
    with get_conn() as conn:
        conn.execute(
            "UPDATE sim_strategies SET positions=?, pending_buys=?, cash=?, last_run_date=? WHERE strategy_id=?",
            (json.dumps(positions), json.dumps(pending_buys or []), cash,
             last_run_date, strategy_id),
        )


def update_strategy_state_extra(strategy_id: str, last_run_date: str,
                                extra: dict) -> None:
    """Persist extra JSON fields on a strategy row without touching positions/cash."""
    cols = {r[1] for r in _conn_columns('sim_strategies')}
    assignments = []
    values = []
    for k, v in extra.items():
        if k not in cols:
            continue
        assignments.append(f'{k}=?')
        values.append(v if isinstance(v, str) else json.dumps(v))
    if not assignments:
        return
    values.append(last_run_date)
    values.append(strategy_id)
    with get_conn() as conn:
        conn.execute(
            f"UPDATE sim_strategies SET {', '.join(assignments)}, last_run_date=? WHERE strategy_id=?",
            tuple(values),
        )


def _conn_columns(table: str) -> list[tuple]:
    with get_conn() as conn:
        return conn.execute(f"PRAGMA table_info({table})").fetchall()


# ───────────────────────────── 净值 / 交易 / 运行日志 ─────────────────────────────

def append_nav(strategy_id: str, date: str, nav: float, cash: float, market_value: float) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sim_nav (strategy_id, date, nav, cash, market_value) VALUES (?,?,?,?,?)",
            (strategy_id, date, nav, cash, market_value),
        )


def get_nav(strategy_id: str) -> pd.DataFrame:
    init_db()
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT date, nav, cash, market_value FROM sim_nav WHERE strategy_id=? ORDER BY date",
            conn, params=(strategy_id,))
    return df


def get_nav_between(strategy_id: str, start: str | None = None,
                    end: str | None = None) -> pd.DataFrame:
    """读取净值区间；start/end 为 YYYY-MM-DD，闭区间。"""
    init_db()
    sql = "SELECT date, nav, cash, market_value FROM sim_nav WHERE strategy_id=?"
    params: list = [strategy_id]
    if start:
        sql += " AND date>=?"
        params.append(start)
    if end:
        sql += " AND date<=?"
        params.append(end)
    sql += " ORDER BY date"
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn, params=tuple(params))


def replace_benchmark(benchmark_id: str, nav: pd.Series) -> None:
    """Replace a benchmark NAV series.

    ``nav`` must be indexed by date; values are NAV levels, not returns.
    """
    init_db()
    rows = [
        (benchmark_id, str(pd.Timestamp(d).date()), float(v))
        for d, v in nav.dropna().items()
        if np.isfinite(v)
    ]
    with get_conn() as conn:
        conn.execute("DELETE FROM sim_benchmarks WHERE benchmark_id=?", (benchmark_id,))
        conn.executemany(
            "INSERT OR REPLACE INTO sim_benchmarks(benchmark_id, date, nav) VALUES (?,?,?)",
            rows,
        )


def get_benchmark(benchmark_id: str) -> pd.DataFrame:
    init_db()
    with get_conn() as conn:
        return pd.read_sql_query(
            "SELECT date, nav FROM sim_benchmarks WHERE benchmark_id=? ORDER BY date",
            conn,
            params=(benchmark_id,),
        )


def replace_timing(strategy_id: str, breadth: pd.Series, allowed: pd.Series) -> None:
    """Replace a strategy's timing-state history."""
    init_db()
    common = breadth.index.intersection(allowed.index)
    rows = []
    for d in common:
        b = breadth.loc[d]
        a = allowed.loc[d]
        if pd.isna(b) or pd.isna(a):
            continue
        rows.append((strategy_id, str(pd.Timestamp(d).date()), float(b), int(bool(a))))
    with get_conn() as conn:
        conn.execute("DELETE FROM sim_timing WHERE strategy_id=?", (strategy_id,))
        conn.executemany(
            "INSERT OR REPLACE INTO sim_timing(strategy_id, signal_date, breadth, allowed) VALUES (?,?,?,?)",
            rows,
        )


def get_latest_timing(strategy_id: str) -> dict | None:
    init_db()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT signal_date, breadth, allowed FROM sim_timing "
            "WHERE strategy_id=? ORDER BY signal_date DESC LIMIT 1",
            (strategy_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "signal_date": str(row["signal_date"])[:10],
        "breadth": float(row["breadth"]),
        "allowed": bool(row["allowed"]),
    }


def record_trade(strategy_id: str, **kwargs) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO sim_trades (strategy_id, symbol, signal_date, buy_date, sell_date,
               buy_price, sell_price, quantity, buy_cost, sell_cost, pnl, return_pct, direction, exit_reason)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (strategy_id, kwargs.get('symbol'), kwargs.get('signal_date'), kwargs.get('buy_date'),
             kwargs.get('sell_date'), kwargs.get('buy_price'), kwargs.get('sell_price'),
             kwargs.get('quantity'), kwargs.get('buy_cost'), kwargs.get('sell_cost'),
             kwargs.get('pnl'), kwargs.get('return_pct'), kwargs.get('direction', 'long'),
             kwargs.get('exit_reason')),
        )


def has_trades(strategy_id: str) -> bool:
    """该策略是否已有任何历史成交（用于判断是否首次建仓）。"""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM sim_trades WHERE strategy_id=? LIMIT 1", (strategy_id,)).fetchone()
    return row is not None


def get_trades(strategy_id: str, limit: int = 500,
               start: str | None = None, end: str | None = None) -> list[dict]:
    init_db()
    with get_conn() as conn:
        sql = "SELECT * FROM sim_trades WHERE strategy_id=? AND sell_date IS NOT NULL"
        params: list = [strategy_id]
        if start:
            sql += " AND (buy_date>=? OR sell_date>=?)"
            params.extend([start, start])
        if end:
            sql += " AND (buy_date<=? OR sell_date<=?)"
            params.extend([end, end])
        sql += " ORDER BY buy_date DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, tuple(params)).fetchall()
    return [dict(r) for r in rows]


def get_trades_touching(strategy_id: str, start: str | None = None,
                        end: str | None = None, limit: int = 100000) -> list[dict]:
    """返回买入日或卖出日落在区间内的交易。

    用于“每日调仓”视图：即使一笔交易在区间前买入、区间内卖出，
    也必须出现在卖出日的调仓明细中。
    """
    init_db()
    sql = "SELECT * FROM sim_trades WHERE strategy_id=?"
    params: list = [strategy_id]
    if start or end:
        conditions = []
        if start:
            conditions.append("(buy_date>=? OR sell_date>=?)")
            params.extend([start, start])
        if end:
            conditions.append("(buy_date<=? OR sell_date<=?)")
            params.extend([end, end])
        sql += " AND (" + " AND ".join(conditions) + ")"
    sql += " ORDER BY buy_date DESC LIMIT ?"
    params.append(limit)
    with get_conn() as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    return [dict(r) for r in rows]


def get_trades_on(strategy_id: str, date: str) -> list[dict]:
    """返回该日买入或卖出的所有成交（用于前端展示「当日调仓」）。"""
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sim_trades WHERE strategy_id=? AND (buy_date LIKE ? OR sell_date LIKE ?)",
            (strategy_id, date + '%', date + '%')).fetchall()
    return [dict(r) for r in rows]


def get_last_run(strategy_id: str) -> dict | None:
    init_db()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM sim_runs WHERE strategy_id=? ORDER BY run_date DESC LIMIT 1",
            (strategy_id,)).fetchone()
    return dict(row) if row else None


def log_run(strategy_id: str, run_date: str, status: str, message: str = '') -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sim_runs (strategy_id, run_date, status, message) VALUES (?,?,?,?)",
            (strategy_id, run_date, status, message),
        )

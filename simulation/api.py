"""模拟盘 Web API（FastAPI）。

启动：

    uvicorn simulation.api:app --host 127.0.0.1 --port 8787

接口一览见 README。所有接口只读 sim_state.db，唯一的写操作是
``POST /api/run`` 触发的后台推进任务。
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from data.database import Database
from simulation import state
from simulation.run import register_examples, run_advance, run_bootstrap, run_daily
from simulation.strategies import all_specs, get_spec

app = FastAPI(title='quant-factor-lab · 模拟盘工作台', version='1.0.0')
app.add_middleware(
    CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'],
)

STATIC_DIR = Path(__file__).parent / 'static'

_TASKS: dict[str, threading.Thread] = {}
_LOCK = threading.Lock()


def _ensure_strategies() -> None:
    if not all_specs():
        register_examples()


def _stats(nav: pd.Series) -> dict:
    nav = nav.dropna()
    if len(nav) < 2:
        return {}
    ret = nav.pct_change().dropna()
    total = float(nav.iloc[-1] / nav.iloc[0] - 1)
    days = max((nav.index[-1] - nav.index[0]).days, 1)
    annual = float((1 + total) ** (365 / days) - 1)
    vol = float(ret.std() * np.sqrt(244)) if len(ret) > 2 else 0.0
    sharpe = float(ret.mean() / ret.std() * np.sqrt(244)) if len(ret) > 2 and ret.std() > 0 else None
    peak = nav.cummax()
    mdd = float((nav / peak - 1).min())
    return {
        'start': str(nav.index[0].date()),
        'end': str(nav.index[-1].date()),
        'total_return': round(total, 4),
        'annual_return': round(annual, 4),
        'annual_vol': round(vol, 4),
        'sharpe': round(sharpe, 3) if sharpe is not None else None,
        'max_drawdown': round(mdd, 4),
    }


def _nav_series(strategy_id: str) -> pd.Series:
    df = state.get_nav(strategy_id)
    if df.empty:
        return pd.Series(dtype=float)
    s = pd.Series(df['nav'].values, index=pd.to_datetime(df['date']))
    return s.sort_index()


def _names() -> dict[str, str]:
    db = Database()
    rows = db.query('SELECT symbol, name FROM stock_info')
    return {r[0]: r[1] for r in rows}


# ───────────────────────────── 页面 ─────────────────────────────

@app.get('/')
def index():
    index_file = STATIC_DIR / 'index.html'
    if not index_file.exists():
        raise HTTPException(404, 'index.html 缺失')
    return FileResponse(index_file)


# ───────────────────────────── 策略 ─────────────────────────────

@app.get('/api/strategies')
def list_strategies():
    _ensure_strategies()
    specs = {s.strategy_id: s for s in all_specs()}
    out = []
    for row in state.list_strategies():
        sid = row['strategy_id']
        spec = specs.get(sid)
        nav = _nav_series(sid)
        out.append({
            'strategy_id': sid,
            'name': row['name'],
            'description': row['description'],
            'pool': row['pool'],
            'initial_capital': row['initial_capital'],
            'last_run_date': row.get('last_run_date'),
            'nav': round(float(nav.iloc[-1]), 4) if len(nav) else None,
            'stats': _stats(nav),
            'position_count': len(row.get('positions') or {}),
            'direction': spec.direction if spec else None,
            'top_k': spec.top_k if spec else None,
        })
    return out


@app.get('/api/strategies/{strategy_id}/nav')
def strategy_nav(strategy_id: str, start: Optional[str] = None, end: Optional[str] = None):
    if get_spec(strategy_id) is None and state.get_strategy(strategy_id) is None:
        raise HTTPException(404, f'未知策略 {strategy_id}')
    df = state.get_nav_between(strategy_id, start, end)
    if df.empty:
        return {'strategy_id': strategy_id, 'points': [], 'stats': {}}
    nav = pd.Series(df['nav'].values, index=pd.to_datetime(df['date'])).sort_index()
    return {
        'strategy_id': strategy_id,
        'points': [{'date': str(d.date()), 'nav': round(float(v), 6)} for d, v in nav.items()],
        'stats': _stats(nav),
    }


@app.get('/api/strategies/{strategy_id}/trades')
def strategy_trades(strategy_id: str, limit: int = 200,
                    start: Optional[str] = None, end: Optional[str] = None):
    return state.get_trades(strategy_id, limit=limit, start=start, end=end)


@app.get('/api/strategies/{strategy_id}/positions')
def strategy_positions(strategy_id: str):
    row = state.get_strategy(strategy_id)
    if row is None:
        raise HTTPException(404, f'未知策略 {strategy_id}')
    names = _names()
    positions = row.get('positions') or {}
    out = []
    for symbol, legs in positions.items():
        legs = legs if isinstance(legs, list) else [legs]
        for leg in legs:
            out.append({
                'symbol': symbol,
                'name': names.get(symbol, ''),
                'quantity': leg.get('qty'),
                'avg_cost': leg.get('avg_cost'),
                'open_date': leg.get('open_date'),
            })
    out.sort(key=lambda x: (x['open_date'] or '', x['symbol']))
    return {'strategy_id': strategy_id, 'cash': row['cash'], 'positions': out}


@app.get('/api/strategies/{strategy_id}/preselect')
def strategy_preselect(strategy_id: str, limit: int = 20):
    rows = state.get_preselect(strategy_id, limit=limit)
    names = _names()
    for r in rows:
        r['name'] = names.get(r['symbol'], '')
    return rows


@app.get('/api/strategies/{strategy_id}/benchmark')
def strategy_benchmark(strategy_id: str):
    spec = get_spec(strategy_id)
    row = state.get_strategy(strategy_id)
    pool = spec.pool if spec else (row or {}).get('pool')
    if not pool:
        raise HTTPException(404, f'未知策略 {strategy_id}')
    df = state.get_benchmark(f'{pool}_equal_weight')
    if df.empty:
        raise HTTPException(404, f'基准 {pool} 尚未生成，先运行 bootstrap')
    nav = _nav_series(strategy_id)
    start = nav.index[0] if len(nav) else None
    bench = pd.Series(df['nav'].values, index=pd.to_datetime(df['date'])).sort_index()
    if start is not None:
        bench = bench[bench.index >= start]
        if len(bench):
            bench = bench / bench.iloc[0]
    return {
        'strategy_id': strategy_id,
        'pool': pool,
        'points': [{'date': str(d.date()), 'nav': round(float(v), 6)} for d, v in bench.items()],
    }


# ───────────────────────────── 运行状态 ─────────────────────────────

@app.get('/api/status')
def system_status():
    return {
        'progress': state.get_progress(),
        'locked': state.is_locked(),
        'running_tasks': [k for k, t in _TASKS.items() if t.is_alive()],
    }


class RunRequest(BaseModel):
    mode: str = 'advance'   # advance | bootstrap | daily
    start: str = '2021-01-01'
    lookback: int = 420


def _start_background(label: str, fn) -> bool:
    with _LOCK:
        t = _TASKS.get(label)
        if t is not None and t.is_alive():
            return False
        task = threading.Thread(target=fn, name=label, daemon=True)
        _TASKS[label] = task
        task.start()
    return True


@app.post('/api/run')
def trigger_run(req: RunRequest = RunRequest()):
    _ensure_strategies()
    if req.mode == 'bootstrap':
        started = _start_background('bootstrap', lambda: run_bootstrap(req.start))
    elif req.mode == 'daily':
        started = _start_background('daily', lambda: run_daily(fetch=True, lookback_days=req.lookback))
    elif req.mode == 'advance':
        started = _start_background('advance', lambda: run_advance(lookback_days=req.lookback))
    else:
        raise HTTPException(400, f'未知模式 {req.mode}')
    if not started:
        return {'status': 'already_running', 'mode': req.mode}
    return {'status': 'started', 'mode': req.mode}


def main() -> None:  # pragma: no cover
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8787)


if __name__ == '__main__':  # pragma: no cover
    main()

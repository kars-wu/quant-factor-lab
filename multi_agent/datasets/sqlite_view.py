"""只读数据库视图：把本地 SQLite 行情库转成研究用 DataView。

为什么不用 ``data.database.Database``：那个类的构造函数会执行 schema 初始化
和迁移（会写库）。研究 worker 必须拿到**只读**连接，否则「访问隔离」不成立。

用法::

    from multi_agent.datasets.sqlite_view import load_sqlite_view
    view = load_sqlite_view('data/stock.db', start='2018-01-01', universe='top_n_amount')
"""
from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd

from config.settings import DB_PATH
from multi_agent.datasets.base import DataView

_COLUMNS = ['symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume',
            'amount', 'turnover_rate', 'pe_ttm', 'pb', 'total_mv', 'circ_mv']


@contextmanager
def readonly_conn(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    path = Path(db_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f'数据库不存在: {path}')
    conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        yield conn
    finally:
        conn.close()


def _chunks(items: List[str], size: int = 800) -> Iterator[List[str]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _load_raw(conn: sqlite3.Connection, symbols: List[str], start: str, end: str) -> pd.DataFrame:
    frames = []
    for batch in _chunks(symbols):
        ph = ','.join('?' * len(batch))
        sql = f"""SELECT {', '.join(_COLUMNS)} FROM stock_daily
                  WHERE symbol IN ({ph}) AND trade_date BETWEEN ? AND ?
                  ORDER BY trade_date"""
        frames.append(pd.read_sql_query(sql, conn, params=tuple(batch) + (start, end)))
    if not frames:
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    for col in _COLUMNS[2:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _qfq_prices(conn: sqlite3.Connection, symbols: List[str], df: pd.DataFrame,
                anchor: str) -> pd.DataFrame:
    """按 hfq 因子把原始价转成以 ``anchor`` 为锚点的前复权价。"""
    factors = []
    for batch in _chunks(symbols):
        ph = ','.join('?' * len(batch))
        sql = f"""SELECT symbol, trade_date, hfq_factor FROM adj_factor
                  WHERE symbol IN ({ph}) AND trade_date <= ?"""
        factors.append(pd.read_sql_query(sql, conn, params=tuple(batch) + (anchor,)))
    if not factors:
        return df
    adj = pd.concat(factors, ignore_index=True)
    if adj.empty:
        return df
    adj['trade_date'] = pd.to_datetime(adj['trade_date'])
    adj['hfq_factor'] = pd.to_numeric(adj['hfq_factor'], errors='coerce')
    anchor_factor = (adj.sort_values('trade_date')
                     .groupby('symbol')['hfq_factor'].last())
    adj['ratio'] = adj['hfq_factor'] / adj['symbol'].map(anchor_factor)
    df = df.merge(adj[['symbol', 'trade_date', 'ratio']], on=['symbol', 'trade_date'], how='left')
    df['ratio'] = df['ratio'].fillna(1.0)
    for col in ('open', 'high', 'low', 'close'):
        df[col] = df[col] * df['ratio']
    return df.drop(columns=['ratio'])


def load_sqlite_view(
    db_path: str | Path | None = None,
    start: str = '2018-01-01',
    end: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    universe: str = 'top_n_amount',
    top_n: int = 500,
    lookback: int = 20,
    adjust: str = 'qfq',
    non_st_mask: Optional[pd.DataFrame] = None,
    include_industry: bool = True,
) -> DataView:
    """构建只读 DataView。

    参数
    ----
    universe : 'all' | 'top_n_amount' | 'none'
        ``top_n_amount`` 用过去 ``lookback`` 日平均成交额排名，并 shift(1) 避免未来函数。
    non_st_mask : 可选的 (date × symbol) 布尔表，True 表示当日非 ST/风险警示。
        必须由上游按 PIT 口径构建；不给就用全量行情（把选择效应留在数据契约里显式声明）。
    include_industry : 是否附带 ``industry_code`` 面板（供分组中性化算子使用）。
        注意：这里用的是 ``stock_info.industry`` 的**当前**分类，套用到历史上属于轻度
        未来信息；严格研究应换成按日期生效的行业变更表。
    """
    db_path = db_path or DB_PATH
    with readonly_conn(db_path) as conn:
        if end is None:
            end = conn.execute('SELECT MAX(trade_date) FROM stock_daily').fetchone()[0]
        if symbols is None:
            symbols = [r[0] for r in conn.execute(
                'SELECT DISTINCT symbol FROM stock_daily WHERE trade_date BETWEEN ? AND ?',
                (start, end),
            )]
        if not symbols:
            raise RuntimeError(f'{start} ~ {end} 无行情数据')

        raw = _load_raw(conn, symbols, start, end)
        industry_map = {}
        if include_industry:
            rows = conn.execute(
                'SELECT symbol, industry FROM stock_info WHERE industry IS NOT NULL'
            ).fetchall()
            for symbol, industry in rows:
                industry_map[symbol] = industry
        if raw.empty:
            raise RuntimeError('行情为空')
        if adjust == 'qfq':
            raw = _qfq_prices(conn, symbols, raw, end)

    def wide(col: str) -> pd.DataFrame:
        return raw.pivot(index='trade_date', columns='symbol', values=col).sort_index()

    close = wide('close')
    panels = {
        'open': wide('open'), 'high': wide('high'), 'low': wide('low'),
        'close': close, 'volume': wide('volume'), 'amount': wide('amount'),
        'turnover_rate': wide('turnover_rate'), 'mv': wide('total_mv'),
        'pb': wide('pb'), 'pe_ttm': wide('pe_ttm'),
    }
    # total_mv 缺失时用 circ_mv / 价格×股本兜底
    circ = wide('circ_mv')
    panels['mv'] = panels['mv'].fillna(circ)

    if include_industry:
        levels = {name: i for i, name in
                  enumerate(sorted(set(industry_map.values())))}
        codes = {sym: float(levels[name]) for sym, name in industry_map.items()}
        panels['industry_code'] = pd.DataFrame(
            np.tile([codes.get(s, np.nan) for s in close.columns], (len(close), 1)),
            index=close.index, columns=close.columns,
        )

    mask = close.notna() & panels['open'].notna() & (panels['open'] > 0)
    if non_st_mask is not None:
        mask = mask & non_st_mask.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)

    if universe == 'top_n_amount':
        avg_amt = panels['amount'].rolling(lookback, min_periods=max(2, lookback // 2)).mean()
        rank = avg_amt.shift(1).rank(axis=1, method='first', ascending=False)
        mask = mask & (rank <= top_n)
    elif universe not in ('all', 'none'):
        raise ValueError(f'未知 universe: {universe}')

    mask = mask.fillna(False).astype(bool)
    # 基准与策略同口径：open-to-open，避免与收盘口径混用带来的基准偏差
    bench = (panels['open'].shift(-1) / panels['open'] - 1).where(mask).mean(axis=1).dropna()

    stamp = hashlib.sha256(
        f'{Path(db_path).name}|{start}|{end}|{universe}|{top_n}|{adjust}|{len(symbols)}'.encode()
    ).hexdigest()[:12]
    return DataView(
        snapshot_id=f'sqlite-{stamp}',
        panels=panels,
        mask=mask,
        benchmark=bench,
        meta={'source': 'sqlite', 'db': str(Path(db_path).name), 'universe': universe,
              'top_n': top_n, 'adjust': adjust, 'start': start, 'end': end},
    )

"""每日行情更新后，按需补拉当天的 ST 板块成员。

幂等：基于 choice_backfill_status 表，已拉过的 (sector_code, date) 不会重复拉。
配合 simulation/run.py 的 daily 流程调用：
  from simulation.st_sync import sync_st_for_date
  sync_st_for_date(db, stamp)
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger

ST_SECTORS = [
    ("001017", "ST"),
    ("001018", "*ST"),
    ("001023", "风险警示股票"),
]

SLEEP_SEC = 0.20


def build_pit_st_mask(db, dates, symbols):
    """构建 PIT 非 ST 掩码；ST 快照缺日时沿用最近一期状态。

    返回值中 ``True`` 表示当日不属于 ST/风险警示板块。表不存在或完全
    没有可用历史快照时返回 ``None``，由调用方决定是否回退当前名称过滤。
    """
    import pandas as pd

    dates = pd.DatetimeIndex(dates)
    symbols = list(symbols)
    if dates.empty:
        return pd.DataFrame(index=dates, columns=symbols, dtype=bool)
    table = db.query_one(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='choice_st_membership'"
    )
    if not table:
        return None
    start = str(dates[0].date())
    end = str(dates[-1].date())
    prior = db.query_one(
        "SELECT MAX(trade_date) AS d FROM choice_st_membership WHERE trade_date<=?",
        (start,),
    )
    snapshot_start = prior['d'] if prior and prior['d'] else start
    rows = db.query(
        """SELECT DISTINCT trade_date, symbol FROM choice_st_membership
           WHERE trade_date BETWEEN ? AND ?""",
        (snapshot_start, end),
    )
    if not rows:
        return None
    st = pd.DataFrame(rows, columns=['date', 'symbol'])
    st['date'] = pd.to_datetime(st['date'])
    st['_v'] = True
    snapshots = st.pivot_table(
        index='date', columns='symbol', values='_v', aggfunc='max'
    ).reindex(columns=symbols).fillna(False)
    combined_dates = snapshots.index.union(dates).sort_values()
    membership = snapshots.reindex(combined_dates).ffill().reindex(dates)
    return ~membership.fillna(False).astype(bool)


def _choice_code(symbol: str) -> str:
    if symbol.startswith(("60", "68")):
        return f"{symbol}.SH"
    if symbol.startswith(("00", "30")):
        return f"{symbol}.SZ"
    raise ValueError(f"不支持的 A 股代码前缀: {symbol}")


def _norm_symbol(code: str) -> str:
    return code.split(".")[0].zfill(6)


def _ensure_tables(db) -> None:
    db.execute(
        """CREATE TABLE IF NOT EXISTS choice_st_membership (
             sector_code TEXT NOT NULL,
             sector_name TEXT NOT NULL,
             trade_date TEXT NOT NULL,
             symbol TEXT NOT NULL,
             PRIMARY KEY (sector_code, trade_date, symbol)
           )"""
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS idx_choice_st_membership_date ON choice_st_membership(trade_date)"
    )
    db.execute(
        """CREATE TABLE IF NOT EXISTS choice_backfill_status (
             task TEXT NOT NULL,
             key TEXT NOT NULL,
             updated_at TEXT NOT NULL,
             PRIMARY KEY (task, key)
           )"""
    )
    db.commit()


def _is_done(db, sector_code: str, date: str) -> bool:
    row = db.query_one(
        "SELECT 1 FROM choice_backfill_status WHERE task=? AND key=?",
        ("st_sector_date", f"{sector_code}|{date}"),
    )
    return row is not None


def _mark_done(db, sector_code: str, date: str) -> None:
    db.execute(
        "INSERT OR REPLACE INTO choice_backfill_status(task, key, updated_at) VALUES(?,?,?)",
        ("st_sector_date", f"{sector_code}|{date}", datetime.now().isoformat(timespec="seconds")),
    )


def _fetch_sector_with_retry(c, sector_code: str, date: str, max_retry: int = 5):
    for attempt in range(1, max_retry + 1):
        ret = c.sector(sector_code, date)
        if ret.ErrorCode == 0:
            return ret
        logger.warning("sector 失败 {} {} error={} {}（第 {}/{} 次）",
                       sector_code, date, ret.ErrorCode, ret.ErrorMsg, attempt, max_retry)
        if ret.ErrorCode == 10000015:
            time.sleep(65)
        elif ret.ErrorCode == 10000014:
            time.sleep(5)
        else:
            time.sleep(2 * attempt)
    return None


def sync_st_for_date(db, stamp: str, c=None) -> bool:
    """同步指定交易日的 ST 板块成员。已完成则跳过。"""
    _ensure_tables(db)
    close_after = False
    if c is None:
        from EmQuantAPI import c as choice_c
        login = choice_c.start()
        if login.ErrorCode != 0:
            logger.error("Choice 登录失败: {} {}", login.ErrorCode, login.ErrorMsg)
            return False
        c = choice_c
        close_after = True

    try:
        need_fetch = [sc for sc, _ in ST_SECTORS if not _is_done(db, sc, stamp)]
        if not need_fetch:
            logger.info("ST 已更新: {}", stamp)
            return True

        for sector_code, sector_name in ST_SECTORS:
            if _is_done(db, sector_code, stamp):
                continue
            ret = _fetch_sector_with_retry(c, sector_code, stamp)
            if ret is None:
                logger.error("ST sector 调用失败: {} {}", sector_code, stamp)
                continue
            rows = [(sector_code, sector_name, stamp, _norm_symbol(x)) for x in ret.Codes]
            if rows:
                db.execute_many(
                    "INSERT OR REPLACE INTO choice_st_membership(sector_code, sector_name, trade_date, symbol) VALUES(?,?,?,?)",
                    rows,
                )
            # 空结果也代表接口成功且当天该板块无成员；只有 ret 错误才不标记完成。
            _mark_done(db, sector_code, stamp)
            logger.info("ST 同步: {} {} -> {} 只", sector_code, stamp, len(rows))
            time.sleep(SLEEP_SEC)
        db.commit()
        return True
    finally:
        if close_after:
            c.stop()


if __name__ == "__main__":
    import argparse

    from data.database import Database

    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="交易日 YYYY-MM-DD")
    args = ap.parse_args()

    db = Database()
    sync_st_for_date(db, args.date)

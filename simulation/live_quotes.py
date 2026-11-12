"""盘中实时行情：腾讯批量快照（qt.gtimg.cn）。

用于账户详情的盘中看盘——只读实时价，不写 stock_daily、
不影响日频净值与交易逻辑。Choice 实时权限（csq/csqsnapshot）
开通后，将 fetch_live_quotes 的数据源替换为 Choice 即可，
接口签名不变。

字段口径与 simulation/run.py:_snapshot_tencent 一致
（成交量手→股 x100，成交额万→元 x1e4）。
"""
from __future__ import annotations

from datetime import datetime
from datetime import time as dtime

import requests
from loguru import logger


def _tencent_code(symbol: str) -> str:
    if symbol.startswith(("60", "68", "9")):
        return f"sh{symbol}"
    if symbol.startswith(("00", "30", "20")):
        return f"sz{symbol}"
    return f"bj{symbol}"


def is_trading_hours(now: datetime | None = None) -> bool:
    """A 股交易时段：周一至周五 09:30-11:30 / 13:00-15:00。"""
    now = now or datetime.now()
    if now.weekday() >= 5:
        return False
    t = now.time()
    am = dtime(9, 30) <= t <= dtime(11, 30)
    pm = dtime(13, 0) <= t <= dtime(15, 0)
    return am or pm


def fetch_live_quotes(symbols: list[str], timeout: float = 8.0) -> dict[str, dict]:
    """拉取指定股票的实时快照，返回 {symbol: {now, open, high, low, pre_close, time}}。

    单次 HTTP 请求（≤80 只一批，持仓通常一批即够）。失败返回空 dict，
    调用方应显示"实时行情不可用"而不是伪造价格。
    """
    if not symbols:
        return {}
    out: dict[str, dict] = {}
    batch = 80
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        codes = ",".join(_tencent_code(s) for s in chunk)
        try:
            r = requests.get(f"https://qt.gtimg.cn/q={codes}", timeout=timeout)
            r.encoding = "gbk"
        except Exception as e:  # noqa: BLE001
            logger.warning('[live] 腾讯行情请求失败: {}', str(e)[:80])
            continue
        for line in r.text.strip().split("\n"):
            if "=" not in line:
                continue
            try:
                body = line.split('"')[1]
                f = body.split("~")
                if len(f) < 47:
                    continue
                price = float(f[3])
                if price <= 0:
                    continue
                out[f[2].zfill(6)] = {
                    "now": price,
                    "open": float(f[5]) if f[5] else None,
                    "high": float(f[33]) if f[33] else None,
                    "low": float(f[34]) if f[34] else None,
                    "pre_close": float(f[4]) if f[4] else None,
                    "time": f[30] if len(f) > 30 else None,
                }
            except (ValueError, IndexError):
                continue
    return out

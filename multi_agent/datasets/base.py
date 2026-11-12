"""数据视图：研究 worker 唯一被允许接触的数据入口。

一个 ``DataView`` 是不可变的宽表集合：

    panels  : {'open': DataFrame(date × symbol), 'close': ..., 'amount': ...}
    mask    : DataFrame(bool) 可交易且在池内
    benchmark: Series 同池等权日收益（用于超额）

所有下游组件（DSL 编译、回测、Gate 审计）只接受 ``DataView``，不直接打开数据库。
这样「研究进程看不到最终保留段」才有可能通过进程/容器隔离实现。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class DataView:
    snapshot_id: str
    panels: Dict[str, pd.DataFrame]
    mask: pd.DataFrame
    benchmark: pd.Series
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        idx = self.panels['close'].index
        cols = self.panels['close'].columns
        for name, df in self.panels.items():
            if not df.index.equals(idx) or list(df.columns) != list(cols):
                raise ValueError(f'面板 {name} 与 close 的索引/列不一致')
        if not self.mask.index.equals(idx) or list(self.mask.columns) != list(cols):
            raise ValueError('mask 与面板索引不一致')
        self.mask = self.mask.astype(bool)

    # ── 基本属性 ──────────────────────────────────────────────
    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.panels['close'].index)

    @property
    def symbols(self) -> List[str]:
        return list(self.panels['close'].columns)

    # ── 切片 ─────────────────────────────────────────────────
    def slice(self, start: Optional[str] = None, end: Optional[str] = None) -> 'DataView':
        lo = pd.Timestamp(start) if start else self.dates[0]
        hi = pd.Timestamp(end) if end else self.dates[-1]
        sel = (self.dates >= lo) & (self.dates <= hi)
        panels = {k: v.loc[sel] for k, v in self.panels.items()}
        bench = self.benchmark[(self.benchmark.index >= lo) & (self.benchmark.index <= hi)]
        suffix = f'[{start or self.dates[0].date()}..{end or self.dates[-1].date()}]'
        return DataView(self.snapshot_id + suffix, panels, self.mask.loc[sel], bench,
                        {**self.meta, 'slice': suffix})

    def cutoff(self, date: str) -> 'DataView':
        """前缀截断：仅保留 <= date 的数据（用于前缀不变性检查）。"""
        return self.slice(None, date)

    def with_panel(self, name: str, values: pd.DataFrame) -> 'DataView':
        panels = dict(self.panels)
        panels[name] = values.reindex(index=self.dates, columns=self.symbols)
        return DataView(self.snapshot_id, panels, self.mask.copy(), self.benchmark,
                        dict(self.meta))

    # ── 常用取数 ─────────────────────────────────────────────
    def open_px(self) -> pd.DataFrame:
        return self.panels['open']

    def close_px(self) -> pd.DataFrame:
        return self.panels['close']

    def amount(self) -> pd.DataFrame:
        return self.panels.get('amount', self.panels['close'] * np.nan)

    def log_mv(self) -> Optional[pd.DataFrame]:
        mv = self.panels.get('mv')
        if mv is None:
            return None
        return np.log(mv.where(mv > 0))

    def info(self) -> Dict[str, object]:
        return {
            'snapshot_id': self.snapshot_id,
            'n_dates': int(len(self.dates)),
            'n_symbols': int(len(self.symbols)),
            'start': str(self.dates[0].date()),
            'end': str(self.dates[-1].date()),
            'avg_universe': float(self.mask.sum(axis=1).mean()),
            'meta': self.meta,
        }

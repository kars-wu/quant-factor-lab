"""
DataLoader：从 SQLite 加载 OHLCV 数据为 pandas DataFrame。

前复权锚点规则：
  以回测结束日（end_date）为锚点，而非"今天"。
  这样保证同一回测区间的结果完全可复现，
  不会因为日后新增除权事件导致历史回测结果改变。
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from data.database import Database
from data.adjfactor import apply_qfq, apply_hfq, get_anchor_factor
from config.settings import MAX_LOOKBACK_DAYS


_RESAMPLE_RULES = {
    "W": "W-FRI",   # 周线，以周五收盘为准
    "M": "ME",      # 月线，月末最后一个交易日
}

_OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "amount": "sum",
}


class DataLoader:

    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def load_ohlcv(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        adjust: str = "qfq",    # 'qfq' 前复权 | 'hfq' 后复权 | '' 不复权
        freq: str = "D",        # 'D' 日线 | 'W' 周线 | 'M' 月线
        lookback_extra: int = MAX_LOOKBACK_DAYS,
    ) -> dict[str, pd.DataFrame]:
        """
        加载多个标的的 OHLCV 数据。

        返回
        ----
        dict[symbol, DataFrame]
        DataFrame.index = trade_date（字符串，'YYYY-MM-DD'）
        DataFrame.columns = [open, high, low, close, volume, amount]

        参数
        ----
        lookback_extra : 额外预加载的历史天数，用于技术指标热身
                         （回测中 ctx.data 会按当日切片，不会泄露未来数据）
        """
        # 预留历史热身窗口
        extended_start = self._offset_date(start_date, -lookback_extra)

        result = {}
        for symbol in symbols:
            df = self._load_single(symbol, extended_start, end_date, adjust)
            if df is not None and not df.empty:
                if freq != "D":
                    df = self._resample(df, freq)
                result[symbol] = df
            else:
                logger.warning("标的 {} 无有效数据，已跳过", symbol)

        return result

    def load_benchmark(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        lookback_extra: int = MAX_LOOKBACK_DAYS,
    ) -> pd.DataFrame:
        """加载基准指数（从 index_daily 表）"""
        extended_start = self._offset_date(start_date, -lookback_extra)
        sql = """
            SELECT trade_date, open, high, low, close, volume, amount
            FROM index_daily
            WHERE symbol = ?
              AND trade_date BETWEEN ? AND ?
            ORDER BY trade_date
        """
        rows = self.db.query(sql, (symbol, extended_start, end_date))
        if not rows:
            logger.warning("基准 {} 数据为空", symbol)
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["trade_date", "open", "high", "low", "close", "volume", "amount"])
        df = df.set_index("trade_date")
        return df

    def get_trade_dates(self, start_date: str, end_date: str) -> list[str]:
        """从交易日历获取区间内所有交易日"""
        sql = """
            SELECT trade_date FROM trade_calendar
            WHERE is_open = 1 AND trade_date BETWEEN ? AND ?
            ORDER BY trade_date
        """
        rows = self.db.query(sql, (start_date, end_date))
        if not rows:
            # 降级：从 stock_daily 推断交易日
            logger.warning("交易日历为空，从 stock_daily 推断交易日")
            rows = self.db.query(
                "SELECT DISTINCT trade_date FROM stock_daily "
                "WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
                (start_date, end_date)
            )
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _load_single(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> pd.DataFrame | None:
        # 1. 读取原始价格 + 换手率（如有）
        sql_price = """
            SELECT trade_date, open, high, low, close, volume, amount,
                   turnover_rate
            FROM stock_daily
            WHERE symbol = ? AND trade_date BETWEEN ? AND ?
            ORDER BY trade_date
        """
        rows = self.db.query(sql_price, (symbol, start_date, end_date))
        if not rows:
            return None

        df = pd.DataFrame(
            rows,
            columns=["trade_date", "open", "high", "low", "close",
                     "volume", "amount", "turnover_rate"]
        )
        df = df.set_index("trade_date")
        df = df.astype({"open": float, "high": float, "low": float,
                        "close": float, "volume": float})
        # turnover_rate 可能为 NULL（旧数据），转为 float 时 None -> NaN
        df["turnover_rate"] = pd.to_numeric(df["turnover_rate"], errors="coerce")

        # 2. 注入 outstanding_share（来自 stock_shares 表，用于换手率/市值计算）
        shares_row = self.db.query_one(
            "SELECT outstanding_share FROM stock_shares WHERE symbol = ?", (symbol,)
        )
        if shares_row and shares_row["outstanding_share"]:
            df["outstanding_share"] = float(shares_row["outstanding_share"])
        else:
            df["outstanding_share"] = float("nan")

        if adjust == "":
            return df

        # 2. 读取复权因子（全量，不限日期范围，确保 anchor 可以找到）
        sql_factor = """
            SELECT trade_date, hfq_factor
            FROM adj_factor
            WHERE symbol = ?
            ORDER BY trade_date
        """
        factor_rows = self.db.query(sql_factor, (symbol,))
        if not factor_rows:
            logger.warning("标的 {} 无复权因子，返回不复权数据", symbol)
            return df

        factors = pd.Series(
            {r[0]: float(r[1]) for r in factor_rows}
        )

        # 3. 锚点因子（以 end_date 为基准，保证可复现性）
        try:
            anchor = get_anchor_factor(factors, end_date)
        except ValueError as e:
            logger.warning("获取锚点因子失败 [{}]: {}，返回不复权数据", symbol, e)
            return df

        # 4. 应用复权
        if adjust == "qfq":
            return apply_qfq(df, factors, anchor)
        elif adjust == "hfq":
            return apply_hfq(df, factors)
        else:
            return df

    def _resample(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """将日线数据重采样为周线/月线"""
        rule = _RESAMPLE_RULES.get(freq, freq)
        df_r = df.resample(rule).agg(_OHLCV_AGG).dropna(subset=["close"])
        # index 转回字符串格式
        df_r.index = df_r.index.strftime("%Y-%m-%d")
        return df_r

    @staticmethod
    def _offset_date(date_str: str, days: int) -> str:
        """将日期字符串偏移 days 天（可为负）"""
        import datetime
        d = datetime.date.fromisoformat(date_str)
        d2 = d + datetime.timedelta(days=days)
        return d2.isoformat()

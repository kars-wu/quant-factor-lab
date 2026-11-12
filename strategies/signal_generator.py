"""
选股层：SignalGenerator。

职责：
  1. 从数据库加载行情数据
  2. 调用信号函数 + 选股器，生成信号列表

输出：DataFrame，每行一个信号，包含 symbol / signal_date / buy_price /
      次日 OHLC 等，供 TradeSimulator 消费。

与交易层完全解耦：SignalGenerator 不知道止损止盈怎么算，
只负责"哪些股票在哪些日子触发了买入信号"。

用法：
    gen = SignalGenerator(db, universe, '2024-01-01', '2025-12-31')
    gen.load_data()
    signals = gen.generate(signal_fn, signal_params, screener)

    # 单标的回测 = universe 只传一只
    gen = SignalGenerator(db, ['600519'], '2024-01-01', '2025-12-31')
"""

from __future__ import annotations

import collections
from typing import Callable, Optional

import pandas as pd
from loguru import logger
from tqdm import tqdm

from data.quality import DataQualityChecker
from strategies.screener import Screener


class SignalGenerator:
    """选股信号生成器。"""

    def __init__(
        self,
        db,
        universe: list[str],
        start_date: str,
        end_date: str,
        min_amount: float = 5e7,            # 最低日均成交额
        exclude_st: bool = True,
        min_listed_days: int = 250,         # 上市不足此天数排除
    ):
        self.db = db
        self.universe = universe
        self.start_date = start_date
        self.end_date = end_date
        self.min_amount = min_amount
        self.exclude_st = exclude_st
        self.min_listed_days = min_listed_days

        self._data: dict[str, pd.DataFrame] = {}
        self._valid_symbols: list[str] = []

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    @property
    def data(self) -> dict[str, pd.DataFrame]:
        """行情数据 {symbol: DataFrame}，供 TradeSimulator 使用。"""
        return self._data

    @property
    def valid_symbols(self) -> list[str]:
        return self._valid_symbols

    def load_data(self, lookback_days: int = 60) -> None:
        """
        从 SQLite 加载行情数据（不复权）。
        额外预加载 lookback_days 天历史，确保信号计算窗口够用。
        """
        logger.info("加载行情数据 | 股票池: {} 只 | {} ~ {}",
                    len(self.universe), self.start_date, self.end_date)

        from data.loader import DataLoader
        extended_start = DataLoader._offset_date(self.start_date, -lookback_days * 2)

        # 排除 ST
        if self.exclude_st:
            # 排除 ST 股票和已退市股票
            st_rows = self.db.query(
                "SELECT symbol FROM stock_info "
                "WHERE name LIKE '%ST%' OR is_active = 0"
            )
            st_set = {r[0] for r in st_rows}
        else:
            st_set = set()

        # 批量加载
        placeholders = ",".join("?" * len(self.universe))
        rows = self.db.query(
            f"""
            SELECT symbol, trade_date, open, high, low, close, volume, amount,
                   turnover_rate
            FROM stock_daily
            WHERE symbol IN ({placeholders})
              AND trade_date BETWEEN ? AND ?
            ORDER BY symbol, trade_date
            """,
            tuple(self.universe) + (extended_start, self.end_date)
        )
        if not rows:
            raise RuntimeError("无行情数据，请确认数据库已初始化")

        df_all = pd.DataFrame(rows, columns=[
            "symbol", "trade_date", "open", "high", "low", "close",
            "volume", "amount", "turnover_rate"
        ])
        df_all["trade_date"] = pd.to_datetime(df_all["trade_date"])
        df_all = df_all.astype({
            "open": float, "high": float, "low": float,
            "close": float, "volume": float, "amount": float,
        })
        df_all["turnover_rate"] = pd.to_numeric(df_all["turnover_rate"], errors="coerce")

        # 加载股本数据
        share_rows = self.db.query(
            f"SELECT symbol, outstanding_share FROM stock_shares WHERE symbol IN ({placeholders})",
            tuple(self.universe)
        )
        shares_map = {r[0]: float(r[1]) for r in share_rows if r[1]}

        # 加载复权因子
        factor_rows = self.db.query(
            f"""
            SELECT symbol, trade_date, hfq_factor
            FROM adj_factor
            WHERE symbol IN ({placeholders})
              AND trade_date BETWEEN ? AND ?
            ORDER BY symbol, trade_date
            """,
            tuple(self.universe) + (extended_start, self.end_date)
        )
        factors_map: dict[str, dict] = collections.defaultdict(dict)
        for r in factor_rows:
            factors_map[r[0]][r[1]] = float(r[2])

        # 按 symbol 分组，过滤
        valid = []
        for symbol, grp in df_all.groupby("symbol"):
            if symbol in st_set:
                continue
            grp = grp.set_index("trade_date").sort_index()
            if symbol in shares_map:
                grp["outstanding_share"] = shares_map[symbol]
            else:
                grp["outstanding_share"] = float("nan")

            if symbol in factors_map:
                factor_series = pd.Series(factors_map[symbol], dtype=float)
                factor_series.index = pd.to_datetime(factor_series.index)
                grp["hfq_factor"] = factor_series.reindex(grp.index)
            else:
                grp["hfq_factor"] = float("nan")

            grp.attrs["symbol"] = symbol

            # 流动性过滤 + 上市天数过滤
            in_range = grp[grp.index >= self.start_date]
            if len(in_range) < 20:
                continue
            if in_range["amount"].mean() < self.min_amount:
                continue
            # 上市天数过滤：数据总长度 < min_listed_days 的跳过
            if self.min_listed_days > 0 and len(grp) < self.min_listed_days:
                continue

            self._data[symbol] = grp
            valid.append(symbol)

        self._valid_symbols = valid
        logger.info("有效标的: {} 只（排除 ST {} 只，流动性不足等）",
                    len(valid), len(st_set))

        # 数据质量校验
        if valid:
            checker = DataQualityChecker()
            # 构建交易日历日期集合
            cal_rows = self.db.query(
                "SELECT trade_date FROM trade_calendar WHERE is_open=1 "
                "ORDER BY trade_date"
            )
            cal_dates = {r[0] for r in cal_rows} if cal_rows else None
            checker.check_batch(self._data, calendar_dates=cal_dates)

    # ------------------------------------------------------------------
    # 信号生成
    # ------------------------------------------------------------------

    def generate(
        self,
        signal_fn: Callable,
        signal_params: dict,
        screener: Optional[Screener] = None,
        buy_price_field: str = "close",
    ) -> pd.DataFrame:
        """
        对每只股票调用 signal_fn，收集所有信号。

        参数
        ----
        buy_price_field : 买入价取自信号日的哪个字段
            'close'（默认）— 收盘价，T日收盘信号→T+1开盘买入
            'open'         — 开盘价，T日开盘信号→T日开盘买入

        返回 DataFrame，列：
          symbol, signal_date, buy_price,
          next_date, next_open, next_high, next_low, next_close

        每行代表一个买入信号。buy_price 的含义由 buy_price_field 决定，
        实际买卖时序由 TradeSimulator 的模拟方法决定。
        """
        if not self._data:
            raise RuntimeError("请先调用 load_data()")

        if screener is not None:
            logger.info("选股器: {}", screener.describe())
        logger.info("生成信号...")
        records = []
        start_ts = pd.Timestamp(self.start_date)
        end_ts = pd.Timestamp(self.end_date)

        for symbol in tqdm(self._valid_symbols, desc="扫描信号", unit="只", ncols=80):
            df = self._data[symbol]
            try:
                signal_mask = signal_fn(df, **signal_params)
            except Exception as e:
                logger.debug("信号计算失败 {}: {}", symbol, e)
                continue

            if screener is not None:
                try:
                    eligible = screener.apply(df)
                    signal_mask = signal_mask & eligible
                except Exception as e:
                    logger.debug("选股器应用失败 {}: {}", symbol, e)
                    continue

            signal_mask = signal_mask & (df.index >= start_ts) & (df.index < end_ts)
            signal_dates = df.index[signal_mask]

            for sig_date in signal_dates:
                loc = df.index.get_loc(sig_date)
                if loc + 1 >= len(df):
                    continue

                next_date = df.index[loc + 1]

                # 跳过停牌复牌（自然日间隔 > 14 天）
                if (next_date - sig_date).days > 14:
                    continue

                # 跳过除权日
                if "hfq_factor" in df.columns:
                    f_t = df["hfq_factor"].get(sig_date)
                    f_t1 = df["hfq_factor"].get(next_date)
                    if (pd.notna(f_t) and pd.notna(f_t1)
                            and f_t > 0 and f_t1 / f_t > 1.005):
                        continue

                buy_price = float(df.loc[sig_date, buy_price_field])
                next_open = float(df.loc[next_date, "open"])
                next_high = float(df.loc[next_date, "high"])
                next_close = float(df.loc[next_date, "close"])
                next_low = float(df.loc[next_date, "low"])

                if buy_price <= 0 or next_open <= 0:
                    continue

                # 过滤买入日涨停
                prev_loc = loc - 1
                if prev_loc >= 0:
                    prev_close = float(df.iloc[prev_loc]["close"])
                    limit_up = prev_close * 1.099
                    if buy_price >= limit_up:
                        continue

                records.append({
                    "symbol": symbol,
                    "signal_date": sig_date,
                    "buy_price": buy_price,
                    "next_date": next_date,
                    "next_open": next_open,
                    "next_high": next_high,
                    "next_low": next_low,
                    "next_close": next_close,
                })

        if not records:
            logger.warning("无信号，请放宽选股条件")
            return pd.DataFrame()

        logger.info("共 {} 个原始信号", len(records))
        return pd.DataFrame(records)

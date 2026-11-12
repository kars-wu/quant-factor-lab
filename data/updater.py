"""
增量更新调度器。

初始化（--init）：
  1. 建表
  2. 拉取股票/ETF 列表写入 stock_info
  3. 拉取交易日历写入 trade_calendar
  4. 对每只标的全量拉取日线 + 复权因子

增量更新（--incremental，每日盘后执行）：
  1. 确认今日是交易日
  2. 找出 last_price_date < 今日的标的
  3. 增量拉取 (last_price_date+1, 今日) 的日线数据
  4. 全量覆盖复权因子（除权日会修改当日及之后所有因子）
"""

import datetime
from typing import Optional

import pandas as pd
from tqdm import tqdm
from loguru import logger

from data.database import Database
from data.fetchers.akshare_fetcher import AkShareFetcher
from config.settings import DEFAULT_START_DATE


class DataUpdater:

    def __init__(self, db: Database, fetcher: AkShareFetcher):
        self.db = db
        self.fetcher = fetcher

    # ------------------------------------------------------------------
    # 标准化辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_market(symbol: str, asset_type: str = "stock") -> str:
        """根据代码与资产类型推断市场。"""
        symbol = str(symbol).strip().zfill(6)

        if asset_type == "etf":
            # 与 fetchers.akshare_fetcher._etf_prefix 保持一致：15/16 深市，其余 5xxxxx 默认沪市
            return "SZ" if symbol.startswith(("15", "16")) else "SH"

        if asset_type == "index":
            return "SZ" if symbol.startswith(("399", "880")) else "SH"

        return "SH" if symbol.startswith(("6", "9")) else "SZ"

    @staticmethod
    def _normalize_date(value) -> str:
        """统一日期格式为 YYYY-MM-DD；空值返回空字符串。"""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""

        s = str(value).strip()
        if not s or s.lower() == "nan":
            return ""

        s = s.replace("/", "-")
        if len(s) == 8 and s.isdigit():
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        return s

    def _upsert_stock_info(self, records: list[dict]):
        """按 (symbol, market) UPSERT 主数据，避免错误元数据长期残留。"""
        today = datetime.date.today().isoformat()
        params = []
        for rec in records:
            symbol = str(rec.get("symbol", "")).strip().zfill(6)
            asset_type = str(rec.get("asset_type", "stock")).strip() or "stock"
            if not symbol:
                continue

            market = str(rec.get("market", "")).strip().upper() or self._infer_market(symbol, asset_type)
            params.append((
                symbol,
                str(rec.get("name", symbol)).strip() or symbol,
                market,
                asset_type,
                str(rec.get("sector", "") or "").strip(),
                str(rec.get("industry", "") or "").strip(),
                self._normalize_date(rec.get("list_date")),
                self._normalize_date(rec.get("delist_date")),
                int(rec.get("is_active", 1) or 1),
                str(rec.get("updated_at", today) or today),
            ))

        if not params:
            return

        sql = (
            "INSERT INTO stock_info(symbol, name, market, asset_type, sector, industry, "
            "list_date, delist_date, is_active, updated_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(symbol, market) DO UPDATE SET "
            "name=excluded.name, "
            "asset_type=excluded.asset_type, "
            "sector=CASE WHEN excluded.sector <> '' THEN excluded.sector ELSE stock_info.sector END, "
            "industry=CASE WHEN excluded.industry <> '' THEN excluded.industry ELSE stock_info.industry END, "
            "list_date=CASE WHEN excluded.list_date <> '' THEN excluded.list_date ELSE stock_info.list_date END, "
            "delist_date=CASE WHEN excluded.delist_date <> '' THEN excluded.delist_date ELSE stock_info.delist_date END, "
            "is_active=excluded.is_active, "
            "updated_at=excluded.updated_at"
        )
        self.db.execute_many(sql, params)

    def _ensure_update_log_rows(self, symbols: list[str]):
        if not symbols:
            return
        params = [(str(s).strip().zfill(6),) for s in symbols]
        self.db.execute_many(
            "INSERT OR IGNORE INTO update_log(symbol, load_status, updated_at) "
            "VALUES(?, 'pending', datetime('now','localtime'))",
            params,
        )

    def _update_log_status(
        self,
        symbol: str,
        status: str,
        last_price_date: Optional[str] = None,
        last_factor_date: Optional[str] = None,
        last_error: Optional[str] = None,
    ):
        current = self.db.query_one(
            "SELECT last_price_date, last_factor_date FROM update_log WHERE symbol = ?",
            (symbol,),
        )
        price_date = last_price_date if last_price_date is not None else (current["last_price_date"] if current else None)
        factor_date = last_factor_date if last_factor_date is not None else (current["last_factor_date"] if current else None)

        self.db.execute(
            "INSERT INTO update_log(symbol, last_price_date, last_factor_date, load_status, last_error, updated_at) "
            "VALUES(?, ?, ?, ?, ?, datetime('now','localtime')) "
            "ON CONFLICT(symbol) DO UPDATE SET "
            "last_price_date=excluded.last_price_date, "
            "last_factor_date=excluded.last_factor_date, "
            "load_status=excluded.load_status, "
            "last_error=excluded.last_error, "
            "updated_at=excluded.updated_at",
            (symbol, price_date, factor_date, status, last_error),
        )

    def _record_quality_issue(
        self,
        symbol: str,
        trade_date: Optional[str],
        table_name: str,
        issue_type: str,
        detail: str,
    ):
        self.db.execute(
            "INSERT INTO data_quality_log(symbol, trade_date, table_name, issue_type, detail) "
            "SELECT ?, ?, ?, ?, ? "
            "WHERE NOT EXISTS ("
            "SELECT 1 FROM data_quality_log WHERE symbol=? AND COALESCE(trade_date,'')=COALESCE(?, '') "
            "AND table_name=? AND issue_type=? AND COALESCE(detail,'')=COALESCE(?, '')"
            ")",
            (
                symbol, trade_date, table_name, issue_type, detail,
                symbol, trade_date, table_name, issue_type, detail,
            ),
        )

    def repair_stock_info_markets(self) -> dict:
        """修正 stock_info 中与代码规则不一致的 market，并清理重复脏记录。"""
        rows = self.db.query(
            "SELECT symbol, market, asset_type, name, sector, industry, list_date, delist_date, is_active, updated_at "
            "FROM stock_info"
        )

        fixed = 0
        merged = 0
        skipped = 0

        for row in rows:
            symbol = row["symbol"]
            current_market = row["market"]
            asset_type = row["asset_type"]
            target_market = self._infer_market(symbol, asset_type)

            if current_market == target_market:
                continue

            target = self.db.query_one(
                "SELECT 1 FROM stock_info WHERE symbol=? AND market=?",
                (symbol, target_market),
            )

            payload = {
                "symbol": symbol,
                "market": target_market,
                "asset_type": asset_type,
                "name": row["name"],
                "sector": row["sector"],
                "industry": row["industry"],
                "list_date": row["list_date"],
                "delist_date": row["delist_date"],
                "is_active": row["is_active"],
                "updated_at": row["updated_at"],
            }

            if target:
                self._upsert_stock_info([payload])
                self.db.execute(
                    "DELETE FROM stock_info WHERE symbol=? AND market=?",
                    (symbol, current_market),
                )
                merged += 1
            else:
                self.db.execute(
                    "UPDATE stock_info SET market=?, updated_at=datetime('now','localtime') "
                    "WHERE symbol=? AND market=?",
                    (target_market, symbol, current_market),
                )
                fixed += 1

        self.db.commit()
        return {
            "fixed": fixed,
            "merged": merged,
            "skipped": skipped,
        }

    def backfill_update_log(self) -> dict:
        """按现有事实表回填/修正 update_log。"""
        symbols = [r[0] for r in self.db.query("SELECT DISTINCT symbol FROM stock_info")]
        self._ensure_update_log_rows(symbols)

        updated = 0
        pending = 0
        failed_kept = 0

        for symbol in symbols:
            price_row = self.db.query_one(
                "SELECT MAX(trade_date) AS last_date FROM ("
                "SELECT trade_date FROM stock_daily WHERE symbol=? "
                "UNION ALL "
                "SELECT trade_date FROM index_daily WHERE symbol=?"
                ")",
                (symbol, symbol),
            )
            factor_row = self.db.query_one(
                "SELECT MAX(trade_date) AS last_date FROM adj_factor WHERE symbol=?",
                (symbol,),
            )
            existing = self.db.query_one(
                "SELECT load_status, last_error FROM update_log WHERE symbol=?",
                (symbol,),
            )

            last_price = price_row["last_date"] if price_row else None
            last_factor = factor_row["last_date"] if factor_row else None

            if existing and existing["load_status"] == "failed" and not last_price:
                status = "failed"
                error = existing["last_error"]
                failed_kept += 1
            elif last_price:
                status = "success"
                error = None
                updated += 1
            else:
                status = "pending"
                error = None
                pending += 1

            self._update_log_status(
                symbol,
                status,
                last_price_date=last_price,
                last_factor_date=last_factor,
                last_error=error,
            )

        self.db.commit()
        return {
            "updated": updated,
            "pending": pending,
            "failed_kept": failed_kept,
            "total": len(symbols),
        }

    def scan_historical_bad_rows(self, table_name: str = "stock_daily", delete_invalid: bool = False) -> dict:
        """扫描已入库历史坏行情，可选择删除。"""
        if table_name not in {"stock_daily", "index_daily"}:
            raise ValueError(f"不支持扫描表: {table_name}")

        rows = self.db.query(
            f"SELECT symbol, trade_date, open, high, low, close FROM {table_name} "
            "WHERE open <= 0 OR high <= 0 OR low <= 0 OR close <= 0 "
            "OR high < low OR high < open OR high < close OR low > open OR low > close"
        )

        deleted = 0
        for row in rows:
            symbol = row["symbol"]
            trade_date = row["trade_date"]
            detail = (
                f"open={row['open']}, high={row['high']}, low={row['low']}, close={row['close']}"
            )
            self._record_quality_issue(symbol, trade_date, table_name, "historical_invalid_ohlc", detail)
            if delete_invalid:
                self.db.execute(
                    f"DELETE FROM {table_name} WHERE symbol=? AND trade_date=?",
                    (symbol, trade_date),
                )
                deleted += 1

        self.db.commit()
        return {
            "table": table_name,
            "invalid_rows": len(rows),
            "deleted": deleted,
        }

    def retry_pending_symbols(self, asset_type: Optional[str] = None, limit: Optional[int] = None) -> dict:
        """对 pending 且无价格数据的标的做小批量重试。"""
        sql = (
            "SELECT ul.symbol, si.asset_type FROM update_log ul "
            "JOIN stock_info si ON ul.symbol = si.symbol "
            "WHERE ul.load_status='pending' AND ul.last_price_date IS NULL"
        )
        params = []
        if asset_type:
            sql += " AND si.asset_type=?"
            params.append(asset_type)
        sql += " ORDER BY ul.symbol"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        rows = self.db.query(sql, tuple(params))
        if not rows:
            return {"selected": 0, "success": 0, "failed": 0, "symbols": []}

        today = datetime.date.today().isoformat()
        success = 0
        failed = 0
        selected_symbols = []

        for row in rows:
            symbol = row["symbol"]
            atype = row["asset_type"]
            selected_symbols.append(symbol)
            try:
                if atype == "index":
                    self._init_index(symbol, DEFAULT_START_DATE, today)
                else:
                    self._init_symbol(symbol, DEFAULT_START_DATE, today)

                state = self.db.query_one(
                    "SELECT last_price_date, load_status FROM update_log WHERE symbol=?",
                    (symbol,),
                )
                if state and state["last_price_date"]:
                    success += 1
                else:
                    failed += 1
                    self._update_log_status(symbol, "failed", last_error="重试后仍无价格数据")
                    self.db.commit()
            except Exception as e:
                failed += 1
                self._update_log_status(symbol, "failed", last_error=f"重试失败: {e}")
                self.db.commit()

        return {
            "selected": len(rows),
            "success": success,
            "failed": failed,
            "symbols": selected_symbols,
        }

    def get_symbols_missing_shares(self, limit: Optional[int] = None) -> list[str]:
        rows = self.db.query(
            """SELECT si.symbol FROM stock_info si
               WHERE si.asset_type='stock'
               AND NOT EXISTS (
                   SELECT 1 FROM stock_shares ss WHERE ss.symbol = si.symbol
               )
               ORDER BY si.symbol"""
        )
        symbols = [r[0] for r in rows]
        return symbols[:limit] if limit is not None else symbols

    def get_symbols_missing_valuation(self, limit: Optional[int] = None) -> list[str]:
        rows = self.db.query(
            """SELECT si.symbol FROM stock_info si
               WHERE si.asset_type='stock'
               AND NOT EXISTS (
                   SELECT 1 FROM stock_valuation sv WHERE sv.symbol = si.symbol
               )
               ORDER BY si.symbol"""
        )
        symbols = [r[0] for r in rows]
        return symbols[:limit] if limit is not None else symbols

    def get_symbols_missing_daily_extras(self, limit: Optional[int] = None) -> list[str]:
        rows = self.db.query(
            """SELECT sd.symbol
               FROM stock_daily sd
               JOIN stock_info si ON sd.symbol = si.symbol
               WHERE si.asset_type='stock'
               GROUP BY sd.symbol
               HAVING SUM(sd.turnover_rate IS NOT NULL) = 0
                   OR SUM(sd.outstanding_share IS NOT NULL) = 0
               ORDER BY sd.symbol"""
        )
        symbols = [r[0] for r in rows]
        return symbols[:limit] if limit is not None else symbols

    def backfill_daily_extras(
        self,
        symbols: Optional[list[str]] = None,
        limit: Optional[int] = None,
        start_date: str = DEFAULT_START_DATE,
    ) -> dict:
        """重拉缺失 turnover_rate/outstanding_share 的个股日线。"""
        if symbols is None:
            symbols = self.get_symbols_missing_daily_extras(limit=limit)
        elif limit is not None:
            symbols = symbols[:limit]

        today = datetime.date.today().isoformat()
        success = 0
        failed = 0
        attempted = []

        for symbol in symbols:
            attempted.append(symbol)
            try:
                df = self.fetcher.fetch_daily(symbol, start_date, today)
                if df is None or df.empty:
                    failed += 1
                    self._update_log_status(symbol, "failed", last_error="补拉日线扩展字段为空")
                    self.db.commit()
                    continue

                self._save_daily(df)
                self._save_shares(df)
                success += 1
                self._update_log_status(symbol, "success", last_error=None)
                self.db.commit()
            except Exception as e:
                failed += 1
                self._update_log_status(symbol, "failed", last_error=f"补拉日线扩展字段失败: {e}")
                self.db.commit()

        return {
            "selected": len(symbols),
            "success": success,
            "failed": failed,
            "symbols": attempted,
        }

    def _sanitize_daily_rows(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """拦截明显脏数据，避免污染主表。"""
        if df is None or df.empty:
            return pd.DataFrame() if df is None else df

        required_cols = ["symbol", "trade_date", "open", "high", "low", "close"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"{table_name} 缺少必要列: {missing_cols}")

        clean_rows = []
        bad_count = 0

        for _, row in df.iterrows():
            symbol = str(row["symbol"]).strip().zfill(6)
            trade_date = self._normalize_date(row["trade_date"])
            try:
                o = float(row["open"])
                h = float(row["high"])
                l = float(row["low"])
                c = float(row["close"])
            except Exception:
                detail = "OHLC 无法转换为数值"
                self._record_quality_issue(symbol, trade_date, table_name, "invalid_numeric", detail)
                bad_count += 1
                continue

            detail = None
            if min(o, h, l, c) <= 0:
                detail = f"存在非正价格 open={o}, high={h}, low={l}, close={c}"
            elif h < l:
                detail = f"high({h}) < low({l})"
            elif h < max(o, c):
                detail = f"high({h}) < max(open, close)({max(o, c)})"
            elif l > min(o, c):
                detail = f"low({l}) > min(open, close)({min(o, c)})"

            if detail:
                self._record_quality_issue(symbol, trade_date, table_name, "invalid_ohlc", detail)
                bad_count += 1
                continue

            clean_rows.append(row)

        if bad_count:
            logger.warning("{} 检测到 {} 行坏数据，已跳过写入", table_name, bad_count)

        if not clean_rows:
            return df.iloc[0:0].copy()
        return pd.DataFrame(clean_rows).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 初始化（首次运行）
    # ------------------------------------------------------------------

    def init_all(
        self,
        symbols: Optional[list[str]] = None,
        start_date: str = DEFAULT_START_DATE,
        include_etf: bool = True,
        resume: bool = True,
    ):
        """
        全量初始化，支持断点续传。

        symbols=None 时初始化全市场；传入列表时只初始化指定标的。
        resume=True  跳过 update_log 中已有记录的标的（断点续传）。
        resume=False 强制重新拉取所有标的（全量重建）。
        """
        today = datetime.date.today().isoformat()

        logger.info("=== 开始初始化数据库 ===")
        self._update_trade_calendar()

        if symbols is None:
            stock_list = self._update_stock_info(include_etf=include_etf)
            symbols = stock_list["symbol"].tolist()
        else:
            records = []
            for s in symbols:
                s = str(s).strip().zfill(6)
                asset_type = "index" if s.startswith(("000", "399", "880")) else (
                    "etf" if s.startswith(("15", "16", "51", "52", "58")) else "stock"
                )
                records.append({
                    "symbol": s,
                    "name": s,
                    "market": self._infer_market(s, asset_type),
                    "asset_type": asset_type,
                    "updated_at": today,
                })
            self._upsert_stock_info(records)

        self._ensure_update_log_rows(symbols)
        self.db.commit()

        # 断点续传：查出已完成的标的，跳过
        if resume:
            done = set(
                row[0] for row in self.db.query(
                    "SELECT symbol FROM update_log WHERE last_price_date IS NOT NULL"
                )
            )
            pending = [s for s in symbols if s not in done]
            if done:
                logger.info(
                    "断点续传：已完成 {} 个，剩余 {} 个",
                    len(done), len(pending)
                )
        else:
            pending = symbols

        logger.info("开始拉取 {} 个标的的历史数据...", len(pending))
        for symbol in tqdm(pending, desc="初始化历史数据"):
            self._init_symbol(symbol, start_date, today)

        logger.info("=== 初始化完成 ===")

    def _init_symbol(self, symbol: str, start_date: str, end_date: str):
        """初始化单个标的的历史数据"""
        try:
            asset_type = self._get_asset_type(symbol)

            if asset_type == "index":
                self._init_index(symbol, start_date, end_date)
                return

            # 1. 拉取不复权日线
            if asset_type == "etf":
                df_price = self.fetcher.fetch_etf_daily(symbol, start_date, end_date)
            else:
                df_price = self.fetcher.fetch_daily(symbol, start_date, end_date)

            if df_price.empty:
                logger.warning("  {} 无日线数据，跳过", symbol)
                self._update_log_status(symbol, "failed", last_error="无日线数据")
                self.db.commit()
                return

            self._save_daily(df_price)
            # 同步更新股本（用最新一条 outstanding_share）
            self._save_shares(df_price)

            # 2. 拉取后复权因子（全量，仅个股）
            # ETF 分红极小，直接存 1.0 因子
            if asset_type == "etf":
                dates = df_price["trade_date"].tolist()
                factor_records = [(symbol, d, 1.0) for d in dates]
                self.db.execute_many(
                    "INSERT OR REPLACE INTO adj_factor(symbol, trade_date, hfq_factor) VALUES(?, ?, ?)",
                    factor_records
                )
            else:
                df_factor = self.fetcher.fetch_hfq_factor(symbol)
                if not df_factor.empty:
                    self._save_factors(df_factor)

            # 3. 更新 update_log
            last_date = df_price["trade_date"].max()
            self.db.execute(
                "INSERT OR REPLACE INTO update_log(symbol, last_price_date, last_factor_date, load_status, last_error, updated_at) "
                "VALUES(?, ?, ?, 'success', NULL, datetime('now','localtime'))",
                (symbol, last_date, end_date)
            )
            self.db.commit()

        except Exception as e:
            logger.error("  初始化 {} 失败: {}", symbol, e)
            self._update_log_status(symbol, "failed", last_error=str(e))
            self.db.commit()

    # ------------------------------------------------------------------
    # 增量更新
    # ------------------------------------------------------------------

    def update_incremental(self):
        """每日盘后增量更新"""
        today = datetime.date.today().isoformat()

        # 确认今日是否为交易日
        row = self.db.query_one(
            "SELECT is_open FROM trade_calendar WHERE trade_date = ?", (today,)
        )
        if row and row["is_open"] == 0:
            logger.info("今日 {} 非交易日，跳过更新", today)
            return
        if row is None:
            logger.warning("交易日历无今日记录，继续更新（可能日历未更新）")

        # 找出需要更新的标的
        rows = self.db.query(
            "SELECT symbol, last_price_date FROM update_log "
            "WHERE last_price_date < ? OR last_price_date IS NULL",
            (today,)
        )
        if not rows:
            logger.info("所有标的数据已是最新")
            return

        logger.info("需要更新 {} 个标的", len(rows))
        for row in tqdm(rows, desc="增量更新"):
            symbol = row["symbol"]
            last_date = row["last_price_date"] or DEFAULT_START_DATE
            self._update_symbol(symbol, last_date, today)

        logger.info("增量更新完成")

    def _update_symbol(self, symbol: str, last_date: str, today: str):
        """增量更新单个标的"""
        try:
            asset_type = self._get_asset_type(symbol)

            # 计算增量起始日（last_date 的次日）
            start = (
                datetime.date.fromisoformat(last_date)
                + datetime.timedelta(days=1)
            ).isoformat()

            if start > today:
                self._update_log_status(symbol, "success", last_error=None)
                self.db.commit()
                return

            # 1. 增量拉取价格（只拉新增部分）
            if asset_type == "index":
                df_price = self.fetcher.fetch_index_daily(symbol, start, today)
            elif asset_type == "etf":
                df_price = self.fetcher.fetch_etf_daily(symbol, start, today)
            else:
                df_price = self.fetcher.fetch_daily(symbol, start, today)

            if not df_price.empty:
                if asset_type == "index":
                    self._save_index_daily(df_price)
                else:
                    self._save_daily(df_price)

                if asset_type == "etf":
                    dates = df_price["trade_date"].tolist()
                    self.db.execute_many(
                        "INSERT OR REPLACE INTO adj_factor(symbol, trade_date, hfq_factor) VALUES(?, ?, ?)",
                        [(symbol, d, 1.0) for d in dates]
                    )

            # 2. 个股：全量覆盖复权因子（除权可能修改历史因子）
            if asset_type == "stock":
                df_factor = self.fetcher.fetch_hfq_factor(symbol)
                if not df_factor.empty:
                    self._save_factors(df_factor)

            # 3. 更新断点记录
            new_last = max(
                df_price["trade_date"].max() if not df_price.empty else last_date,
                last_date
            )
            self.db.execute(
                "INSERT OR REPLACE INTO update_log(symbol, last_price_date, last_factor_date, load_status, last_error, updated_at) "
                "VALUES(?, ?, ?, 'success', NULL, datetime('now','localtime'))",
                (symbol, new_last, today)
            )
            self.db.commit()

        except Exception as e:
            logger.error("增量更新 {} 失败: {}", symbol, e)
            self._update_log_status(symbol, "failed", last_error=str(e))
            self.db.commit()

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _update_trade_calendar(self):
        logger.info("更新交易日历...")
        df = self.fetcher.fetch_trade_calendar()
        if df.empty:
            logger.warning("交易日历拉取失败")
            return
        params = [(r["trade_date"], r["is_open"]) for _, r in df.iterrows()]
        self.db.execute_many(
            "INSERT OR REPLACE INTO trade_calendar(trade_date, is_open) VALUES(?, ?)",
            params
        )
        logger.info("  交易日历写入 {} 条", len(params))

    def _update_stock_info(self, include_etf: bool = True) -> pd.DataFrame:
        logger.info("更新股票列表...")
        today = datetime.date.today().isoformat()
        df_stock = self.fetcher.fetch_stock_list()

        all_df = [df_stock]
        if include_etf:
            try:
                df_etf = self.fetcher.fetch_etf_list()
                all_df.append(df_etf)
            except Exception as e:
                logger.warning("ETF 列表获取失败: {}", e)

        df_all = pd.concat(all_df, ignore_index=True)
        records = []
        for _, r in df_all.iterrows():
            records.append({
                "symbol": r["symbol"],
                "name": r["name"],
                "market": r.get("market", ""),
                "asset_type": r["asset_type"],
                "sector": r.get("sector", ""),
                "industry": r.get("industry", ""),
                "list_date": r.get("list_date", ""),
                "is_active": 1,
                "updated_at": today,
            })
        self._upsert_stock_info(records)
        self._ensure_update_log_rows(df_all["symbol"].astype(str).tolist())
        logger.info("  股票/ETF 列表写入 {} 条", len(records))
        return df_all

    def _save_daily(self, df: pd.DataFrame):
        """将日线数据批量写入 stock_daily（INSERT OR REPLACE，支持更新 outstanding_share）"""
        df = self._sanitize_daily_rows(df, "stock_daily")
        if df.empty:
            logger.warning("stock_daily 无有效行可写入")
            return

        def _val(row, col):
            v = row.get(col, None)
            return None if (v is None or (isinstance(v, float) and pd.isna(v))) else float(v)

        params = []
        for _, row in df.iterrows():
            params.append((
                row["symbol"], row["trade_date"],
                _val(row, "open"), _val(row, "high"),
                _val(row, "low"), _val(row, "close"),
                _val(row, "volume"), _val(row, "amount"),
                _val(row, "turnover_rate"),
                _val(row, "outstanding_share"),  # 时间序列股本，防未来函数
                _val(row, "pe_ttm"),
                _val(row, "pb"),
            ))

        self.db.execute_many(
            "INSERT OR REPLACE INTO stock_daily"
            "(symbol, trade_date, open, high, low, close, volume, amount, "
            "turnover_rate, outstanding_share, pe_ttm, pb) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params
        )

    def _save_factors(self, df: pd.DataFrame):
        """将复权因子批量写入 adj_factor（INSERT OR REPLACE，每次全量覆盖）"""
        params = [
            (row["symbol"], row["trade_date"], float(row["hfq_factor"]))
            for _, row in df.iterrows()
            if pd.notna(row["hfq_factor"]) and float(row["hfq_factor"]) > 0
        ]
        if params:
            self.db.execute_many(
                "INSERT OR REPLACE INTO adj_factor(symbol, trade_date, hfq_factor) VALUES(?, ?, ?)",
                params
            )

    def _save_shares(self, df: pd.DataFrame):
        """
        从日线 DataFrame 中提取最新一条 outstanding_share，存入 stock_shares 表。
        outstanding_share 来自 stock_zh_a_daily，为总股本（股）。
        """
        if "outstanding_share" not in df.columns:
            return
        # 取最新记录
        latest = df[df["outstanding_share"].notna()].iloc[-1] if df["outstanding_share"].notna().any() else None
        if latest is None:
            return
        symbol = latest["symbol"]
        outstanding = float(latest["outstanding_share"])
        if outstanding <= 0:
            return
        self.db.execute(
            "INSERT OR REPLACE INTO stock_shares(symbol, outstanding_share, circ_share, updated_at) "
            "VALUES(?, ?, ?, datetime('now','localtime'))",
            (symbol, outstanding, outstanding)
        )
        self.db.commit()

    def update_shares(self, symbols: Optional[list[str]] = None):
        """
        批量更新 stock_shares 表（总股本/流通股本）。
        symbols=None 时更新全部在 update_log 中有记录的个股。
        此操作每只股票需 1 次 API 调用，全市场约需 2-3 小时。
        """
        if symbols is None:
            rows = self.db.query(
                """SELECT ul.symbol FROM update_log ul
                   JOIN stock_info si ON ul.symbol = si.symbol
                   WHERE si.asset_type = 'stock'
                   AND NOT EXISTS (
                       SELECT 1 FROM stock_shares ss WHERE ss.symbol = ul.symbol
                   )"""
            )
            symbols = [r[0] for r in rows]

        if not symbols:
            logger.info("所有个股股本数据已是最新")
            return

        logger.info("开始拉取 {} 只股票的股本数据...", len(symbols))
        success = 0
        for symbol in tqdm(symbols, desc="更新股本"):
            result = self.fetcher.fetch_shares(symbol)
            if result and result.get("outstanding_share", 0) > 0:
                self.db.execute(
                    "INSERT OR REPLACE INTO stock_shares(symbol, outstanding_share, circ_share, updated_at) "
                    "VALUES(?, ?, ?, datetime('now','localtime'))",
                    (symbol, result["outstanding_share"], result["circ_share"])
                )
                success += 1
            if success % 100 == 0 and success > 0:
                self.db.commit()
        self.db.commit()
        logger.info("股本数据更新完成: {}/{}", success, len(symbols))

    def update_valuation(self, symbols: list[str]):
        """
        批量更新总市值和PB数据（百度估值接口）。

        将 total_mv / pb / pe_ttm 写入 stock_valuation 表。
        每只股票需 2 次 API 调用（总市值 + 市净率），全市场约需 4-6 小时。

        数据特点：
          - 已处理所有股本变动（送转/可转债转股/增发/回购），无未来函数风险
          - '全部'周期覆盖上市以来历史（约9天频率），'近三年'为日频
          - total_mv 单位为亿元
        """
        logger.info("开始拉取 {} 只股票的估值数据(总市值+PB)...", len(symbols))
        success = 0
        for symbol in tqdm(symbols, desc="更新估值"):
            try:
                df = self.fetcher.fetch_valuation(symbol)
                if df is None or df.empty:
                    continue

                # 批量写入 stock_valuation 表（INSERT OR REPLACE）
                params = []
                for _, row in df.iterrows():
                    mv = row.get('total_mv')
                    pb = row.get('pb')
                    pe = row.get('pe_ttm')
                    params.append((
                        symbol,
                        row['trade_date'],
                        float(mv) if mv and pd.notna(mv) else None,
                        float(pb) if pb and pd.notna(pb) else None,
                        float(pe) if pe and pd.notna(pe) else None,
                    ))

                self.db.execute_many(
                    "INSERT OR REPLACE INTO stock_valuation"
                    "(symbol, trade_date, total_mv, pb, pe_ttm) VALUES(?, ?, ?, ?, ?)",
                    params,
                )
                success += 1
            except Exception as e:
                logger.debug("估值拉取失败 {}: {}", symbol, e)

            if success % 50 == 0 and success > 0:
                self.db.commit()
        self.db.commit()
        logger.info("估值数据更新完成: {}/{}", success, len(symbols))

    def update_financial_reports(
        self,
        symbols: Optional[list[str]] = None,
        skip_existing: bool = True,
    ):
        """
        批量拉取三大报表，写入 financial_report 表。

        数据来源：新浪财经 stock_financial_report_sina 接口
        每只股票 3 次 API 调用（利润表/资产负债表/现金流量表），50RPM 限速下约 3.6 秒/只。
        全市场 ~5200 只约需 5 小时。

        参数：
          symbols: 指定股票代码列表。None 表示全市场（stock_info.asset_type='stock' 且 is_active=1）
          skip_existing: True 时跳过 financial_report 表里已有数据的 symbol（首次初始化用）
                        False 时强制重拉并 UPSERT（增量更新用）

        写入策略：INSERT OR REPLACE，按 (symbol, report_date, data_source, is_preliminary) UPSERT
        新数据进来后旧数据会被覆盖，保证最新版本。
        """
        # 确定待拉取的 symbol 列表
        if symbols is None:
            rows = self.db.query(
                "SELECT symbol FROM stock_info WHERE asset_type='stock' AND is_active=1 "
                "ORDER BY symbol"
            )
            symbols = [r["symbol"] for r in rows]
            logger.info("全市场财报拉取: {} 只股票 | skip_existing={}", len(symbols), skip_existing)
        else:
            symbols = [str(s).strip().zfill(6) for s in symbols]
            logger.info("指定股票财报拉取: {} 只 | skip_existing={}", len(symbols), skip_existing)

        if not symbols:
            logger.warning("待拉取列表为空")
            return

        # 跳过已有数据的 symbol（首次初始化场景）
        if skip_existing:
            existing_rows = self.db.query(
                "SELECT DISTINCT symbol FROM financial_report"
            )
            existing = {r["symbol"] for r in existing_rows}
            pending = [s for s in symbols if s not in existing]
            logger.info("跳过已有数据: {} 只 | 待拉取: {} 只", len(symbols) - len(pending), len(pending))
            symbols = pending

        if not symbols:
            logger.info("无新增股票需拉取")
            return

        success = 0
        failed = 0
        for symbol in tqdm(symbols, desc="财报拉取"):
            try:
                df = self.fetcher.fetch_financial_report(symbol)
                if df is None or df.empty:
                    failed += 1
                    continue

                # UPSERT 写入
                params = []
                for _, row in df.iterrows():
                    def _val(v):
                        if v is None or (isinstance(v, float) and pd.isna(v)):
                            return None
                        try:
                            return float(v)
                        except (TypeError, ValueError):
                            return None

                    params.append((
                        symbol,
                        row["report_date"],
                        row["report_type"],
                        row["announce_date"],
                        _val(row.get("revenue")),
                        _val(row.get("operating_income")),
                        _val(row.get("operating_cost")),
                        _val(row.get("sell_expense")),
                        _val(row.get("admin_expense")),
                        _val(row.get("rd_expense")),
                        _val(row.get("finance_expense")),
                        _val(row.get("net_profit")),
                        _val(row.get("net_profit_attr")),
                        _val(row.get("eps")),
                        _val(row.get("total_assets")),
                        _val(row.get("total_liabilities")),
                        _val(row.get("equity_attr")),
                        _val(row.get("equity")),
                        _val(row.get("cash")),
                        _val(row.get("accounts_receivable")),
                        _val(row.get("inventory")),
                        _val(row.get("goodwill")),
                        _val(row.get("contract_liability")),
                        _val(row.get("interest_bearing_debt")),
                        _val(row.get("bvps")),
                        _val(row.get("operating_cf")),
                        _val(row.get("investing_cf")),
                        _val(row.get("financing_cf")),
                        _val(row.get("capex")),
                        _val(row.get("free_cf")),
                        int(row.get("is_preliminary") or 0),
                        str(row.get("data_source") or "sina"),
                        str(row.get("is_audited")) if pd.notna(row.get("is_audited")) else None,
                    ))

                self.db.execute_many(
                    "INSERT OR REPLACE INTO financial_report ("
                    "symbol, report_date, report_type, announce_date, "
                    "revenue, operating_income, operating_cost, sell_expense, admin_expense, "
                    "rd_expense, finance_expense, net_profit, net_profit_attr, eps, "
                    "total_assets, total_liabilities, equity_attr, equity, "
                    "cash, accounts_receivable, inventory, goodwill, "
                    "contract_liability, interest_bearing_debt, bvps, "
                    "operating_cf, investing_cf, financing_cf, capex, free_cf, "
                    "is_preliminary, data_source, is_audited"
                    ") VALUES (" + ",".join(["?"] * 33) + ")",
                    params,
                )
                success += 1
                if success % 50 == 0 and success > 0:
                    logger.info("已写入 {} 只", success)
            except Exception as e:
                failed += 1
                logger.debug("财报拉取失败 {}: {}", symbol, e)

        self.db.commit()
        logger.info("财报更新完成: 成功 {} | 失败 {} | 总计 {}", success, failed, len(symbols))

    def _get_asset_type(self, symbol: str) -> str:
        """从 stock_info 表查询标的类型，默认返回 'stock'"""
        row = self.db.query_one(
            "SELECT asset_type FROM stock_info WHERE symbol = ? "
            "ORDER BY CASE asset_type WHEN 'stock' THEN 1 WHEN 'etf' THEN 2 WHEN 'index' THEN 3 ELSE 9 END "
            "LIMIT 1",
            (symbol,)
        )
        return row["asset_type"] if row else "stock"

    def init_symbols(
        self,
        symbols: list[str],
        asset_types: dict[str, str],
        start_date: str = DEFAULT_START_DATE,
        resume: bool = True,
    ):
        """
        对指定标的列表初始化数据（不需要拉取全市场列表）。
        asset_types: {symbol: 'stock'/'etf'/'index'}
        resume=True 跳过已完成的标的。
        """
        today = datetime.date.today().isoformat()

        records = []
        for symbol, atype in asset_types.items():
            symbol = str(symbol).strip().zfill(6)
            records.append({
                "symbol": symbol,
                "name": symbol,
                "market": self._infer_market(symbol, atype),
                "asset_type": atype,
                "updated_at": today,
            })
        self._upsert_stock_info(records)
        self._ensure_update_log_rows(list(asset_types.keys()))
        self.db.commit()

        if resume:
            done = set(
                row[0] for row in self.db.query(
                    "SELECT symbol FROM update_log WHERE last_price_date IS NOT NULL"
                )
            )
            pending = [s for s in symbols if s not in done]
            if done:
                logger.info("断点续传：已完成 {}，剩余 {}", len(done), len(pending))
        else:
            pending = symbols

        logger.info("开始拉取 {} 个标的...", len(pending))
        for symbol in tqdm(pending, desc="初始化数据"):
            atype = asset_types.get(symbol, "stock")
            if atype == "index":
                self._init_index(symbol, start_date, today)
            else:
                self._init_symbol(symbol, start_date, today)

        logger.info("完成")

    def _init_index(self, symbol: str, start_date: str, end_date: str):
        """初始化指数日线数据（写入 index_daily 表）"""
        try:
            df = self.fetcher.fetch_index_daily(symbol, start_date, end_date)
            if df.empty:
                logger.warning("  指数 {} 无数据，跳过", symbol)
                return
            self._save_index_daily(df)

            last_date = df["trade_date"].max()
            self.db.execute(
                "INSERT OR REPLACE INTO update_log(symbol, last_price_date, last_factor_date, load_status, last_error, updated_at) "
                "VALUES(?, ?, ?, 'success', NULL, datetime('now','localtime'))",
                (symbol, last_date, end_date)
            )
            self.db.commit()
            logger.debug("  指数 {} 写入 {} 条", symbol, len(df))
        except Exception as e:
            logger.error("  初始化指数 {} 失败: {}", symbol, e)
            self._update_log_status(symbol, "failed", last_error=str(e))
            self.db.commit()

    def _save_index_daily(self, df: pd.DataFrame):
        """将指数日线写入 index_daily 表（INSERT OR REPLACE）"""
        df = self._sanitize_daily_rows(df, "index_daily")
        if df.empty:
            logger.warning("index_daily 无有效行可写入")
            return

        def _val(row, col):
            v = row.get(col, None)
            return None if (v is None or (isinstance(v, float) and pd.isna(v))) else float(v)

        params = []
        for _, row in df.iterrows():
            params.append((
                row["symbol"], row["trade_date"],
                _val(row, "open"), _val(row, "high"),
                _val(row, "low"), _val(row, "close"),
                _val(row, "volume"), _val(row, "amount"),
            ))
        self.db.execute_many(
            "INSERT OR REPLACE INTO index_daily"
            "(symbol, trade_date, open, high, low, close, volume, amount) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            params
        )

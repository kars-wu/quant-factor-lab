"""
AkShare 数据源适配器（基于实测接口）。

接口选型（经实测验证）：
  个股日线：ak.stock_zh_a_daily(symbol='sz000001', adjust='')
            返回列：date, open, high, low, close, volume, amount
            支持 adjust='' / 'hfq' / 'qfq'，全历史数据

  ETF日线：ak.fund_etf_hist_sina(symbol='sh510300')
           返回列：date, open, high, low, close, volume, amount
           无复权参数，需另行处理（ETF 分红即复权）

  指数日线：ak.stock_zh_index_daily(symbol='sh000300')
            返回列：date, open, high, low, close, volume

  交易日历：ak.tool_trade_date_hist_sina()

复权因子推算：
  个股：同时拉取 adjust='' 和 adjust='hfq'，逐日 hfq / raw 得因子
        注意：因子存在浮点微小波动（非严格单调），属正常现象
  ETF：直接使用不复权价格存储（ETF 现金分红分红因子极小，对日线影响可忽略）
"""

from __future__ import annotations

import pandas as pd
import akshare as ak
from loguru import logger

from .base_fetcher import BaseFetcher, RateLimiter


# stock_zh_a_daily / stock_zh_a_hist 列名映射
_STOCK_DAILY_COLS = {
    "date": "trade_date",
    "日期": "trade_date",
    "open": "open", "开盘": "open",
    "high": "high", "最高": "high",
    "low": "low", "最低": "low",
    "close": "close", "收盘": "close",
    "volume": "volume", "成交量": "volume",
    "amount": "amount", "成交额": "amount",
    "turnover": "turnover_rate", "换手率": "turnover_rate",
}

# ETF 代码前缀规则
def _etf_prefix(symbol: str) -> str:
    """
    ETF 市场前缀推断。

    经验规则：
      - 深市 ETF/LOF 常见为 15xxxx / 16xxxx
      - 沪市 ETF 常见为 5xxxxx（如 510/512/515/518/520/530/560/563/588...）
    """
    if symbol.startswith(("15", "16")):
        return f"sz{symbol}"
    return f"sh{symbol}"

# 指数代码前缀规则
def _index_prefix(symbol: str) -> str:
    if symbol.startswith(("000", "880", "399")):
        return f"sh{symbol}" if symbol.startswith("000") else f"sz{symbol}"
    return f"sh{symbol}"


class AkShareFetcher(BaseFetcher):

    def __init__(self, rate_limiter: RateLimiter | None = None):
        super().__init__(rate_limiter)

    # ------------------------------------------------------------------
    # 股票/ETF 列表
    # ------------------------------------------------------------------

    def fetch_stock_list(self) -> pd.DataFrame:
        """
        获取全市场股票列表（沪深两市，含主板+创业板）。
        返回列：[symbol, name, market, asset_type, list_date, sector]
        sector: '主板' / '创业板' / '科创板'
        """
        self._throttle()
        records = []

        # 深市（主板 + 创业板，接口无需参数）
        try:
            df_sz = self._retry(ak.stock_info_sz_name_code)
            for _, row in df_sz.iterrows():
                symbol = str(row.get("A股代码", "")).strip().zfill(6)
                if not symbol or symbol == "000000":
                    continue
                records.append({
                    "symbol": symbol,
                    "name": str(row.get("A股简称", "")).strip(),
                    "market": "SZ",
                    "asset_type": "stock",
                    "list_date": str(row.get("A股上市日期", "")).replace("-", ""),
                    "sector": str(row.get("板块", "")),
                })
            logger.info("深市股票: {} 只", len(df_sz))
        except Exception as e:
            logger.warning("深市股票列表获取失败: {}", e)

        self._throttle()
        # 沪市主板
        try:
            df_sh = self._retry(ak.stock_info_sh_name_code, symbol="主板A股")
            for _, row in df_sh.iterrows():
                symbol = str(row.get("证券代码", "")).strip().zfill(6)
                if not symbol:
                    continue
                records.append({
                    "symbol": symbol,
                    "name": str(row.get("证券简称", "")).strip(),
                    "market": "SH",
                    "asset_type": "stock",
                    "list_date": str(row.get("上市日期", "")).replace("-", ""),
                    "sector": "主板",
                })
            logger.info("沪市主板: {} 只", len(df_sh))
        except Exception as e:
            logger.warning("沪市主板列表获取失败: {}", e)

        self._throttle()
        # 科创板
        try:
            df_kcb = self._retry(ak.stock_info_sh_name_code, symbol="科创板")
            for _, row in df_kcb.iterrows():
                symbol = str(row.get("证券代码", "")).strip().zfill(6)
                if not symbol:
                    continue
                records.append({
                    "symbol": symbol,
                    "name": str(row.get("证券简称", "")).strip(),
                    "market": "SH",
                    "asset_type": "stock",
                    "list_date": str(row.get("上市日期", "")).replace("-", ""),
                    "sector": "科创板",
                })
            logger.info("科创板: {} 只", len(df_kcb))
        except Exception as e:
            logger.warning("科创板列表获取失败: {}", e)

        df = pd.DataFrame(records).drop_duplicates(subset=["symbol"])
        df = df[df["symbol"].str.match(r"^\d{6}$")]
        return df.reset_index(drop=True)

    def fetch_etf_list(self) -> pd.DataFrame:
        """获取 ETF 列表（新浪财经）"""
        self._throttle()
        try:
            df = self._retry(ak.fund_etf_category_sina, symbol="ETF基金")
        except Exception as e:
            logger.warning("ETF 列表获取失败: {}", e)
            return pd.DataFrame()

        records = []
        for _, row in df.iterrows():
            code = str(row.get("代码", ""))
            symbol = code.replace("sh", "").replace("sz", "").zfill(6)
            market = "SH" if code.startswith("sh") else "SZ"
            records.append({
                "symbol": symbol,
                "name": row.get("名称", ""),
                "market": market,
                "asset_type": "etf",
                "list_date": "",
            })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # 个股日线（不复权）
    # ------------------------------------------------------------------

    def fetch_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        拉取个股不复权日线数据。
        接口：ak.stock_zh_a_daily（支持全历史，无需分批）
        symbol 格式：'000001'（自动加 sz/sh 前缀）
        返回列：[symbol, trade_date, open, high, low, close, volume, amount]
        """
        self._throttle()
        prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
        full_symbol = f"{prefix}{symbol}"

        try:
            df = self._retry(
                ak.stock_zh_a_daily,
                symbol=full_symbol,
                adjust="",
            )
        except Exception as e:
            logger.warning("获取 {} 日线失败: {}", symbol, e)
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.rename(columns=_STOCK_DAILY_COLS)
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")

        # 按日期范围过滤
        df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
        df["symbol"] = symbol

        # turnover_rate: stock_zh_a_daily 返回的 "turnover" 映射后为小数形式（0.07 = 7%）
        # outstanding_share: 总股本（股）
        keep = ["symbol", "trade_date", "open", "high", "low", "close",
                "volume", "amount", "turnover_rate", "outstanding_share"]
        existing = [c for c in keep if c in df.columns]
        return df[existing].reset_index(drop=True)

    def fetch_shares(self, symbol: str) -> dict:
        """
        获取单只股票当前总股本。
        返回: {'symbol': ..., 'outstanding_share': ..., 'circ_share': ...}
        使用 stock_zh_a_daily 最新一条记录中的 outstanding_share。
        注意：stock_zh_a_daily 只提供总股本，暂用总股本近似流通股。
        """
        self._throttle()
        prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
        full_symbol = f"{prefix}{symbol}"
        try:
            df = self._retry(ak.stock_zh_a_daily, symbol=full_symbol, adjust="")
            if df is None or df.empty:
                return {}
            latest = df.iloc[-1]
            outstanding = float(latest.get("outstanding_share", 0) or 0)
            return {
                "symbol": symbol,
                "outstanding_share": outstanding,
                "circ_share": outstanding,  # 暂用总股本近似
            }
        except Exception as e:
            logger.debug("获取 {} 股本失败: {}", symbol, e)
            return {}

    # ------------------------------------------------------------------
    # 后复权因子推算（个股专用）
    # ------------------------------------------------------------------

    def fetch_hfq_latest(self, symbol: str) -> pd.DataFrame:
        """只拉取后复权日线的最新一条收盘价，用于快速探测因子是否变化。"""
        prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
        try:
            self._throttle()
            df = self._retry(ak.stock_zh_a_daily, symbol=f"{prefix}{symbol}", adjust="hfq")
        except Exception as e:
            logger.warning("获取 {} 最新后复权价格失败: {}", symbol, e)
            return pd.DataFrame()
        if df is None or df.empty or "date" not in df.columns or "close" not in df.columns:
            return pd.DataFrame()
        out = df[["date", "close"]].rename(columns={"date": "trade_date", "close": "close_hfq"}).copy()
        out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.strftime("%Y-%m-%d")
        return out.tail(1).reset_index(drop=True)

    def fetch_hfq_factor(self, symbol: str) -> pd.DataFrame:
        """
        通过同时拉取不复权和后复权数据，推算每日后复权因子。
        hfq_factor(t) = close_hfq(t) / close_raw(t)

        注意：因子存在浮点微小波动（量级 ~1e-4），属正常现象，
              在 adjfactor.py 的 apply_qfq 中已做容错处理。

        返回列：[symbol, trade_date, hfq_factor]
        """
        prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
        full_symbol = f"{prefix}{symbol}"

        self._throttle()
        try:
            df_raw = self._retry(ak.stock_zh_a_daily, symbol=full_symbol, adjust="")
        except Exception as e:
            logger.warning("获取 {} 不复权数据失败（复权因子）: {}", symbol, e)
            return pd.DataFrame()

        self._throttle()
        try:
            df_hfq = self._retry(ak.stock_zh_a_daily, symbol=full_symbol, adjust="hfq")
        except Exception as e:
            logger.warning("获取 {} 后复权数据失败（复权因子）: {}", symbol, e)
            return pd.DataFrame()

        if df_raw is None or df_raw.empty or df_hfq is None or df_hfq.empty:
            return pd.DataFrame()

        df_raw = df_raw.rename(columns={"date": "trade_date", "close": "close_raw"})
        df_hfq = df_hfq.rename(columns={"date": "trade_date", "close": "close_hfq"})

        df_raw["trade_date"] = pd.to_datetime(df_raw["trade_date"]).dt.strftime("%Y-%m-%d")
        df_hfq["trade_date"] = pd.to_datetime(df_hfq["trade_date"]).dt.strftime("%Y-%m-%d")

        merged = pd.merge(
            df_raw[["trade_date", "close_raw"]],
            df_hfq[["trade_date", "close_hfq"]],
            on="trade_date",
            how="inner",
        )

        # 过滤无效行
        merged = merged[merged["close_raw"].abs() > 1e-6].copy()
        merged["hfq_factor"] = merged["close_hfq"] / merged["close_raw"]
        merged["symbol"] = symbol

        return merged[["symbol", "trade_date", "hfq_factor"]].reset_index(drop=True)

    # ------------------------------------------------------------------
    # ETF 日线（新浪财经，无复权参数）
    # ------------------------------------------------------------------

    def fetch_etf_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        拉取 ETF 日线数据（新浪财经接口，全历史）。
        ETF 分红直接体现在净值中，此处存储不复权价格（与净值一致），
        adj_factor 默认设为 1.0。
        返回列：[symbol, trade_date, open, high, low, close, volume, amount]
        """
        self._throttle()
        sina_symbol = _etf_prefix(symbol)

        try:
            df = self._retry(ak.fund_etf_hist_sina, symbol=sina_symbol)
        except Exception as e:
            logger.warning("获取 ETF {} 日线失败: {}", symbol, e)
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.rename(columns={"date": "trade_date"})
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
        df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
        df["symbol"] = symbol

        keep = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"]
        existing = [c for c in keep if c in df.columns]
        return df[existing].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 指数日线
    # ------------------------------------------------------------------

    def fetch_index_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        获取指数日线（如 '000300'、'000001'）。
        接口：ak.stock_zh_index_daily，全历史。
        返回列：[symbol, trade_date, open, high, low, close, volume]
        """
        self._throttle()
        sina_symbol = _index_prefix(symbol)

        try:
            df = self._retry(ak.stock_zh_index_daily, symbol=sina_symbol)
        except Exception as e:
            logger.warning("获取指数 {} 失败: {}", symbol, e)
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.rename(columns={"date": "trade_date"})
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
        df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
        df["symbol"] = symbol

        keep = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"]
        existing = [c for c in keep if c in df.columns]
        return df[existing].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 交易日历
    # ------------------------------------------------------------------

    def fetch_trade_calendar(self, start_year: int = 2010) -> pd.DataFrame:
        self._throttle()
        try:
            df = self._retry(ak.tool_trade_date_hist_sina)
        except Exception as e:
            logger.warning("获取交易日历失败: {}", e)
            return pd.DataFrame()

        df.columns = ["trade_date"]
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
        df = df[df["trade_date"] >= f"{start_year}-01-01"].copy()
        df["is_open"] = 1
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 个股估值数据（百度接口，总市值 + PB）
    # ------------------------------------------------------------------

    def fetch_valuation(self, symbol: str) -> pd.DataFrame:
        """
        拉取个股总市值和市净率（百度估值接口）。

        返回列：[trade_date, total_mv, pb]
        total_mv 单位：亿元
        pb 无量纲

        数据来源：ak.stock_zh_valuation_baidu
        该接口已处理所有股本变动（送转/可转债转股/增发/回购），
        不存在 outstanding_share 未来函数问题。

        period='全部' 覆盖上市以来全部历史（约9天频率），
        period='近三年' 覆盖近三年日频数据。
        """
        records = []

        for indicator, col_name in [('总市值', 'total_mv'), ('市净率', 'pb'), ('市盈率(TTM)', 'pe_ttm')]:
            self._throttle()
            try:
                df = self._retry(
                    ak.stock_zh_valuation_baidu,
                    symbol=symbol,
                    indicator=indicator,
                    period='全部',
                )
                if df is None or df.empty:
                    continue
                df = df.rename(columns={'date': 'trade_date', 'value': col_name})
                df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
                df = df.set_index('trade_date')
                records.append(df)
            except Exception as e:
                logger.debug("获取 {} {} 失败: {}", symbol, indicator, e)

        if not records:
            return pd.DataFrame()

        # 合并 total_mv, pb, pe_ttm
        result = records[0]
        for df in records[1:]:
            result = result.join(df, how='outer')

        result = result.reset_index()
        result['symbol'] = symbol
        cols = ['trade_date', 'symbol']
        for c in ['total_mv', 'pb', 'pe_ttm']:
            if c in result.columns:
                cols.append(c)
        return result[cols].sort_values('trade_date')

    # ------------------------------------------------------------------
    # 财务报表（新浪接口，三大报表合一）
    # ------------------------------------------------------------------

    # 利润表字段映射（新浪列名 → schema 字段名）
    _INCOME_COLS = {
        '报告日': 'report_date',
        '营业收入': 'revenue',
        '营业总收入': 'operating_income',
        '营业成本': 'operating_cost',
        '销售费用': 'sell_expense',
        '管理费用': 'admin_expense',
        '研发费用': 'rd_expense',
        '财务费用': 'finance_expense',
        '净利润': 'net_profit',
        '归属于母公司所有者的净利润': 'net_profit_attr',
        '基本每股收益': 'eps',
        '公告日期': 'announce_date',
        '是否审计': 'is_audited',
    }

    # 资产负债表字段映射
    _BALANCE_COLS = {
        '报告日': 'report_date',
        '货币资金': 'cash',
        '应收账款': 'accounts_receivable',
        '存货': 'inventory',
        '商誉': 'goodwill',
        '合同负债': 'contract_liability',
        '资产总计': 'total_assets',
        '负债合计': 'total_liabilities',
        '归属于母公司股东权益合计': 'equity_attr',
        '所有者权益(或股东权益)合计': 'equity',
        '短期借款': '_short_loan',
        '长期借款': '_long_loan',
        '应付债券': '_bond_payable',
        '一年内到期的非流动负债': '_current_portion_noncurrent',
        '实收资本(或股本)': '_paid_in_capital',
        '公告日期': 'announce_date',
    }

    # 现金流量表字段映射
    _CASH_COLS = {
        '报告日': 'report_date',
        '经营活动产生的现金流量净额': 'operating_cf',
        '投资活动产生的现金流量净额': 'investing_cf',
        '筹资活动产生的现金流量净额': 'financing_cf',
        '购建固定资产、无形资产和其他长期资产所支付的现金': 'capex',
        '公告日期': 'announce_date',
    }

    @staticmethod
    def _infer_report_type(report_date: str) -> str:
        """根据报告期末日推断报告类型。"""
        mmdd = report_date[5:10] if len(report_date) == 10 else report_date[4:8]
        return {
            '03-31': 'q1', '0331': 'q1',
            '06-30': 'interim', '0630': 'interim',
            '09-30': 'q3', '0930': 'q3',
            '12-31': 'annual', '1231': 'annual',
        }.get(mmdd, 'other')

    @staticmethod
    def _to_iso_date(s) -> str:
        """'20260331' 或 '2026-03-31' → '2026-03-31'；空值返回 ''"""
        if s is None or (isinstance(s, float) and pd.isna(s)):
            return ''
        s = str(s).strip()
        if not s or s.lower() == 'nan':
            return ''
        if len(s) == 8 and s.isdigit():
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        return s

    def fetch_financial_report(self, symbol: str) -> pd.DataFrame:
        """
        拉取个股三大报表（利润表 + 资产负债表 + 现金流量表），合并为长表。

        数据来源：ak.stock_financial_report_sina（新浪财经，全历史季报）
        接口需 3 次 API 调用（每张报表一次）

        返回列（与 financial_report 表对齐）：
          symbol, report_date, report_type, announce_date,
          revenue, operating_income, operating_cost, sell_expense, admin_expense,
          rd_expense, finance_expense, net_profit, net_profit_attr, eps,
          total_assets, total_liabilities, equity_attr, equity,
          cash, accounts_receivable, inventory, goodwill,
          contract_liability, interest_bearing_debt, bvps,
          operating_cf, investing_cf, financing_cf, capex, free_cf,
          is_preliminary, data_source, is_audited
        """
        prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
        full_symbol = f"{prefix}{symbol}"

        # 三张报表分别拉取，按 report_date 合并
        merged: pd.DataFrame | None = None
        for statement, col_map in [
            ('利润表', self._INCOME_COLS),
            ('资产负债表', self._BALANCE_COLS),
            ('现金流量表', self._CASH_COLS),
        ]:
            self._throttle()
            try:
                df = self._retry(
                    ak.stock_financial_report_sina,
                    stock=full_symbol,
                    symbol=statement,
                )
            except Exception as e:
                logger.debug("拉取 {} {} 失败: {}", symbol, statement, e)
                continue

            if df is None or df.empty:
                continue

            # 重命名 + 标准化日期
            df = df.rename(columns=col_map)
            df = df[[c for c in df.columns if c in col_map.values()]]
            df['report_date'] = df['report_date'].apply(self._to_iso_date)
            df['announce_date'] = df['announce_date'].apply(self._to_iso_date)
            df = df[df['report_date'] != '']  # 去掉无报告日的行

            if merged is None:
                merged = df
            else:
                # 同一 report_date 合并；announce_date 优先取非空值
                ann = df[['report_date', 'announce_date']].drop_duplicates('report_date')
                ann = ann.rename(columns={'announce_date': 'announce_date_new'})
                merged = merged.merge(ann, on='report_date', how='outer')
                merged['announce_date'] = merged['announce_date'].where(
                    merged['announce_date'].eq('') | merged['announce_date'].isna(),
                    merged['announce_date'].fillna(merged['announce_date_new'])
                )
                merged = merged.drop(columns=['announce_date_new'], errors='ignore')
                # 列合并：取非空
                for c in df.columns:
                    if c in ('report_date', 'announce_date'):
                        continue
                    if c in merged.columns:
                        merged[c] = merged[c].fillna(df.set_index('report_date')[c]
                                                       .reindex(merged['report_date']).values)
                    else:
                        merged[c] = df.set_index('report_date').reindex(merged['report_date'])[c].values

        if merged is None or merged.empty:
            return pd.DataFrame()

        df = merged

        # 推断报告类型
        df['report_type'] = df['report_date'].apply(self._infer_report_type)

        # 衍生字段：有息负债 = 短贷 + 长贷 + 应付债券 + 一年内到期
        def _safe_sum(row, cols):
            total = 0.0
            for c in cols:
                v = row.get(c)
                if v is not None and pd.notna(v):
                    try:
                        total += float(v)
                    except (TypeError, ValueError):
                        pass
            return total if total > 0 else None

        debt_cols = ['_short_loan', '_long_loan', '_bond_payable', '_current_portion_noncurrent']
        df['interest_bearing_debt'] = df.apply(lambda r: _safe_sum(r, debt_cols), axis=1)

        # 衍生字段：自由现金流 = 经营现金流 - 资本开支
        def _calc_free_cf(row):
            ocf = row.get('operating_cf')
            capex = row.get('capex')
            if pd.notna(ocf) and pd.notna(capex):
                try:
                    return float(ocf) - float(capex)
                except (TypeError, ValueError):
                    return None
            return None
        df['free_cf'] = df.apply(_calc_free_cf, axis=1)

        # 衍生字段：每股净资产 = 归母权益 / 实收资本（A股面值1元，实收资本=股本数）
        def _calc_bvps(row):
            eq = row.get('equity_attr')
            cap = row.get('_paid_in_capital')
            if pd.notna(eq) and pd.notna(cap) and float(cap) > 0:
                try:
                    return float(eq) / float(cap)
                except (TypeError, ValueError, ZeroDivisionError):
                    return None
            return None
        df['bvps'] = df.apply(_calc_bvps, axis=1)

        # 清理临时列
        tmp_cols = [c for c in df.columns if c.startswith('_')]
        df = df.drop(columns=tmp_cols)

        # 加 symbol 和元信息
        df['symbol'] = symbol
        df['is_preliminary'] = 0
        df['data_source'] = 'sina'

        # 字段对齐 schema
        schema_cols = [
            'symbol', 'report_date', 'report_type', 'announce_date',
            'revenue', 'operating_income', 'operating_cost', 'sell_expense',
            'admin_expense', 'rd_expense', 'finance_expense',
            'net_profit', 'net_profit_attr', 'eps',
            'total_assets', 'total_liabilities', 'equity_attr', 'equity',
            'cash', 'accounts_receivable', 'inventory', 'goodwill',
            'contract_liability', 'interest_bearing_debt', 'bvps',
            'operating_cf', 'investing_cf', 'financing_cf', 'capex', 'free_cf',
            'is_preliminary', 'data_source', 'is_audited',
        ]
        for c in schema_cols:
            if c not in df.columns:
                df[c] = None

        return df[schema_cols].sort_values('report_date').reset_index(drop=True)

"""
因子分析与验证工具包。

━━━ 完整流程 ━━━

  1. 因子构造：原始数据 → 每只股票每天一个数值（因子值）
  2. IC 分析：因子值 vs 未来收益率的相关性
  3. 因子中性化：剥离市值/行业影响
  4. 分层回测：按因子值分组，看单调性
  5. 衰减分析：IC 随持有天数的变化
  6. 组合验证：多因子合成 → 实际回测

━━━ 核心概念 ━━━

  IC (Information Coefficient) = 每日所有股票的(因子值, 未来收益)的 Spearman 相关
    IC > 0.03 且 ICIR > 0.5 → 有效因子
    IC < 0.02 或 ICIR < 0.3 → 无效因子

  因子中性化：factor_new = factor - α - β × log(market_cap)
    剥离市值效应后，IC 仍然有效才是真因子

用法：
    from strategies.factor_analysis import FactorAnalyzer

    analyzer = FactorAnalyzer(db)
    analyzer.load_data(['600519', '000858', ...], '2020-01-01', '2026-05-14')

    # 构造因子
    analyzer.add_factor('momentum_5d', lambda df: df['close'].pct_change(5))
    analyzer.add_factor('rsi_14', lambda df: rsi(df['close'], 14))

    # IC 分析
    ic = analyzer.calc_ic('momentum_5d', forward_days=5)
    analyzer.print_ic_summary(ic)

    # 分层回测
    layers = analyzer.layer_backtest('momentum_5d', n_layers=5)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from data.adjfactor import apply_hfq, apply_qfq, get_anchor_factor

pd.set_option('future.no_silent_downcasting', True)


@dataclass
class ICResult:
    """IC 分析结果。"""
    factor_name: str
    forward_days: int
    ic_series: pd.Series        # 每日 IC 值
    ic_mean: float
    ic_std: float
    icir: float                 # IC 信息比率 = mean / std
    ic_positive_ratio: float    # IC > 0 的比例
    t_stat: float               # t 检验统计量

    @property
    def is_effective(self) -> bool:
        """是否有效：IC > 0.03 且 ICIR > 0.5"""
        return abs(self.ic_mean) > 0.03 and abs(self.icir) > 0.5

    def summary(self) -> str:
        flag = "✓ 有效" if self.is_effective else "✗ 无效"
        return (f"{self.factor_name} (前瞻{self.forward_days}天) | "
                f"IC均值={self.ic_mean:.4f} ICIR={self.icir:.2f} "
                f"t={self.t_stat:.2f} 正率={self.ic_positive_ratio:.1%} {flag}")


class FactorAnalyzer:
    """因子分析与验证器。"""

    def __init__(self, db):
        self._db = db
        self._data: dict[str, pd.DataFrame] = {}  # {symbol: df}
        self._factors: dict[str, pd.Series] = {}  # {factor_name: MultiIndex Series}
        self._returns: dict[int, pd.Series] = {}  # {forward_days: MultiIndex Series}
        self._symbols: list[str] = []
        self._market_cap: pd.Series = None  # 每日每股票市值

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def load_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        min_amount: float = 5e7,
        adjust: str = '',
        anchor_date: str | None = None,
    ):
        """加载行情数据（与 SignalGenerator 类似但更精简）。

        参数
        ----
        adjust : '' | 'qfq' | 'hfq'
            价格口径。''=不复权；'qfq'=前复权；'hfq'=后复权。
        anchor_date : str | None
            当 adjust='qfq' 时的锚点日期。None 时默认使用 end_date。
            这实现“以回测区间末端为锚点的动态前复权”，保证锚点日真实价格不变。
        """
        if adjust not in {'', 'qfq', 'hfq'}:
            raise ValueError(f'未知 adjust: {adjust}')
        if anchor_date is None:
            anchor_date = end_date

        # 每次重新加载数据时清空旧缓存，避免不同价格口径混用
        self._data = {}
        self._factors = {}
        self._returns = {}
        self._symbols = []
        self._fin_cache = {}

        def _chunked_query(symbol_list: list[str], sql_template: str, extra_params: tuple = ()):
            all_rows = []
            chunk_size = 800
            for i in range(0, len(symbol_list), chunk_size):
                batch = symbol_list[i:i + chunk_size]
                placeholders = ",".join("?" * len(batch))
                all_rows.extend(
                    self._db.query(
                        sql_template.format(placeholders=placeholders),
                        tuple(batch) + extra_params,
                    )
                )
            return all_rows

        rows = _chunked_query(
            symbols,
            """
            SELECT symbol, trade_date, open, high, low, close, volume, amount,
                   turnover_rate, pe_ttm, pb, total_mv, circ_mv
            FROM stock_daily
            WHERE symbol IN ({placeholders})
              AND trade_date BETWEEN ? AND ?
            ORDER BY symbol, trade_date
            """,
            (start_date, end_date),
        )
        if not rows:
            raise RuntimeError("无行情数据")

        df_all = pd.DataFrame(rows, columns=[
            "symbol", "trade_date", "open", "high", "low", "close",
            "volume", "amount", "turnover_rate",
            "pe_ttm", "pb", "total_mv", "circ_mv",
        ])
        df_all["trade_date"] = pd.to_datetime(df_all["trade_date"])
        df_all = df_all.astype({
            "open": float, "high": float, "low": float,
            "close": float, "volume": float, "amount": float,
        })

        # 加载股本
        share_rows = _chunked_query(
            symbols,
            "SELECT symbol, outstanding_share FROM stock_shares WHERE symbol IN ({placeholders})",
        )
        shares_map = {r[0]: float(r[1]) for r in share_rows if r[1]}

        # 预加载复权因子与估值数据，避免对大股票池逐只查询导致极慢
        adj_rows = _chunked_query(
            symbols,
            "SELECT symbol, trade_date, hfq_factor FROM adj_factor WHERE symbol IN ({placeholders}) ORDER BY symbol, trade_date",
        )
        adj_map = {}
        if adj_rows:
            adj_df = pd.DataFrame(adj_rows, columns=['symbol', 'trade_date', 'hfq_factor'])
            adj_df['trade_date'] = pd.to_datetime(adj_df['trade_date'])
            adj_df['hfq_factor'] = pd.to_numeric(adj_df['hfq_factor'], errors='coerce')
            for symbol, grp_adj in adj_df.groupby('symbol'):
                adj_map[symbol] = pd.Series(grp_adj['hfq_factor'].values, index=grp_adj['trade_date']).sort_index()

        val_rows_all = _chunked_query(
            symbols,
            "SELECT symbol, trade_date, total_mv, pb, pe_ttm FROM stock_valuation WHERE symbol IN ({placeholders}) ORDER BY symbol, trade_date",
        )
        valuation_map = {}
        if val_rows_all:
            val_all_df = pd.DataFrame(val_rows_all, columns=['symbol', 'trade_date', 'total_mv', 'pb', 'pe_ttm'])
            val_all_df['trade_date'] = pd.to_datetime(val_all_df['trade_date'])
            for symbol, grp_val in val_all_df.groupby('symbol'):
                g = grp_val[['trade_date', 'total_mv', 'pb', 'pe_ttm']].copy()
                for col in ['total_mv', 'pb', 'pe_ttm']:
                    g[col] = pd.to_numeric(g[col], errors='coerce')
                valuation_map[symbol] = g.set_index('trade_date').sort_index()

        valid = []
        for symbol, grp in df_all.groupby("symbol"):
            grp = grp.set_index("trade_date").sort_index()
            grp.attrs["symbol"] = symbol

            # 先保留一份原始不复权价格，供后续成交/验证或双轨扩展使用
            for col in ['open', 'high', 'low', 'close']:
                grp[f'raw_{col}'] = grp[col].astype(float)

            # 若指定复权口径，则在此把价格列切换到复权价
            if adjust != '':
                factors = adj_map.get(symbol)
                if factors is not None and not factors.empty:
                    try:
                        if adjust == 'qfq':
                            anchor = get_anchor_factor(factors, anchor_date)
                            grp = apply_qfq(grp, factors, anchor)
                        else:
                            grp = apply_hfq(grp, factors)
                    except Exception as e:
                        logger.warning('复权处理失败 [{}]: {}，回退到不复权', symbol, e)
                else:
                    logger.warning('标的 {} 缺少复权因子，回退到不复权', symbol)

            # 市值：优先从 stock_valuation 表读取（百度接口，无未来函数）
            # 回退到 outstanding_share 时间序列（如有）
            # 最后回退到 stock_shares 单一值（有未来函数风险）
            val_df = valuation_map.get(symbol)
            if val_df is not None and not val_df.empty:
                # 估值数据是日历日(~9天频率)，reindex到交易日并forward-fill
                val_aligned = val_df.reindex(grp.index, method='ffill')
                grp["market_cap"] = val_aligned["total_mv"]
                if "pb" in val_aligned.columns:
                    grp["pb"] = val_aligned["pb"]
                if "pe_ttm" in val_aligned.columns:
                    grp["pe_ttm"] = val_aligned["pe_ttm"]
            elif 'outstanding_share' in grp.columns and grp['outstanding_share'].notna().any():
                grp["market_cap"] = grp["close"] * grp["outstanding_share"] / 1e8
            elif symbol in shares_map:
                # ⚠ 未来函数: 用最新股本计算所有历史日期的市值
                grp["market_cap"] = grp["close"] * shares_map[symbol] / 1e8
            else:
                grp["market_cap"] = np.nan

            # 流动性过滤
            if grp["amount"].mean() < min_amount:
                continue
            grp.attrs['price_adjust'] = adjust
            grp.attrs['anchor_date'] = anchor_date
            self._data[symbol] = grp
            valid.append(symbol)

        self._symbols = valid
        logger.info("加载 {} 只标的 | {} ~ {} | adjust={} | anchor={}", len(valid), start_date, end_date, adjust or 'raw', anchor_date)

    # ------------------------------------------------------------------
    # 财报数据加载（防未来函数对齐）
    # ------------------------------------------------------------------

    # 缓存: {symbol: {field: pd.Series(trade_date索引)}}
    _fin_cache: dict = {}

    def _load_financial_series(self, symbol: str, field: str) -> pd.Series:
        """
        读取某只股票的某个财报字段，按 announce_date 对齐到交易日。

        防未来函数核心：
          对每个交易日 T，取 announce_date <= T 的最新一条财报记录。
          用 pd.merge_asof(direction='backward') 实现。

        返回: pd.Series, index=trade_date, values=字段值
        """
        df = self._data.get(symbol)
        if df is None:
            return pd.Series(dtype=float)

        if symbol in self._fin_cache and field in self._fin_cache[symbol]:
            return self._fin_cache[symbol][field]

        rows = self._db.query(
            "SELECT report_date, announce_date, report_type, is_preliminary, "
            "net_profit_attr, revenue, operating_income, operating_cost, "
            "operating_cf, capex, free_cf, equity_attr, equity, "
            "total_assets, total_liabilities, cash, inventory, "
            "accounts_receivable, goodwill, interest_bearing_debt, bvps, eps "
            "FROM financial_report WHERE symbol=? AND announce_date IS NOT NULL "
            "AND announce_date != '' ORDER BY announce_date, is_preliminary",
            (symbol,),
        )
        if not rows:
            empty = pd.Series(np.nan, index=df.index, name=field)
            return empty

        fin = pd.DataFrame(
            [tuple(r) for r in rows],
            columns=["report_date", "announce_date", "report_type", "is_preliminary",
                     "net_profit_attr", "revenue", "operating_income", "operating_cost",
                     "operating_cf", "capex", "free_cf", "equity_attr", "equity",
                     "total_assets", "total_liabilities", "cash", "inventory",
                     "accounts_receivable", "goodwill", "interest_bearing_debt", "bvps", "eps"],
        )
        fin["announce_date"] = pd.to_datetime(fin["announce_date"])
        # 同一公告日多条（预告+正式）取最后一条（正式版，因 ORDER BY is_preliminary 升序）
        fin = fin.drop_duplicates("announce_date", keep="last")
        fin = fin.sort_values("announce_date").reset_index(drop=True)

        trade_df = df.index.to_frame(index=False).rename(columns={"trade_date": "td"})
        merged = pd.merge_asof(
            trade_df.sort_values("td"),
            fin,
            left_on="td",
            right_on="announce_date",
            direction="backward",
        ).set_index("td")

        # 缓存所有字段
        if symbol not in self._fin_cache:
            self._fin_cache[symbol] = {}
            for c in fin.columns:
                if c in merged.columns:
                    s = merged[c].copy()
                    s.index.name = "trade_date"
                    self._fin_cache[symbol][c] = s

        result = self._fin_cache[symbol].get(field)
        if result is None:
            result = pd.Series(np.nan, index=df.index, name=field)
        return result

    def _calc_ttm(self, symbol: str, field: str = "net_profit_attr") -> pd.Series:
        """
        计算滚动 TTM（trailing twelve months）值。

        财报累计逻辑：Q1=1-3月, 中报=1-6月, Q3=1-9月, 年报=1-12月
        TTM 公式：
          - 年报: 直接用年报值
          - 季报: 去年年报 + 今年本期 - 去年同期

        返回: pd.Series, index=trade_date, TTM 值（防未来函数对齐）
        """
        df = self._data.get(symbol)
        if df is None:
            return pd.Series(dtype=float)

        rows = self._db.query(
            "SELECT report_date, announce_date, report_type, "
            f"{field} FROM financial_report WHERE symbol=? "
            "AND announce_date IS NOT NULL AND announce_date != '' "
            "ORDER BY announce_date, is_preliminary",
            (symbol,),
        )
        if not rows:
            return pd.Series(np.nan, index=df.index)

        raw = pd.DataFrame(
            [tuple(r) for r in rows],
            columns=["report_date", "announce_date", "report_type", field],
        )
        raw["report_date"] = pd.to_datetime(raw["report_date"])
        raw["announce_date"] = pd.to_datetime(raw["announce_date"])
        raw = raw.drop_duplicates("report_date", keep="last")

        # 构造 report_date → TTM 映射
        ttm_map = {}
        for rd, row in raw.set_index("report_date").iterrows():
            rt = row["report_type"]
            val = row[field]
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            val = float(val)

            if rt == "annual":
                ttm_map[rd] = val
            else:
                last_annual = pd.Timestamp(rd.year - 1, 12, 31)
                if last_annual not in raw["report_date"].values:
                    continue
                annual_val = raw.loc[raw["report_date"] == last_annual, field].iloc[0]
                if annual_val is None or (isinstance(annual_val, float) and pd.isna(annual_val)):
                    continue
                annual_val = float(annual_val)

                spy_map = {
                    "q1": pd.Timestamp(rd.year - 1, 3, 31),
                    "interim": pd.Timestamp(rd.year - 1, 6, 30),
                    "q3": pd.Timestamp(rd.year - 1, 9, 30),
                }
                spy = spy_map.get(rt)
                if spy is None or spy not in raw["report_date"].values:
                    continue
                spy_val = raw.loc[raw["report_date"] == spy, field].iloc[0]
                if spy_val is None or (isinstance(spy_val, float) and pd.isna(spy_val)):
                    continue
                spy_val = float(spy_val)

                ttm_map[rd] = annual_val + val - spy_val

        if not ttm_map:
            return pd.Series(np.nan, index=df.index)

        # 对齐到交易日：用 announce_date 做 merge_asof
        ttm_df = pd.DataFrame(
            [(rd, v) for rd, v in ttm_map.items()],
            columns=["report_date", "ttm"],
        )
        ann_df = raw[["report_date", "announce_date"]].drop_duplicates("report_date", keep="last")
        ttm_df = ttm_df.merge(ann_df, on="report_date", how="left")
        ttm_df = ttm_df.sort_values("announce_date").drop_duplicates("announce_date", keep="last")

        trade_df = df.index.to_frame(index=False).rename(columns={"trade_date": "td"})
        merged = pd.merge_asof(
            trade_df.sort_values("td"),
            ttm_df[["announce_date", "ttm"]],
            left_on="td",
            right_on="announce_date",
            direction="backward",
        )
        result = merged.set_index("td")["ttm"]
        result.index.name = "trade_date"
        return result

    # ------------------------------------------------------------------
    # 因子构造
    # ------------------------------------------------------------------

    def add_factor(self, name: str, fn: Callable, **kwargs):
        """
        注册一个因子。

        fn 接收单只股票的 DataFrame，返回一个 pd.Series（index=trade_date）。
        kwargs 会传给 fn。

        内置因子示例：
            analyzer.add_factor('momentum_5d', lambda df: df['close'].pct_change(5))
            analyzer.add_factor('rsi_14', lambda df: calc_rsi(df['close'], 14))
            analyzer.add_factor('turnover', lambda df: df['turnover_rate'])
        """
        records = []
        for symbol in self._symbols:
            df = self._data[symbol]
            try:
                factor_values = fn(df, **kwargs)
                for date, val in factor_values.items():
                    if pd.notna(val):
                        records.append((date, symbol, val))
            except Exception as e:
                logger.debug("因子 {} 计算失败 {}: {}", name, symbol, e)

        if not records:
            logger.warning("因子 {} 无有效值", name)
            return

        factor_df = pd.DataFrame(records, columns=['date', 'symbol', 'value'])
        factor_df = factor_df.set_index(['date', 'symbol']).sort_index()
        self._factors[name] = factor_df['value']
        logger.info("因子 {} 注册完成 | {} 条", name, len(factor_df))

    def add_builtin_factors(self):
        """注册通用因子（教科书定义，公开、无调参）。

        开源仓库只保留通用定义；研究中实际筛选出的因子组合不在此处。
        自定义因子有两种扩展方式：

            analyzer.add_factor('my_factor', my_fn)
            strategies.factor_zoo.register_factor('my_factor', my_fn)
        """
        from strategies.factor_zoo import BUILTIN_FACTORS

        for spec in BUILTIN_FACTORS:
            if spec.name in self._factors:
                continue
            self.add_factor(spec.name, spec.fn)

    # ------------------------------------------------------------------
    # 数据接口（供 MultiFactorSynthesizer 使用）
    # ------------------------------------------------------------------

    def get_factor_matrix(self, factor_names: list[str] = None) -> pd.DataFrame:
        """
        返回对齐的因子矩阵。

        行 = (date, symbol) MultiIndex
        列 = 各因子名
        值 = 因子值

        参数
        ----
        factor_names : 要包含的因子列表，None=全部（排除 _neutral 后缀）
        """
        if factor_names is None:
            factor_names = [n for n in self._factors if not n.endswith('_neutral')]

        cols = {}
        for name in factor_names:
            f = self._factors.get(name)
            if f is not None:
                cols[name] = f

        if not cols:
            return pd.DataFrame()

        matrix = pd.DataFrame(cols)
        return matrix.sort_index()

    def get_forward_returns(self, forward_days: int) -> pd.Series:
        """
        返回前瞻收益率序列（供 MultiFactorSynthesizer 作为目标变量）。
        """
        return self._get_returns(forward_days, basis='close')

    def get_forward_returns_open(self, forward_days: int) -> pd.Series:
        """
        对齐回测收益口径的收益序列：因子日 d 的收益 = 次日(d+1)开盘买入、
        持有 forward_days 天、(d+1+days)收盘卖出，即 close[d+1+days] / open[d+1] - 1。

        与 get_forward_returns（close-to-close）的区别：买入价用次日开盘 open 而非
        当日收盘，消除 IC 收益口径与回测(次日开盘买、收盘卖)的错位。
        """
        return self._get_returns(forward_days, basis='open')

    # ------------------------------------------------------------------
    # IC 分析
    # ------------------------------------------------------------------

    def _get_returns(self, forward_days: int, basis: str = 'close') -> pd.Series:
        """
        计算前瞻收益率。

        参数
        ----
        forward_days : 前瞻/调仓周期天数
        basis        : 收益口径
            - 'close' : 收盘买入 → 收盘卖出，close[t] → close[t+days]（默认）
            - 'open'  : 次日开盘买入 → 收盘卖出，close[t+days]/open[t] - 1
                       （与回测一致：T 日信号、T+1 开盘买、T+days 收盘卖）
        """
        key = f'{basis}_{forward_days}'
        if key in self._returns:
            return self._returns[key]

        if basis == 'close':
            ret = self._close_close_returns(forward_days)
        elif basis == 'open':
            ret = self._open_close_returns(forward_days)
        else:
            raise ValueError(f"未知收益口径 basis: {basis}")
        self._returns[key] = ret
        return ret

    def _close_close_returns(self, forward_days: int) -> pd.Series:
        """close 口径前瞻收益：因子日 t 的收益 = close[t+days] / close[t] - 1。"""
        records = []
        for symbol in self._symbols:
            df = self._data[symbol]
            fwd_ret = df['close'].pct_change(forward_days).shift(-forward_days)
            for date, val in fwd_ret.items():
                if pd.notna(val):
                    records.append((date, symbol, val))
        ret_df = pd.DataFrame(records, columns=['date', 'symbol', 'ret'])
        ret_df = ret_df.set_index(['date', 'symbol']).sort_index()
        return ret_df['ret']

    def _open_close_returns(self, forward_days: int) -> pd.Series:
        """open 口径前瞻收益：因子日 t 的收益 = 次日(t+1)开盘买入、收盘卖出。

        对应回测执行（T 收盘信号 / T+1 开盘买 / T+1+days 收盘卖）：
            ret(t) = close[t+1+days] / open[t+1] - 1
        """
        records = []
        for symbol in self._symbols:
            df = self._data[symbol]
            buy_open = df['open'].shift(-1)                      # t -> open[t+1]
            sell_close = df['close'].shift(-(1 + forward_days))  # t -> close[t+1+days]
            fwd_ret = sell_close / buy_open - 1
            fwd_ret = fwd_ret.replace([np.inf, -np.inf], np.nan)
            for date, val in fwd_ret.items():
                if pd.notna(val):
                    records.append((date, symbol, val))
        ret_df = pd.DataFrame(records, columns=['date', 'symbol', 'ret'])
        ret_df = ret_df.set_index(['date', 'symbol']).sort_index()
        return ret_df['ret']

        records = []
        for symbol in self._symbols:
            df = self._data[symbol]
            fwd_ret = df['close'].pct_change(forward_days).shift(-forward_days)
            for date, val in fwd_ret.items():
                if pd.notna(val):
                    records.append((date, symbol, val))

        ret_df = pd.DataFrame(records, columns=['date', 'symbol', 'ret'])
        ret_df = ret_df.set_index(['date', 'symbol']).sort_index()
        self._returns[forward_days] = ret_df['ret']
        return ret_df['ret']

    def calc_ic(self, factor_name: str, forward_days: int = 5,
               non_overlapping: bool = True,
               return_basis: str = 'close') -> ICResult:
        """
        计算 IC（Information Coefficient）。

        非重叠模式（正确）:
          每 forward_days 天取一个截面, rank(factor) vs rank(return), 算一次Spearman
          周期之间不重叠, 每个IC独立
          forward_days=20 → 每年约12.6个独立IC

        重叠模式（旧, 不推荐）:
          每天算一次, 前瞻forward_days天 → 相邻IC高度自相关(95%+)

        参数
        ----
        factor_name     : 因子名称
        forward_days    : 前瞻/调仓周期天数
        non_overlapping : True=非重叠(正确), False=每日重叠(旧模式)
        return_basis    : 收益口径 'close'|'open'，见 _get_returns。
                          'open' 对应回测（T+1 开盘买入 / 收盘卖出）。
        """
        factor = self._factors.get(factor_name)
        if factor is None:
            raise ValueError(f"因子 {factor_name} 未注册")

        returns = self._get_returns(forward_days, basis=return_basis)

        common_idx = factor.index.intersection(returns.index)
        factor_aligned = factor.loc[common_idx]
        returns_aligned = returns.loc[common_idx]

        all_dates = sorted(common_idx.get_level_values('date').unique())
        if non_overlapping:
            sample_dates = all_dates[::forward_days]
        else:
            sample_dates = all_dates
        ic_values = []

        for date in sample_dates:
            mask = factor_aligned.index.get_level_values('date') == date
            f_vals = factor_aligned[mask].values
            r_vals = returns_aligned[mask].values

            # 至少需要 5 只股票才有意义
            valid = ~(np.isnan(f_vals) | np.isnan(r_vals))
            if valid.sum() < 5:
                continue

            f_valid = f_vals[valid]
            r_valid = r_vals[valid]

            if f_valid.std() == 0 or r_valid.std() == 0:
                continue

            ic, _ = stats.spearmanr(f_valid, r_valid)
            if not np.isnan(ic):
                ic_values.append((date, ic))

        ic_series = pd.Series([v for _, v in ic_values],
                              index=pd.DatetimeIndex([d for d, _ in ic_values]),
                              name='IC')

        ic_mean = float(ic_series.mean())
        ic_std = float(ic_series.std())
        icir = ic_mean / ic_std if ic_std > 0 else 0
        positive_ratio = float((ic_series > 0).mean())
        n = len(ic_series)
        t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 and n > 0 else 0

        return ICResult(
            factor_name=factor_name,
            forward_days=forward_days,
            ic_series=ic_series,
            ic_mean=ic_mean,
            ic_std=ic_std,
            icir=icir,
            ic_positive_ratio=positive_ratio,
            t_stat=float(t_stat),
        )

    # ------------------------------------------------------------------
    # 因子中性化
    # ------------------------------------------------------------------

    def neutralize_by_cap(self, factor_name: str) -> str:
        """
        对市值做线性中性化，返回中性化后的因子名。

        factor_neutral = factor - α - β × log(market_cap)

        如果中性化后 IC 仍然有效，说明因子有独立于市值的预测力。
        """
        factor = self._factors[factor_name]
        neutralized = []

        dates = factor.index.get_level_values('date').unique()
        for date in dates:
            mask = factor.index.get_level_values('date') == date
            f_vals = factor[mask]
            syms = f_vals.index.get_level_values('symbol')

            # 获取当日市值（按 syms 顺序）
            mc_vals = []
            for sym in syms:
                df = self._data.get(sym)
                if df is not None and date in df.index:
                    mc = df.loc[date, 'market_cap']
                    mc_vals.append(mc if pd.notna(mc) else np.nan)
                else:
                    mc_vals.append(np.nan)

            log_mc = np.log(np.array(mc_vals, dtype=float).clip(min=1e-8))
            f_arr = f_vals.values

            # 过滤 NaN
            valid = ~(np.isnan(f_arr) | np.isnan(log_mc))
            if valid.sum() < 5:
                continue

            from sklearn.linear_model import LinearRegression
            X = log_mc[valid].reshape(-1, 1)
            y = f_arr[valid]
            reg = LinearRegression().fit(X, y)
            residual = y - reg.predict(X)

            valid_syms = syms[valid]
            for i, sym in enumerate(valid_syms):
                neutralized.append((date, sym, residual[i]))

        neu_df = pd.DataFrame(neutralized, columns=['date', 'symbol', 'value'])
        neu_df = neu_df.set_index(['date', 'symbol']).sort_index()

        neu_name = f"{factor_name}_neutral"
        self._factors[neu_name] = neu_df['value']
        logger.info("因子 {} 中性化完成 → {}", factor_name, neu_name)
        return neu_name


    def orthogonalize_factor(
        self,
        factor_name: str,
        base_factors: list[str],
        log_transform: bool = True,
    ) -> str:
        """
        对目标因子做「对一组基础因子」的逐日横截面线性回归，取残差作为正交化因子。

        用于增量因子体检：剔除候选因子中与基础因子共线的那部分信息后，
        仅保留纯增量（残差），再评估其 IC，判断该因子是否真正带来额外预测力。

        实现：
          - 逐日 OLS：target_i = a + sum_j b_j * base_j + epsilon_i
          - 残差 epsilon_i 即正交化后的因子值（对基础因子无线性暴露）
          - 基础因子逐日做标准化（cross-sectional z-score），避免量纲影响

        参数
        ----
        factor_name   : 候选（增量）因子名
        base_factors  : 基础因子名列表
        log_transform : 对目标因子先做 sign(x)*log1p(|x|) 平滑，缓解极端值

        返回新因子名 ``{factor_name}_ortho``，注册到 self._factors。
        """
        if factor_name not in self._factors:
            raise ValueError(f"因子 {factor_name} 未注册")

        from sklearn.linear_model import LinearRegression

        factor = self._factors[factor_name]
        base_series = {f: self._factors.get(f) for f in base_factors}
        base_series = {f: s for f, s in base_series.items() if s is not None}
        if not base_series:
            raise ValueError("base_factors 均未注册")

        idx = factor.index
        base_matrix = pd.DataFrame({f: s.reindex(idx) for f, s in base_series.items()})
        target = factor.reindex(idx)

        dates = idx.get_level_values('date').unique()
        residual_records = []

        for date in dates:
            date_mask = target.index.get_level_values('date') == date
            y = target[date_mask].values.astype(float)
            X = base_matrix[date_mask].values.astype(float)
            syms = target.index[date_mask].get_level_values('symbol')

            if log_transform:
                y_work = np.sign(y) * np.log1p(np.abs(y))
            else:
                y_work = y

            # 行内过滤 NaN（任一列缺失则剔除该行）
            valid = ~(np.isnan(y_work) | np.isnan(X).any(axis=1))
            if valid.sum() < 10:
                continue

            y_v = y_work[valid]
            X_v = X[valid]
            # 逐日标准化基础因子（避免量纲/共线问题）
            mu = X_v.mean(axis=0)
            sd = X_v.std(axis=0)
            sd[sd == 0] = 1.0
            X_s = (X_v - mu) / sd

            reg = LinearRegression().fit(X_s, y_v)
            residual = y_v - reg.predict(X_s)

            valid_syms = syms[valid]
            for i, sym in enumerate(valid_syms):
                residual_records.append((date, sym, residual[i]))

        ortho_df = pd.DataFrame(residual_records, columns=['date', 'symbol', 'value'])
        if ortho_df.empty:
            raise RuntimeError(f"因子 {factor_name} 正交化后无有效观测")
        ortho_df = ortho_df.set_index(['date', 'symbol']).sort_index()

        ortho_name = f"{factor_name}_ortho"
        self._factors[ortho_name] = ortho_df['value']
        logger.info("因子 {} 正交化完成 → {} | 观测数 {}", factor_name, ortho_name, len(ortho_df))
        return ortho_name


    # ------------------------------------------------------------------
    # 分层回测
    # ------------------------------------------------------------------

    def layer_backtest(
        self,
        factor_name: str,
        n_layers: int = 5,
        forward_days: int = 5,
        return_basis: str = 'close',
    ) -> pd.DataFrame:
        """
        分层回测：按因子值分 N 层，看各层收益单调性。

        如果因子有效，收益应随因子值单调变化。
        """
        factor = self._factors[factor_name]
        returns = self._get_returns(forward_days, basis=return_basis)

        common_idx = factor.index.intersection(returns.index)
        factor_aligned = factor.loc[common_idx]
        returns_aligned = returns.loc[common_idx]

        dates = common_idx.get_level_values('date').unique()
        layer_returns = defaultdict(list)

        for date in dates:
            mask = factor_aligned.index.get_level_values('date') == date
            f_vals = factor_aligned[mask]
            r_vals = returns_aligned[mask]

            valid = ~(f_vals.isna() | r_vals.isna())
            if valid.sum() < n_layers:
                continue

            f_valid = f_vals[valid]
            r_valid = r_vals[valid]

            # 按因子值排序，分 N 层
            sorted_idx = f_valid.sort_values().index
            layer_size = len(sorted_idx) // n_layers

            for layer in range(n_layers):
                start = layer * layer_size
                end = (layer + 1) * layer_size if layer < n_layers - 1 else len(sorted_idx)
                layer_syms = sorted_idx[start:end]
                layer_ret = r_valid.loc[layer_syms].mean()
                layer_returns[f"L{layer+1}"].append(layer_ret)

        # 汇各层平均收益
        result = {}
        for layer, rets in layer_returns.items():
            result[layer] = {
                'mean_return': np.mean(rets),
                'std': np.std(rets),
                'n_days': len(rets),
            }

        df = pd.DataFrame(result).T
        df.index.name = 'layer'
        return df

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------

    def print_ic_summary(self, ic: ICResult):
        """打印 IC 分析摘要。"""
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=f"IC 分析: {ic.factor_name}", box=box.ROUNDED)
        table.add_column("指标", style="cyan")
        table.add_column("数值", justify="right")

        rows = [
            ("前瞻天数",       f"{ic.forward_days}天"),
            ("IC 均值",       f"{ic.ic_mean:.4f}"),
            ("IC 标准差",     f"{ic.ic_std:.4f}"),
            ("ICIR",          f"{ic.icir:.2f}"),
            ("IC 正率",       f"{ic.ic_positive_ratio:.1%}"),
            ("t 统计量",      f"{ic.t_stat:.2f}"),
            ("交易日数",      str(len(ic.ic_series))),
            ("判定",          "✓ 有效" if ic.is_effective else "✗ 无效"),
        ]
        for k, v in rows:
            table.add_row(k, v)
        console.print(table)

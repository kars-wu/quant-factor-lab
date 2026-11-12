"""模拟盘数据层：加载复权行情面板 + 构建股票池掩码。

设计原则：
  - 面板统一为 (date, ticker) MultiIndex，date 为 Timestamp，ticker 为 6 位字符串；
  - 股票池掩码在同一索引上给出 bool Series，只使用 T 日及更早信息；
  - 任何「当前存活/当前成分」过滤都不会被用来回溯历史（那是未来函数）。
"""
from __future__ import annotations

import pandas as pd
from loguru import logger

from data.database import Database
from strategies.factor_analysis import FactorAnalyzer

LOADING_START = '2018-01-01'


def active_symbols(db: Database, start: str, end: str) -> list[str]:
    """信号域：区间内有行情的 A 股。

    注意：这里用 ``is_active=1`` 只是为了减少无谓的加载量。做严格历史研究时，
    应改用带退市日期的标的表，或直接用行情表里出现过的 symbol，避免幸存者偏差。
    """
    rows = db.query(
        """SELECT sd.symbol FROM stock_daily sd
           JOIN stock_info si ON sd.symbol = si.symbol
           WHERE si.asset_type = 'stock' AND si.is_active = 1
             AND sd.trade_date BETWEEN ? AND ?
           GROUP BY sd.symbol""",
        (start, end),
    )
    return [r[0] for r in rows]


def load_panel(
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: list[str] | None = None,
    min_amount: float = 1e4,
    adjust: str = 'qfq',
    anchor_date: str | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """加载行情面板。

    返回
    ----
    panel : DataFrame，index=(date, ticker)，列含 open/high/low/close/volume/amount
    data  : {symbol: DataFrame}，index=DatetimeIndex（引擎逐日撮合用）
    """
    db = Database()
    end = end_date or db.query_one('SELECT MAX(trade_date) FROM stock_daily')[0]
    start = start_date or LOADING_START
    if symbols is None:
        symbols = active_symbols(db, start, end)

    analyzer = FactorAnalyzer(db)
    analyzer.load_data(
        symbols, start, end,
        min_amount=min_amount,
        adjust=adjust,
        anchor_date=anchor_date or end,
    )
    data = analyzer._data

    parts = []
    for symbol in analyzer._symbols:
        df = data[symbol]
        sub = df[['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
        sub['ticker'] = symbol
        sub['date'] = sub.index
        parts.append(sub)
    if not parts:
        raise RuntimeError('无行情数据，请先执行数据初始化：python -m scheduler.daily_update --init-all')

    panel = pd.concat(parts).set_index(['date', 'ticker']).sort_index()
    panel = panel[~panel.index.duplicated()]
    panel.index = panel.index.set_names(['date', 'ticker'])
    logger.info('面板加载完成: {} 行, {} 只标的, {} ~ {}',
                len(panel), panel.index.get_level_values('ticker').nunique(), start, end)
    return panel, data


def _mask_from_wide(wide: pd.DataFrame, panel: pd.DataFrame) -> pd.Series:
    idx = panel.index
    # pandas 3 的 stack 不再接受 dropna，缺失行由 reindex 后的 fillna 兜住
    mask = wide.stack().reindex(idx).fillna(False).astype(bool)
    mask.index = mask.index.set_names(['date', 'symbol'])
    return mask


def build_universe(
    panel: pd.DataFrame,
    pool: str = 'top_n_amount',
    db: Database | None = None,
    top_n: int = 500,
    lookback: int = 20,
) -> pd.Series:
    """按池类型构造 (date, ticker) 的成员掩码。

    pool：
      - ``all``            ：面板内所有标的
      - ``top_n_amount``   ：过去 ``lookback`` 日平均成交额前 ``top_n``，并 shift(1)
                             （T 日池只用 T-1 及更早的成交额）
      - ``hs300_like``     ：按 T-1 日流通市值取前 300（指数代理，非真实成分）
    """
    close_w = panel['close'].unstack('ticker')
    if pool == 'all':
        return _mask_from_wide(close_w.notna(), panel)

    if pool == 'top_n_amount':
        amount_w = panel['amount'].unstack('ticker')
        avg_amt = amount_w.rolling(lookback, min_periods=max(2, lookback // 2)).mean()
        rank = avg_amt.shift(1).rank(axis=1, method='first', ascending=False)
        member = (rank <= top_n) & close_w.notna()
        return _mask_from_wide(member, panel)

    if pool == 'hs300_like':
        db = db or Database()
        rows = db.query(
            "SELECT symbol, trade_date, circ_mv FROM stock_daily "
            "WHERE circ_mv IS NOT NULL AND trade_date BETWEEN ? AND ?",
            (str(close_w.index.min().date()), str(close_w.index.max().date())),
        )
        mv = pd.DataFrame(rows, columns=['symbol', 'trade_date', 'circ_mv'])
        if mv.empty:
            raise RuntimeError('stock_daily.circ_mv 为空，无法构建市值代理池')
        mv['trade_date'] = pd.to_datetime(mv['trade_date'])
        mv_w = mv.pivot(index='trade_date', columns='symbol', values='circ_mv')
        mv_w = mv_w.reindex(index=close_w.index).ffill().reindex(columns=close_w.columns)
        rank = mv_w.shift(1).rank(axis=1, method='first', ascending=False)
        member = (rank <= 300) & close_w.notna()
        return _mask_from_wide(member, panel)

    raise ValueError(f'未知股票池: {pool}')


def trading_dates_of(panel: pd.DataFrame) -> list[str]:
    return sorted({str(pd.Timestamp(d).date()) for d in panel.index.get_level_values(0)})


def market_filter_hs300(panel: pd.DataFrame, index_code: str = '000300',
                        db: Database | None = None) -> pd.Series:
    """沪深300 > MA20 且 > MA60 的择时开关，index=date(str)，值为 bool。"""
    db = db or Database()
    start = str(pd.Timestamp(panel.index.get_level_values(0).min()).date())
    end = str(pd.Timestamp(panel.index.get_level_values(0).max()).date())
    rows = db.query(
        "SELECT trade_date, close FROM index_daily WHERE symbol = ? "
        "AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
        (index_code, start, end),
    )
    if not rows:
        raise RuntimeError(f'index_daily 缺少 {index_code} 行情')
    series = pd.Series({pd.Timestamp(r[0]): float(r[1]) for r in rows}).sort_index()
    ma20 = series.rolling(20, min_periods=10).mean()
    ma60 = series.rolling(60, min_periods=30).mean()
    flag = (series > ma20) & (series > ma60)
    flag = flag.astype('boolean').fillna(True).astype(bool)
    return pd.Series(flag.values, index=[str(d.date()) for d in flag.index])


def benchmark_nav(panel: pd.DataFrame, pool_mask: pd.Series,
                  start: str | None = None) -> pd.Series:
    """同池等权买入持有基准（按收盘价，等权日度再平衡收益累乘）。"""
    close_w = panel['close'].unstack('ticker')
    mask_w = (
        pool_mask.rename_axis(['date', 'ticker'])
        .unstack('ticker')
        .reindex(index=close_w.index, columns=close_w.columns)
        .fillna(False)
        .astype(bool)
    )
    ret = close_w.pct_change().where(mask_w)
    daily = ret.mean(axis=1).dropna()
    if start:
        daily = daily[daily.index >= pd.Timestamp(start)]
    nav = (1.0 + daily).cumprod()
    return nav / nav.iloc[0] if len(nav) else nav

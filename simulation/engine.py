"""
模拟盘日频引擎：T 日收盘生成信号 → T+1 开盘成交 → 每日记录净值。

与回测口径一致：
  - 信号基于 T 日收盘后可得信息（因子已显式 shift(1)）
  - 买入/卖出均以 T+1 开盘价成交
  - 长仓：T+1 买，持有到卖信号或最长持有期；半仓滚动（每仓 5%）
  - 多空：D1 多 / D10 空 各半仓（fixed_initial 5‰×N），T+2 收盘平
  - 成本：单边 0.15% 佣金 + 卖出 0.1% 印花税（与回测 CostModel 一致）

数据流：
  update_all(panel_df, trading_dates)  — 给定最新面板，把所有策略推进到最新交易日
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from simulation import state
from simulation.strategies import StrategySpec, all_specs

# 成本参数（与 strategies/cost_model.py 完全一致, 保证模拟盘与线下回测可比）
COMMISSION_RATE = 0.0001      # 佣金 万一（买卖双向）
STAMP_TAX = 0.001             # 印花税 千一（仅卖出）
SLIPPAGE_RATE = 0.0           # 竞价策略按开盘价成交，不计滑点
MIN_COMMISSION = 5.0          # 最低佣金 5 元


def _buy_cost(price: float, qty: int) -> float:
    amount = price * qty
    return max(amount * COMMISSION_RATE, MIN_COMMISSION) + amount * SLIPPAGE_RATE


def _sell_cost(price: float, qty: int) -> float:
    amount = price * qty
    return max(amount * COMMISSION_RATE, MIN_COMMISSION) + amount * (STAMP_TAX + SLIPPAGE_RATE)


def _fmt_side(positions: dict) -> str:
    return ','.join(f'{s}:{p["qty"]}@{p["avg_cost"]:.2f}' for s, p in positions.items())


def _ts(d):
    """字符串日期 → Timestamp（df index 为 DatetimeIndex 时用）。"""
    return pd.Timestamp(d)


def _has(df, d):
    return _ts(d) in df.index


def _px(df, d, col):
    return float(df.loc[_ts(d), col])


def _mark_px(df, d, col):
    """净值估值价格：当日无行情时，回看最近一个有效价格。

    停牌/缺行情不能导致持仓市值从组合中消失，否则净值会出现
    “停牌日假回撤、复牌日假反弹”。这里仅用于 mark-to-market，
    不改变买卖成交规则。
    """
    ts = _ts(d)
    if ts in df.index:
        return float(df.loc[ts, col])
    s = df.loc[:ts, col].dropna()
    if s.empty:
        raise ValueError(f'无 {d} 及之前的有效价格')
    return float(s.iloc[-1])


def _nearest_date(dates: list[str], target: str, backward: bool = True) -> str | None:
    """在 dates 中找到 <= target 的最近日期（backward=True）或 >= target 的最近日期。"""
    import bisect
    idx = bisect.bisect_right(dates, target) - 1
    if backward:
        return dates[idx] if idx >= 0 else None
    return dates[idx + 1] if idx + 1 < len(dates) else None


def _norm_signal_index(s: pd.Series) -> pd.Series:
    """规范化信号索引为 (date_str, ticker_str)，并排序（构建缓存时调用一次）。"""
    lvl0 = [str(pd.Timestamp(x).date()) for x in s.index.get_level_values(0)]
    lvl1 = [str(x) for x in s.index.get_level_values(1)]
    s = s.set_axis(pd.MultiIndex.from_arrays([lvl0, lvl1], names=['date', 'ticker']))
    return s.sort_index()


def _signal_on_date(signal_fn, panel_df: pd.DataFrame, date: str,
                    cache: pd.Series | None = None, spec: StrategySpec | None = None) -> pd.Series:
    """调用策略信号函数，取指定日期（或之前最近有值日）的信号。
    cache 为预计算的规范化 (date, ticker) 全量信号时直接二分切片（避免每日重算/重建索引）。
    spec.precomputed 且无 cache 时返回空（信号由外部注入）。"""
    if spec is not None and spec.precomputed and cache is None:
        return pd.Series(dtype=float)
    s = cache if cache is not None else signal_fn(panel_df)
    if isinstance(s, pd.Series) and s.index.nlevels == 2:
        if cache is None:
            s = _norm_signal_index(s)
        if spec is not None and spec.signal_ffill is False:
            # 择时/间断信号：缺失日必须为空，不能复用最近一日信号
            try:
                return s.xs(date, level=0)
            except KeyError:
                return pd.Series(dtype=float)
        dates = s.index.get_level_values(0)
        d = _nearest_date(sorted(set(dates)), date)
        if d is None:
            return pd.Series(dtype=float)
        return s.xs(d, level=0)
    if isinstance(s, dict):
        return pd.Series(s)
    return s


def build_pool_index(universe: pd.Series) -> dict[str, set]:
    """把 (date, symbol) 池掩码转成 {date_str: set(symbols)}（仅 True 成员），供引擎 O(1) 过滤。"""
    from collections import defaultdict
    vals = universe.values.astype(bool)
    if not vals.any():
        return {}
    sub = universe[vals]
    lvl0 = [str(pd.Timestamp(x).date()) for x in sub.index.get_level_values(0)]
    lvl1 = [str(x) for x in sub.index.get_level_values(1)]
    idx = defaultdict(set)
    for dt, sym in zip(lvl0, lvl1):
        idx[dt].add(sym)
    return dict(idx)


def build_signal_cache(panel_df: pd.DataFrame,
                       strategy_ids: set | None = None) -> dict[str, pd.Series]:
    """预计算已注册策略的全量信号序列（bootstrap 时调用一次, 索引已规范化）。
    precomputed=True 的策略（如 bestModel）跳过——信号由外部注入 cache。
    同一 signal_fn 被多个策略引用时只计算一次（如 h6/n12 的 hs300/top500 双池版）。
    strategy_ids: 只缓存这些策略（省内存；None=全部）。"""
    cache = {}
    seen: dict[int, pd.Series] = {}
    for spec in all_specs():
        if spec.precomputed:
            continue
        if strategy_ids is not None and spec.strategy_id not in strategy_ids:
            continue
        if id(spec.signal_fn) in seen:
            cache[spec.strategy_id] = seen[id(spec.signal_fn)]
            continue
        try:
            raw = spec.signal_fn(panel_df)
            norm = _norm_signal_index(raw)
            seen[id(spec.signal_fn)] = norm
            cache[spec.strategy_id] = norm
        except Exception as e:
            logger.error('[sim] 信号计算失败 {}: {}', spec.strategy_id, e)
    return cache


def run_strategy_day(spec: StrategySpec, panel_df: pd.DataFrame,
                     data: dict[str, pd.DataFrame],
                     trading_dates: list[str], today: str,
                     signal_cache: dict | None = None,
                     universe: dict | None = None) -> dict:
    """
    推进一个策略到 today（含）：
      1) 处理到期卖出（持仓超过最长持有期 → 今日卖出）
      2) 生成今日信号 → 明日开盘买入
      3) 记录今日净值
    返回 {'status': ..., 'message': ..., 'nav': ...}
    """
    st = state.get_strategy(spec.strategy_id)
    if st is None:
        # 首次运行：初始化（历史回填在 engine.update_all 之前由 bootstrap 完成）
        state.register_strategy(spec.strategy_id, spec.name, spec.description,
                                spec.initial_capital, spec.pool, {}, spec.initial_capital)
        st = state.get_strategy(spec.strategy_id)
    positions: dict = st['positions'] or {}
    # 兼容旧格式：dict[str, dict]（每股一笔）→ dict[str, list[dict]]（每股可多笔）
    for _k, _v in list(positions.items()):
        if isinstance(_v, dict):
            positions[_k] = [_v]
    cash: float = st['cash']
    last_run: str | None = st.get('last_run_date')

    # 需要推进的日期序列（last_run 之后到今天）
    if last_run is None:
        # 无历史状态：从注册日起只记录今天净值（回填由 bootstrap 做）
        dates_to_process = [today]
    else:
        try:
            i0 = trading_dates.index(last_run) + 1
        except ValueError:
            i0 = trading_dates.index(trading_dates[0])
        dates_to_process = [d for d in trading_dates[i0:] if d <= today]
    if not dates_to_process:
        return {'status': 'skip', 'message': 'already up-to-date', 'nav': None}

    errors = []
    # open-open原子换仓：T+1开盘先卖旧仓，再用释放资金买入新仓。
    # pending_buys 持久化在策略状态中，支持跨日/跨进程执行。
    pending_buys: list[dict] = list(st.get('pending_buys') or [])
    for d in dates_to_process:
        # 1) 到期卖出（T+1 买、T+2 卖 → 每笔持仓 open_date + hold_days ≤ d 时卖出）
        sells = []
        for sym, legs in list(positions.items()):
            kept = []
            for pos in legs:
                open_date = pos.get('open_date')
                hold_td = 0
                if open_date:
                    try:
                        i_open = trading_dates.index(open_date)
                        i_now = trading_dates.index(d)
                        hold_td = i_now - i_open
                    except ValueError:
                        hold_td = 99
                if hold_td < spec.hold_days:
                    kept.append(pos)   # 还没到该笔卖出日
                    continue
                # 到卖出日：按策略指定价格卖出（默认 T+2 收盘；可用 sell_price='open' 改成开盘卖）
                df = data.get(sym)
                if df is None or not _has(df, d):
                    kept.append(pos)
                    continue
                px = _px(df, d, spec.sell_price)
                if px <= 0:
                    kept.append(pos)
                    continue
                qty = pos['qty']
                gross = px * qty
                cost = _sell_cost(px, qty)
                proceeds = gross - cost
                cash += proceeds
                sells.append((sym, px, qty, cost, pos))
            if kept:
                positions[sym] = kept
            else:
                positions.pop(sym, None)

        for sym, px, qty, cost, pos in sells:
            pnl = (px - pos['avg_cost']) * qty - cost - pos.get('buy_cost', 0)
            state.record_trade(spec.strategy_id, symbol=sym, signal_date=pos.get('signal_date'),
                               buy_date=pos.get('open_date'), sell_date=d,
                               buy_price=pos['avg_cost'], sell_price=px, quantity=qty,
                               buy_cost=pos.get('buy_cost'), sell_cost=cost,
                               pnl=pnl, return_pct=(px - pos['avg_cost']) / pos['avg_cost'],
                               direction='long', exit_reason='hold_expire')

        # 处理上一交易日生成的开盘买入计划（先卖旧仓释放资金，再买新仓）
        remaining_pending: list[dict] = []
        for order in pending_buys:
            sym = order['symbol']
            px = float(order['price'])
            qty = int(order['qty'])
            cost = float(order['cost'])
            signal_date = order.get('signal_date')
            buy_date = order.get('buy_date')
            # 只在计划买入日成交；过期挂单不使用旧价格成交。
            if buy_date and d < buy_date:
                remaining_pending.append(order)
                continue
            if buy_date and d > buy_date:
                continue
            if cash < px * qty + cost:
                qty = int((cash - cost) // (px * 100)) * 100
                if qty < 100:
                    continue
                cost = _buy_cost(px, qty)
            cash -= px * qty + cost
            positions.setdefault(sym, []).append(
                {'qty': qty, 'avg_cost': px, 'open_date': d,
                 'signal_date': signal_date, 'buy_cost': cost})
        pending_buys = remaining_pending

        # 2) 生成信号 → 明日开盘买入
        sig = _signal_on_date(spec.signal_fn, panel_df, d, (signal_cache or {}).get(spec.strategy_id), spec)
        if (
            universe is not None
            and spec.filter_signal_by_universe
            and sig is not None
            and len(sig) > 0
        ):
            pool_syms = universe.get(d, set())
            sig = sig[sig.index.get_level_values('ticker').isin(pool_syms)]
        if sig is not None and len(sig) > 0:
            sig = sig.dropna()
            if len(sig) > 0:
                # 明日 = 下一个交易日（可能尚无行情；预选仍记录, 买入待次日）
                nxt = _nearest_date(trading_dates, d, backward=False)
                # 选择目标：分数最小/最大端 candidate_pool 只候选（与回测 CANDIDATE_POOL 一致,
                # 买入时按分数顺延成交 top_k 只——涨停/停牌/资金不足的候选跳过补位）
                if spec.buy_low:
                    picks = sig.nsmallest(max(spec.candidate_pool, spec.top_k))
                else:
                    picks = sig.nlargest(max(spec.candidate_pool, spec.top_k))
                # 记录每日预选（T 日信号 → T+1 买入目标）。
                # 即便次日行情未到（如最新交易日）, 也记录, 保证"明日预选"始终可用。
                buy_date = nxt if (nxt is not None and nxt > d) else None
                try:
                    state.record_preselect(spec.strategy_id, d, buy_date,
                                           [(sym, float(score)) for sym, score in picks.items()])
                except Exception as e:
                    logger.warning('[sim] 预选记录失败 {}: {}', spec.strategy_id, e)
                if nxt is not None and nxt > d:
                    # 多空模式：D1 多 / D10 空（同一日开两腿, 用 fixed_initial 小仓位）
                    if spec.long_short == 'decile':
                        # 这里由上层专门处理（见 run_strategy_day_longshort），此处跳过长仓逻辑
                        pass
                    else:
                        # 与线下 PortfolioManager.daily_cash_split 一致：
                        # 当日可用资金均分给当日可买候选（复利滚动, 非每仓固定 position_size）
                        # 1) 先收集当日可买候选（top_k 内、明日开盘可成交）。
                        #    允许重复买入：同一股票在持有期内再次被选中也照常建新仓（独立多笔）。
                        candidates = []
                        # 明日开盘买入前，今日已到期的持仓会先在次日流程中卖出；
                        # 计算可用仓位时应排除“明日将到期卖出”的持仓，否则
                        # open-open每日换仓策略会被错误地隔日交易。
                        current_legs = 0
                        for _syms, _legs in positions.items():
                            for _pos in _legs:
                                _od = _pos.get('open_date')
                                _hold = 0
                                if _od:
                                    try:
                                        _hold = trading_dates.index(nxt) - trading_dates.index(_od)
                                    except ValueError:
                                        _hold = 99
                                if _hold < spec.hold_days:
                                    current_legs += 1
                        for sym, score in picks.items():
                            if len(candidates) >= spec.top_k:   # 成交上限（与回测 TOP_K 一致）
                                break
                            if current_legs + len(candidates) >= spec.max_positions:
                                break
                            if spec.forbid_repeat and sym in positions:
                                continue   # 禁止重复开仓：已持仓（昨日在途）股票跳过
                            df = data.get(sym)
                            if df is None or not _has(df, nxt):
                                continue
                            px = _px(df, nxt, 'open')
                            if px <= 0:
                                continue
                            # 执行过滤：T+1 开盘涨幅超过阈值时放弃（竞价追高风险）。
                            # open 是成交时点价格，prev_close 是 T 日已知价格，无未来函数。
                            if spec.buy_gap_limit is not None:
                                prev_close = _px(df, d, 'close')
                                if prev_close <= 0 or px / prev_close - 1 > spec.buy_gap_limit:
                                    continue
                            candidates.append((sym, px))
                        # 2) 仓位预算：按“目标总资产”而非“当前现金”分配。
                        #    每日先在脑内将旧仓按 spec.nav_price mark-to-market，得到总资产；
                        #    再计算现金目标 = 总资产 × cash_reserve。
                        #    这样 50% 现金保留始终是总资产的一半，不会因历史盈亏漂移。
                        first_buy = (len(positions) == 0 and not state.has_trades(spec.strategy_id))
                        scale = 1.0
                        vt = spec.params.get('vol_target')
                        if vt and not first_buy:
                            # 波动率目标：scale = min(target_vol / 过去20日净值波动, cap)
                            # （与回测 vol_target_ret 口径一致, 用已推进净值算, 天然 shift1 防未来）
                            try:
                                _navs = state.get_nav(spec.strategy_id)
                                _rets = _navs['nav'].pct_change().dropna()
                                _vol = float(_rets.rolling(20).std().iloc[-1]) if len(_rets) >= 20 else 0.0
                                if _vol > 0:
                                    # 只降不升：波动超目标缩仓, 低波动满仓（scale>1 的预算放大
                                    # 会造成现金畸形分配, 与回测收益缩放效果不等价）
                                    scale = float(np.clip(vt / _vol, 0.2, 1.0))
                            except Exception:  # noqa: BLE001
                                scale = 1.0
                        reserve = spec.cash_reserve
                        if reserve <= 0 and first_buy:
                            reserve = 0.5
                            # open-open原子换仓策略不做首日半仓，
                            # 否则研究回测与模拟盘资金口径不一致。
                            if spec.same_open_reinvest:
                                reserve = 0.0

                        # open-open日频换仓：T+1开盘先卖旧仓，释放的现金可在同一开盘价
                        # 用于新仓买入。此口径对应理论回测的原子换仓假设。
                        # 注意：释放资金只用于“计划新买入”，不能改变现金科目；
                        # 实际现金要等次日卖出后才入账，否则会变成杠杆。
                        if spec.same_open_reinvest:
                            release_cash = 0.0
                            for _sym, _legs in positions.items():
                                _df = data.get(_sym)
                                if _df is None or not _has(_df, nxt):
                                    continue
                                for _pos in _legs:
                                    _od = _pos.get('open_date')
                                    if not _od:
                                        continue
                                    try:
                                        _hold = trading_dates.index(nxt) - trading_dates.index(_od)
                                    except ValueError:
                                        continue
                                    if _hold < spec.hold_days:
                                        continue
                                    try:
                                        _sell_px = _px(_df, nxt, spec.sell_price)
                                    except KeyError:
                                        continue
                                    if _sell_px <= 0:
                                        continue
                                    release_cash += (
                                        _sell_px * _pos['qty']
                                        - _sell_cost(_sell_px, _pos['qty'])
                                    )
                        else:
                            release_cash = 0.0

                        if reserve > 0:
                            mark_mv = 0.0
                            for _sym, _legs in positions.items():
                                _df = data.get(_sym)
                                if _df is None:
                                    continue
                                _qty = sum(p['qty'] for p in _legs)
                                try:
                                    mark_mv += _qty * _mark_px(_df, d, spec.nav_price)
                                except ValueError:
                                    pass
                            total_assets = cash + mark_mv
                            target_cash = total_assets * reserve
                            day_budget = max(cash + release_cash - target_cash, 0.0) * scale
                        else:
                            day_budget = (cash + release_cash) * scale
                        if candidates and day_budget > 0:
                            per = day_budget / len(candidates)
                            for sym, px in candidates:
                                qty = int(per // (px * 100)) * 100
                                if qty < 100:
                                    continue
                                cost = _buy_cost(px, qty)
                                available = cash + release_cash
                                if available < px * qty + cost:
                                    qty = int((available - cost) // (px * 100)) * 100
                                    if qty < 100:
                                        continue
                                # 只记录计划买入；现金在 nxt 实际成交时再扣。
                                # 旧仓也在 nxt 卖出，释放资金后自然轧差。
                                pending_buys.append({
                                    'symbol': sym, 'price': float(px), 'qty': int(qty),
                                    'cost': float(cost), 'signal_date': d, 'buy_date': nxt,
                                })
        # 3) 记录今日净值
        mv = 0.0
        for sym, legs in positions.items():
            df = data.get(sym)
            if df is None:
                continue
            total_qty = sum(p['qty'] for p in legs)
            try:
                mv += total_qty * _mark_px(df, d, spec.nav_price)
            except ValueError:
                continue
        nav = (cash + mv) / spec.initial_capital
        state.append_nav(spec.strategy_id, d, nav, cash, mv)
        state.update_strategy_state_with_pending(
            spec.strategy_id, positions, cash, d, pending_buys)

    return {'status': 'ok', 'message': '; '.join(errors) if errors else 'ok',
            'nav': nav, 'positions': len(positions), 'cash': cash}


def materialize_pending_orders_from_preselect(
    spec: StrategySpec,
    account_id: str,
    data: dict[str, pd.DataFrame],
    signal_date: str,
    execution_date: str,
) -> list[dict]:
    """把已保存的 T 日预选物化成 T+1 开盘订单。

    正常路径下，run_strategy_day_persistent 在 T 日就会写入 pending_buys。
    但如果 T 日收盘后本地行情还没有 T+1 交易日（例如 T 为最新数据日），
    引擎无法用 T+1 open 计算订单，只能先保存预选。等 T+1 行情落库后，
    账户从 T 推进到 T+1 时，用本函数补生成订单，再交给原执行流程成交。

    该函数只补 pending_buys，不直接成交、不写净值。
    """
    st = state.get_strategy(account_id)
    if st is None:
        return []

    pending_orders: list[dict] = list(st.get('pending_buys') or [])
    if any(o.get('buy_date') == execution_date for o in pending_orders):
        return pending_orders

    positions: dict = st['positions'] or {}
    cash = float(st['cash'])
    rows = state.get_preselect(account_id, signal_date=signal_date, limit=1000)
    picks = [(str(r['symbol']), float(r['score'] or 0)) for r in rows]

    # 低频策略非信号日没有预选时应保持持仓；日频/择时策略信号为空则目标为空。
    if not picks and spec.signal_dates is not None and signal_date not in spec.signal_dates:
        return pending_orders

    target_symbols: set[str] = set()
    for sym, _score in picks:
        if len(target_symbols) >= spec.top_k:
            break
        df = data.get(sym)
        if df is None or not _has(df, execution_date):
            continue
        px = _px(df, execution_date, 'open')
        if px <= 0:
            continue
        if spec.buy_gap_limit is not None:
            prev_close = _px(df, signal_date, 'close')
            if prev_close <= 0 or px / prev_close - 1 > spec.buy_gap_limit:
                continue
        target_symbols.add(sym)

    held_symbols = set(positions.keys())
    exit_symbols = held_symbols - target_symbols
    entry_symbols = target_symbols - held_symbols

    for sym in exit_symbols:
        legs = positions.get(sym, [])
        if not legs:
            continue
        df = data.get(sym)
        if df is None or not _has(df, execution_date):
            continue
        px = _px(df, execution_date, spec.sell_price)
        if px <= 0:
            continue
        leg = legs[0]
        if any(o.get('side') == 'sell' and o.get('symbol') == sym
               and o.get('buy_date') == execution_date for o in pending_orders):
            continue
        pending_orders.append({
            'side': 'sell', 'symbol': sym, 'qty': int(leg['qty']),
            'price': float(px), 'cost': float(_sell_cost(px, leg['qty'])),
            'signal_date': signal_date, 'buy_date': execution_date,
        })

    release_cash = sum(
        float(o['price']) * int(o['qty']) - float(o.get('cost', 0))
        for o in pending_orders
        if o.get('side') == 'sell' and o.get('buy_date') == execution_date
    )

    reserve = spec.cash_reserve
    if reserve > 0:
        mark_mv = 0.0
        for sym, legs in positions.items():
            df = data.get(sym)
            if df is None:
                continue
            qty = sum(p['qty'] for p in legs)
            try:
                mark_mv += qty * _mark_px(df, signal_date, spec.nav_price)
            except ValueError:
                pass
        total_assets = cash + mark_mv
        target_cash = total_assets * reserve
        budget = max(cash + release_cash - target_cash, 0.0)
    else:
        budget = cash + release_cash

    if entry_symbols and budget > 0:
        per = budget / len(entry_symbols)
        for sym in entry_symbols:
            if sym in positions:
                continue
            if any(o.get('side') == 'buy' and o.get('symbol') == sym
                   and o.get('buy_date') == execution_date for o in pending_orders):
                continue
            df = data.get(sym)
            if df is None or not _has(df, execution_date):
                continue
            px = _px(df, execution_date, 'open')
            if px <= 0:
                continue
            qty = int(per // (px * 100)) * 100
            if qty < 100:
                continue
            cost = _buy_cost(px, qty)
            if budget < px * qty + cost:
                qty = int((budget - cost) // (px * 100)) * 100
                if qty < 100:
                    continue
                cost = _buy_cost(px, qty)
            budget -= px * qty + cost
            pending_orders.append({
                'side': 'buy', 'symbol': sym, 'price': float(px),
                'qty': int(qty), 'cost': float(cost),
                'signal_date': signal_date, 'buy_date': execution_date,
            })

    if pending_orders != list(st.get('pending_buys') or []):
        state.update_strategy_state_with_pending(
            account_id,
            positions,
            cash,
            st.get('last_run_date'),
            pending_orders,
            extra_state=st.get('extra_state') or {},
        )
    return pending_orders


def run_strategy_day_persistent(spec: StrategySpec, panel_df: pd.DataFrame,
                                data: dict[str, pd.DataFrame],
                                trading_dates: list[str], today: str,
                                signal_cache: dict | None = None,
                                universe: dict | None = None) -> dict:
    """持续持仓模式：只在目标组合成员变化时交易。

    研究回测中的换手率是“新增持仓数 / 当前持仓数”，
    隐含假设同一股票连续入选时不重复买卖。
    此函数按该口径执行：
      - T日生成目标topK；
      - T+1开盘卖出不在目标中的旧仓；
      - T+1开盘买入新进入目标的股票；
      - 连续入选的持仓不交易。
    """
    st = state.get_strategy(spec.strategy_id)
    if st is None:
        state.register_strategy(spec.strategy_id, spec.name, spec.description,
                                spec.initial_capital, spec.pool, {}, spec.initial_capital)
        st = state.get_strategy(spec.strategy_id)

    positions: dict = st['positions'] or {}
    for _k, _v in list(positions.items()):
        if isinstance(_v, dict):
            positions[_k] = [_v]
    cash: float = st['cash']
    pending_orders: list[dict] = list(st.get('pending_buys') or [])

    last_run = st.get('last_run_date')
    if last_run is None:
        dates_to_process = [today]
    else:
        try:
            i0 = trading_dates.index(last_run) + 1
        except ValueError:
            i0 = 0
        dates_to_process = [d for d in trading_dates[i0:] if d <= today]
    if not dates_to_process:
        return {'status': 'skip', 'message': 'already up-to-date', 'nav': None}

    # 最新信号日运行时还没有次日行情，无法生成开盘订单。下一次增量
    # 推进时，所有持续持仓策略都应从该日保存的预选补生成订单。
    # 仅处理紧邻的上一交易日，避免把缺失的历史信号当作清仓信号。
    first_day = dates_to_process[0]
    first_idx = trading_dates.index(first_day)
    if last_run is not None and first_idx > 0 and trading_dates[first_idx - 1] == last_run:
        if not any(o.get('buy_date') == first_day for o in pending_orders):
            pending_orders = materialize_pending_orders_from_preselect(
                spec, spec.strategy_id, data, last_run, first_day)

    nav = None
    extra_state = st.get('extra_state') or {}
    has_overlay = bool(spec.hedge_overlay)
    if has_overlay:
        nav_hist = state.get_nav(spec.strategy_id)
        prior_stock_nav = float(extra_state.get('prior_stock_nav', 1.0))
        prior_full_nav = float(extra_state.get('prior_full_nav',
                                               nav_hist['nav'].iloc[-1] if len(nav_hist) else 1.0))
    else:
        prior_stock_nav = 1.0
        prior_full_nav = 1.0
    for d in dates_to_process:
        # 1) 执行上一信号日生成的开盘订单：先卖后买
        sells_due = [o for o in pending_orders if o.get('side') == 'sell' and o.get('buy_date') == d]
        buys_due = [o for o in pending_orders if o.get('side') == 'buy' and o.get('buy_date') == d]

        for order in sells_due:
            sym = order['symbol']
            qty = int(order['qty'])
            df = data.get(sym)
            if df is None or not _has(df, d):
                continue
            px = _px(df, d, spec.sell_price)
            if px <= 0:
                continue
            legs = positions.get(sym, [])
            if not legs:
                continue
            # 持续持仓模式下通常一股一腿；若有多腿则卖出第一笔匹配数量
            leg = legs[0]
            sell_qty = min(qty, leg['qty'])
            cost = _sell_cost(px, sell_qty)
            cash += px * sell_qty - cost
            pnl = (px - leg['avg_cost']) * sell_qty - cost - leg.get('buy_cost', 0)
            state.record_trade(
                spec.strategy_id, symbol=sym, signal_date=leg.get('signal_date'),
                buy_date=leg.get('open_date'), sell_date=d,
                buy_price=leg['avg_cost'], sell_price=px, quantity=sell_qty,
                buy_cost=leg.get('buy_cost'), sell_cost=cost, pnl=pnl,
                return_pct=(px - leg['avg_cost']) / leg['avg_cost'],
                direction='long', exit_reason='target_exit')
            remaining_qty = leg['qty'] - sell_qty
            if remaining_qty > 0:
                legs[0] = {**leg, 'qty': remaining_qty}
            else:
                legs.pop(0)
            if legs:
                positions[sym] = legs
            else:
                positions.pop(sym, None)

        for order in buys_due:
            sym = order['symbol']
            px = float(order['price'])
            qty = int(order['qty'])
            cost = float(order['cost'])
            if cash < px * qty + cost:
                qty = int((cash - cost) // (px * 100)) * 100
                if qty < 100:
                    continue
                cost = _buy_cost(px, qty)
            cash -= px * qty + cost
            positions.setdefault(sym, []).append({
                'qty': qty, 'avg_cost': px, 'open_date': d,
                'signal_date': order.get('signal_date'), 'buy_cost': cost})
            state.record_trade(
                spec.strategy_id, symbol=sym,
                signal_date=order.get('signal_date'), buy_date=d,
                sell_date=None, buy_price=px, sell_price=None,
                quantity=qty, buy_cost=cost, sell_cost=None,
                pnl=0.0, return_pct=0.0, direction='long',
                exit_reason='open')

        pending_orders = [o for o in pending_orders if o.get('buy_date') != d]

        # 2) 生成T日信号，确定T+1目标组合。
        #    低频策略显式给出signal_dates：非信号日保持当前目标组合；
        #    信号日若信号为空，则允许目标组合清空（与研究回测一致）。
        target_symbols: set[str] = set()
        if spec.signal_dates is not None and d not in spec.signal_dates:
            sig = None
            target_symbols = set(positions.keys())
        else:
            sig = _signal_on_date(spec.signal_fn, panel_df, d,
                                  (signal_cache or {}).get(spec.strategy_id), spec)
            if (
                universe is not None
                and spec.filter_signal_by_universe
                and sig is not None
                and len(sig) > 0
            ):
                pool_syms = universe.get(d, set())
                sig = sig[sig.index.get_level_values('ticker').isin(pool_syms)]

        nxt = _nearest_date(trading_dates, d, backward=False)
        if sig is not None and len(sig) > 0:
            sig = sig.dropna()
            if len(sig) > 0:
                if spec.buy_low:
                    picks = sig.nsmallest(max(spec.candidate_pool, spec.top_k))
                else:
                    picks = sig.nlargest(max(spec.candidate_pool, spec.top_k))
                try:
                    state.record_preselect(
                        spec.strategy_id, d, nxt,
                        [(sym, float(score)) for sym, score in picks.items()])
                except Exception as e:
                    logger.warning('[sim] 预选记录失败 {}: {}', spec.strategy_id, e)

                if nxt is not None and nxt > d:
                    for sym, _score in picks.items():
                        if len(target_symbols) >= spec.top_k:
                            break
                        df = data.get(sym)
                        if df is None or not _has(df, nxt):
                            continue
                        px = _px(df, nxt, 'open')
                        if px <= 0:
                            continue
                        if spec.buy_gap_limit is not None:
                            prev_close = _px(df, d, 'close')
                            if prev_close <= 0 or px / prev_close - 1 > spec.buy_gap_limit:
                                continue
                        target_symbols.add(sym)

        # 3) 生成T+1开盘调仓订单
        if nxt is not None and nxt > d:
            held_symbols = set(positions.keys())
            exit_symbols = held_symbols - target_symbols
            entry_symbols = target_symbols - held_symbols

            for sym in exit_symbols:
                legs = positions.get(sym, [])
                if not legs:
                    continue
                df = data.get(sym)
                if df is None or not _has(df, nxt):
                    continue
                px = _px(df, nxt, spec.sell_price)
                if px <= 0:
                    continue
                leg = legs[0]
                pending_orders.append({
                    'side': 'sell', 'symbol': sym, 'qty': int(leg['qty']),
                    'price': float(px), 'cost': float(_sell_cost(px, leg['qty'])),
                    'signal_date': d, 'buy_date': nxt,
                })

            release_cash = 0.0
            for order in pending_orders:
                if order.get('side') != 'sell' or order.get('buy_date') != nxt:
                    continue
                release_cash += order['price'] * order['qty'] - order['cost']

            reserve = spec.cash_reserve
            if reserve > 0:
                mark_mv = 0.0
                for sym, legs in positions.items():
                    df = data.get(sym)
                    if df is None:
                        continue
                    qty = sum(p['qty'] for p in legs)
                    try:
                        mark_mv += qty * _mark_px(df, d, spec.nav_price)
                    except ValueError:
                        pass
                total_assets = cash + mark_mv
                target_cash = total_assets * reserve
                budget = max(cash + release_cash - target_cash, 0.0)
            else:
                budget = cash + release_cash

            if entry_symbols and budget > 0:
                logger.info('[sim] {} | {} | entry_symbols={} held={} budget={}',
                            spec.strategy_id, d, len(entry_symbols), len(positions), budget)
                per = budget / len(entry_symbols)
                for sym in entry_symbols:
                    # 防止重放时重复买入：已有持仓或待成交订单则跳过
                    if sym in positions:
                        continue
                    if any(o.get('side') == 'buy' and o.get('symbol') == sym for o in pending_orders):
                        continue
                    df = data.get(sym)
                    if df is None or not _has(df, nxt):
                        continue
                    px = _px(df, nxt, 'open')
                    if px <= 0:
                        continue
                    qty = int(per // (px * 100)) * 100
                    if qty < 100:
                        continue
                    cost = _buy_cost(px, qty)
                    if budget < px * qty + cost:
                        qty = int((budget - cost) // (px * 100)) * 100
                        if qty < 100:
                            continue
                        cost = _buy_cost(px, qty)
                    budget -= px * qty + cost
                    pending_orders.append({
                        'side': 'buy', 'symbol': sym, 'price': float(px),
                        'qty': int(qty), 'cost': float(cost),
                        'signal_date': d, 'buy_date': nxt,
                    })

        # 4) 记录今日净值；可选对冲overlay直接乘在股票腿净值上。
        # overlay已包含对冲成本，不改变现金/持仓，只影响组合收益。
        mv = 0.0
        for sym, legs in positions.items():
            df = data.get(sym)
            if df is None:
                continue
            total_qty = sum(p['qty'] for p in legs)
            try:
                mv += total_qty * _mark_px(df, d, spec.nav_price)
            except ValueError:
                continue
        stock_nav = (cash + mv) / spec.initial_capital
        if has_overlay:
            hedge_ret = spec.hedge_overlay.get(d, 0.0)
            stock_return = stock_nav / prior_stock_nav if prior_stock_nav else 1.0
            full_return = stock_return * (1.0 + hedge_ret) - 1.0
            nav = prior_full_nav * (1.0 + full_return)
            prior_stock_nav = stock_nav
            prior_full_nav = nav
        else:
            nav = stock_nav
        state.append_nav(spec.strategy_id, d, nav, cash, mv)
        state.update_strategy_state_with_pending(
            spec.strategy_id, positions, cash, d, pending_orders,
            extra_state={
                'prior_stock_nav': prior_stock_nav,
                'prior_full_nav': prior_full_nav,
            })

    return {'status': 'ok', 'message': 'ok', 'nav': nav,
            'positions': len(positions), 'cash': cash}


def run_strategy_day_longshort(spec: StrategySpec, panel_df: pd.DataFrame,
                               data: dict[str, pd.DataFrame],
                               trading_dates: list[str], today: str,
                               signal_cache: dict | None = None,
                               universe: dict | None = None) -> dict:
    """
    多空模式：D1 多 / D10 空 各半仓（fixed_initial 5‰×N）。
    每日：T 日收盘取 D1/D10 → T+1 开盘开仓 → T+2 收盘平仓（与回测 run_longshort 一致）。
    净值 = 1 + Σ 0.5×(r_D1 − r_D10)，按日累计。
    """
    st = state.get_strategy(spec.strategy_id)
    if st is None:
        state.register_strategy(spec.strategy_id, spec.name, spec.description,
                                spec.initial_capital, spec.pool, {}, spec.initial_capital)
        st = state.get_strategy(spec.strategy_id)

    last_run = st.get('last_run_date')
    if last_run is None:
        dates_to_process = [today]
    else:
        try:
            i0 = trading_dates.index(last_run) + 1
        except ValueError:
            i0 = 0
        dates_to_process = [d for d in trading_dates[i0:] if d <= today]
    if not dates_to_process:
        return {'status': 'skip', 'message': 'already up-to-date', 'nav': None}

    nav_hist = state.get_nav(spec.strategy_id)
    last_nav = float(nav_hist['nav'].iloc[-1]) if len(nav_hist) else 1.0

    # 在途多空腿：{open_date: {sym: {'side','px','qty'}}}
    # 持久化在 strategy 状态的 positions 字段（多空模式复用该字段）
    open_legs: dict[str, dict] = st['positions'] or {}

    for d in dates_to_process:
        # 1) 平今日到期的腿（open_date 后第 2 个交易日 → T+2 收盘平）
        day_pnl_long = 0.0
        day_pnl_short = 0.0
        due_legs = []
        for od, legs in open_legs.items():
            try:
                i_open = trading_dates.index(od)
            except ValueError:
                continue
            i_now = trading_dates.index(d)
            if i_now - i_open >= 2:   # 持有 2 个交易日（T+1 开、T+2 平）
                due_legs.extend((sym, leg, od) for sym, leg in legs.items())
        for sym, leg, od in due_legs:
            df = data.get(sym)
            if df is not None and _has(df, d):
                sp = _px(df, d, 'close')
                # 多空腿成本：买卖双向佣金+滑点, 卖出加印花税（与 CostModel 一致）
                round_trip_cost = 2 * COMMISSION_RATE + STAMP_TAX + 2 * SLIPPAGE_RATE
                if leg['side'] == 'long':
                    ret = (sp - leg['px']) / leg['px'] - round_trip_cost
                    day_pnl_long += ret * leg['px'] * leg['qty']
                else:
                    ret = (leg['px'] - sp) / leg['px'] - round_trip_cost
                    day_pnl_short += ret * leg['px'] * leg['qty']
                state.record_trade(spec.strategy_id, symbol=sym, signal_date=leg.get('signal_date'),
                                   buy_date=od, sell_date=d,
                                   buy_price=leg['px'], sell_price=sp, quantity=leg['qty'],
                                   buy_cost=leg['px']*leg['qty']*COMMISSION_RATE,
                                   sell_cost=sp*leg['qty']*(COMMISSION_RATE+STAMP_TAX),
                                   pnl=ret*leg['px']*leg['qty'], return_pct=ret,
                                   direction=leg['side'], exit_reason='ls_fixed')
            open_legs[od].pop(sym)
        for od in [od for od in open_legs if not open_legs[od]]:
            open_legs.pop(od)

        # 2) 用今日信号开新腿（明日开盘成交）
        sig = _signal_on_date(spec.signal_fn, panel_df, d, (signal_cache or {}).get(spec.strategy_id), spec)
        if universe is not None and sig is not None and len(sig) > 0:
            pool_syms = universe.get(d, set())
            sig = sig[sig.index.get_level_values('ticker').isin(pool_syms)]
        nxt = _nearest_date(trading_dates, d, backward=False)
        if sig is not None and len(sig) > 0 and nxt is not None and nxt > d:
            sig = sig.dropna()
            if len(sig) >= 20:
                th = spec.ls_decile
                q = sig.rank(pct=True)
                long_picks = q[q <= th].index.tolist()
                short_picks = q[q >= 1 - th].index.tolist()
                new_legs = {}
                for sym in long_picks + short_picks:
                    df = data.get(sym)
                    if df is None or not _has(df, nxt):
                        continue
                    px = _px(df, nxt, 'open')
                    if px <= 0:
                        continue
                    budget = spec.initial_capital * 0.005
                    qty = int(budget // (px * 100)) * 100
                    if qty < 100:
                        continue
                    side = 'long' if sym in long_picks else 'short'
                    new_legs[sym] = {'side': side, 'px': px, 'qty': qty, 'signal_date': d}
                if new_legs:
                    open_legs[nxt] = new_legs

        # 3) 记录今日净值：与回测 verify 一致——腿组合市值变动 / 总初始资金
        #    r_long = (已实现腿PnL + 在途腿市值变动) / initial_capital
        #    （verify 的 r1 = nav1.pct_change(), nav1=组合市值/初始资金, 即 Σ腿PnL/initial）
        mv_long = 0.0
        mv_short = 0.0
        for od, legs in open_legs.items():
            for sym, leg in legs.items():
                df = data.get(sym)
                if df is None or not _has(df, d):
                    continue
                cp = _px(df, d, 'close')
                if leg['side'] == 'long':
                    mv_long += (cp - leg['px']) * leg['qty']
                else:
                    mv_short += (leg['px'] - cp) * leg['qty']
        r_long = (day_pnl_long + mv_long) / spec.initial_capital
        r_short = (day_pnl_short + mv_short) / spec.initial_capital
        last_nav = last_nav + 0.5 * (r_long - r_short)
        state.append_nav(spec.strategy_id, d, last_nav, last_nav, 0.0)
        state.update_strategy_state(spec.strategy_id, open_legs, last_nav, d)

    return {'status': 'ok', 'message': 'ok', 'nav': last_nav, 'positions': 0, 'cash': last_nav}


def update_all(panel_df: pd.DataFrame, data: dict[str, pd.DataFrame],
               trading_dates: list[str], today: str,
               signal_cache: dict | None = None,
               universe: dict | None = None) -> list[dict]:
    """推进所有已注册策略到 today。
    universe : {pool_name: {date_str: set(symbols)}}，按 spec.pool 选择；None=不过滤。"""
    results = []
    for spec in all_specs():
        try:
            uni = (universe or {}).get(spec.pool)
            if spec.persistent_selection:
                r = run_strategy_day_persistent(spec, panel_df, data, trading_dates, today, signal_cache, uni)
            elif spec.long_short == 'decile':
                r = run_strategy_day_longshort(spec, panel_df, data, trading_dates, today, signal_cache, uni)
            else:
                r = run_strategy_day(spec, panel_df, data, trading_dates, today, signal_cache, uni)
            state.log_run(spec.strategy_id, today, r.get('status', 'ok'), r.get('message', ''))
            results.append({'strategy_id': spec.strategy_id, 'status': r.get('status'),
                            'message': r.get('message'), 'nav': r.get('nav')})
            logger.info('[sim] {} | {} | nav={}', spec.strategy_id, today, r.get('nav'))
        except Exception as e:
            import traceback
            traceback.print_exc()
            state.log_run(spec.strategy_id, today, 'error', str(e))
            results.append({'strategy_id': spec.strategy_id, 'status': 'error', 'message': str(e)})
    return results

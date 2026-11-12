"""
交易层：TradeSimulator。

职责：接收信号列表 + 出场规则 → 模拟交易 → 输出 Trade 列表。

完全不知道信号怎么来的（选股逻辑），也不计算评测指标（交给
PerformanceAnalyzer）。只做一件事：给定买入信号和出场规则，
模拟出每笔交易的实际买卖过程。

两种出场模式：
  1. simulate_next_day  — 次日限价单出场（T+1，旧模式）
  2. simulate_with_stop — 多日止损止盈出场（新模式）

用法：
    sim = TradeSimulator(gen.data)
    trades = sim.simulate_with_stop(signals, stop_method='atr',
                                     stop_params={'n_atr': 2.0},
                                     rr_ratio=2.0, max_hold_days=10)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger

from strategies.cost_model import CostModel
from strategies.stops import STOP_METHODS, calc_target
from strategies.types import Trade


class TradeSimulator:
    """交易模拟器：信号 + 出场规则 → Trade 列表。"""

    def __init__(self, data: dict[str, pd.DataFrame], cost: CostModel = None):
        """
        参数
        ----
        data : {symbol: DataFrame}，由 SignalGenerator.load_data() 产生。
               多日模拟时需要按 symbol 查 K 线逐日遍历。
        cost : 交易成本模型，默认使用 CostModel() 统一参数
        """
        self._data = data
        self._cost = cost or CostModel()

    # ------------------------------------------------------------------
    # 模式 1：次日限价单出场（T+1）
    # ------------------------------------------------------------------

    def simulate_next_day(
        self,
        signals: pd.DataFrame,
        sell_pct: float,
        open_stop_loss_pct: float = None,
    ) -> list[Trade]:
        """
        次日限价单模拟（旧模式）。

        成交逻辑（按优先级）：
          target     = buy_price × (1 + sell_pct)
          stop_price = buy_price × (1 - open_stop_loss_pct)  [若设置了止损]

          1. 次日开盘 >= target      → 以开盘价成交
          2. 次日开盘 <= stop_price  → 以开盘价止损
          3. 次日最高 >= target      → 以 target 成交
          4. 其余                    → 以收盘价成交
        """
        if signals.empty:
            return []

        buy_prices = signals["buy_price"].values
        targets = buy_prices * (1 + sell_pct)
        next_opens = signals["next_open"].values
        next_highs = signals["next_high"].values
        next_closes = signals["next_close"].values

        if open_stop_loss_pct is not None:
            stop_prices = buy_prices * (1 - open_stop_loss_pct)
            sell_prices = np.where(
                next_opens >= targets, next_opens,
                np.where(
                    next_opens <= stop_prices, next_opens,
                    np.where(
                        next_highs >= targets, targets, next_closes,
                    )
                )
            )
        else:
            sell_prices = np.where(
                next_opens >= targets, next_opens,
                np.where(
                    next_highs >= targets, targets, next_closes,
                )
            )
        filled = next_highs >= targets

        gross_return = (sell_prices - buy_prices) / buy_prices
        cost = self._total_cost()

        trades = []
        for i, row in enumerate(signals.itertuples()):
            trades.append(Trade(
                symbol=row.symbol,
                signal_date=str(row.signal_date.date()),
                buy_date=str(row.signal_date.date()),
                sell_date=str(row.next_date.date()),
                buy_price=float(buy_prices[i]),
                sell_price=float(sell_prices[i]),
                return_pct=float(gross_return[i]),
                net_return_pct=float(gross_return[i] - cost),
                sell_pct=sell_pct,
                filled=bool(filled[i]),
                hold_days=0,
            ))
        return trades

    # ------------------------------------------------------------------
    # 模式 3：开盘买入 → 次日开盘卖出（持仓1天）
    # ------------------------------------------------------------------

    def simulate_next_open(self, signals: pd.DataFrame) -> list[Trade]:
        """
        开盘买入，次日开盘卖出。持仓固定1天。

        买入价 = signals['buy_price']（信号日开盘价，由 buy_price_field='open' 生成）
        卖出价 = signals['next_open']（次日开盘价）

        无止损止盈，不做参数扫描。
        """
        if signals.empty:
            return []

        buy_prices = signals["buy_price"].values
        sell_prices = signals["next_open"].values
        gross_return = (sell_prices - buy_prices) / buy_prices
        cost = self._total_cost()

        trades = []
        for i, row in enumerate(signals.itertuples()):
            trades.append(Trade(
                symbol=row.symbol,
                signal_date=str(row.signal_date.date()),
                buy_date=str(row.signal_date.date()),
                sell_date=str(row.next_date.date()),
                buy_price=float(buy_prices[i]),
                sell_price=float(sell_prices[i]),
                return_pct=float(gross_return[i]),
                net_return_pct=float(gross_return[i] - cost),
                hold_days=1,
                exit_reason="open_to_open",
            ))
        return trades

    # ------------------------------------------------------------------
    # 模式 2b：卖出信号出场（基于技术面的主动出场）
    # ------------------------------------------------------------------

    def simulate_with_exit_signal(
        self,
        signals: pd.DataFrame,
        exit_fn: Callable,
        exit_params: dict = None,
        max_hold_days: int = 20,
        stop_price_col: str = None,
    ) -> list[Trade]:
        """
        基于卖出信号的多日持仓模拟。

        每日检查 exit_fn 是否触发卖出信号，同时设最大持仓天数作为兜底。

        参数
        ----
        exit_fn        : 卖出信号函数，签名 fn(df, **params) -> bool Series
        exit_params    : 卖出信号参数
        max_hold_days  : 最大持仓天数（时间止损兜底）
        stop_price_col : 若提供，用该列值作为止损价（如 'low'），
                         否则不设固定止损
        """
        if signals.empty:
            return []
        exit_params = exit_params or {}
        cost = self._total_cost()
        trades = []

        for row in signals.itertuples():
            symbol = row.symbol
            df = self._data.get(symbol)
            if df is None:
                continue

            try:
                sig_loc = df.index.get_loc(row.signal_date)
            except KeyError:
                continue

            buy_loc = sig_loc + 1
            if buy_loc >= len(df):
                continue

            buy_date = df.index[buy_loc]
            buy_open = float(df.iloc[buy_loc]["open"])
            if buy_open <= 0:
                continue

            # 逐日检查出场
            sell_date = buy_date
            sell_price = buy_open
            exit_reason = "time"
            actual_hold = 1

            for i in range(max_hold_days):
                day_loc = buy_loc + i
                if day_loc >= len(df):
                    break

                bar = df.iloc[day_loc]
                bar_date = df.index[day_loc]
                day_close = float(bar["close"])

                # 检查卖出信号（用截至当日的数据）
                sub_df = df.iloc[:day_loc + 1]
                try:
                    exit_mask = exit_fn(sub_df, **exit_params)
                except Exception:
                    continue

                if exit_mask.iloc[-1]:
                    sell_date = bar_date
                    sell_price = day_close
                    actual_hold = i + 1
                    exit_reason = "signal"
                    break

            # 时间止损兜底
            if exit_reason == "time":
                day_loc = min(buy_loc + max_hold_days - 1, len(df) - 1)
                sell_date = df.index[day_loc]
                sell_price = float(df.iloc[day_loc]["close"])
                actual_hold = max_hold_days

            gross = (sell_price - buy_open) / buy_open
            trades.append(Trade(
                symbol=row.symbol,
                signal_date=str(row.signal_date.date()),
                buy_date=str(buy_date.date()),
                sell_date=str(sell_date.date()),
                buy_price=buy_open,
                sell_price=sell_price,
                return_pct=gross,
                net_return_pct=gross - cost,
                hold_days=actual_hold,
                exit_reason=exit_reason,
            ))

        return trades

    def simulate_limit_up_hold_until_break(
        self,
        signals: pd.DataFrame,
        limit_up_pct: float = 0.099,
    ) -> list[Trade]:
        """
        断板反包专用出场：若 T+1 封板，则一直持有到再次断板。

        规则
        ----
        1. 信号日收盘买入（buy_price 已由 SignalGenerator 写入）
        2. 若 T+1 收盘涨幅 < limit_up_pct：T+1 收盘卖出
        3. 若 T+1 收盘涨幅 >= limit_up_pct：继续持有
        4. 后续第一个“非涨停收盘日”（日涨幅 < limit_up_pct）视为再次断板，
           以该日收盘价卖出
        5. 若直到样本结束仍未再次断板，则以最后一个交易日收盘价卖出

        跌停卖出约束（用户新增规则）
        --------------------------
        若计划卖出日遇到跌停：
          - 一字跌停：视为无法卖出，继续等待；
            直到首次“不再跌停”的交易日，以该日收盘价卖出
          - 带量跌停：视为盘中可在跌停价成交，
            以当日跌停价卖出

        说明
        ----
        仅用日线 OHLCV 近似“封板/断板”：
          - 封板：当日收盘涨幅 >= limit_up_pct
          - 断板：当日收盘涨幅 < limit_up_pct
          - 一字跌停：open=high=low=close，且收盘跌幅 <= -limit_up_pct
          - 带量跌停：收盘跌幅 <= -limit_up_pct，但不是一字板

        这比真实盘中“炸板”定义更粗，但适合当前数据库条件。
        """
        if signals.empty:
            return []

        trades = []
        cost = self._total_cost()

        for row in signals.itertuples():
            symbol = row.symbol
            df = self._data.get(symbol)
            if df is None:
                continue

            try:
                buy_loc = df.index.get_loc(row.signal_date)
            except KeyError:
                continue

            if buy_loc + 1 >= len(df):
                continue

            buy_date = df.index[buy_loc]
            buy_price = float(row.buy_price)
            if buy_price <= 0:
                continue

            sell_loc = buy_loc + 1
            exit_reason = "t1_no_limit"

            while sell_loc < len(df):
                prev_close = float(df.iloc[sell_loc - 1]["close"])
                day_close = float(df.iloc[sell_loc]["close"])
                if prev_close <= 0 or day_close <= 0:
                    break

                day_ret = (day_close - prev_close) / prev_close

                if sell_loc == buy_loc + 1:
                    if day_ret < limit_up_pct:
                        exit_reason = "t1_no_limit"
                        break
                    exit_reason = "break_limit"
                    sell_loc += 1
                    continue

                if day_ret < limit_up_pct:
                    exit_reason = "break_limit"
                    break

                sell_loc += 1

            if sell_loc >= len(df):
                sell_loc = len(df) - 1
                exit_reason = "end_of_data"

            actual_sell_loc, sell_price, sell_note = self._resolve_limit_down_exit(
                df=df,
                planned_sell_loc=sell_loc,
                limit_down_pct=limit_up_pct,
            )

            sell_date = df.index[actual_sell_loc]
            if sell_note != "normal_close":
                exit_reason = f"{exit_reason}_{sell_note}"

            gross = (sell_price - buy_price) / buy_price

            trades.append(Trade(
                symbol=row.symbol,
                signal_date=str(row.signal_date.date()),
                buy_date=str(buy_date.date()),
                sell_date=str(sell_date.date()),
                buy_price=buy_price,
                sell_price=sell_price,
                return_pct=gross,
                net_return_pct=gross - cost,
                hold_days=int(max(actual_sell_loc - buy_loc, 0)),
                exit_reason=exit_reason,
            ))

        return trades

    @staticmethod
    def _is_one_price_bar(bar, tol: float = 1e-6) -> bool:
        """是否一字线（OHLC 几乎相等）。"""
        o = float(bar["open"])
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])
        return (
            abs(o - h) <= tol
            and abs(h - l) <= tol
            and abs(l - c) <= tol
        )

    def _resolve_limit_down_exit(
        self,
        df: pd.DataFrame,
        planned_sell_loc: int,
        limit_down_pct: float,
    ) -> tuple[int, float, str]:
        """
        根据跌停卖出约束，求实际卖出日和卖出价。

        返回
        ----
        (actual_sell_loc, sell_price, note)

        note:
          - normal_close             正常按收盘卖出
          - touch_limit_down         带量跌停，按跌停价卖出
          - wait_after_one_word_ld   一字跌停后等待到首个非跌停日收盘卖出
          - end_of_data_locked_ld    样本结束仍被一字跌停锁死，只能记最后收盘
        """
        waited_one_word = False

        for loc in range(planned_sell_loc, len(df)):
            if loc <= 0:
                return loc, float(df.iloc[loc]["close"]), "normal_close"

            prev_close = float(df.iloc[loc - 1]["close"])
            bar = df.iloc[loc]
            close_price = float(bar["close"])

            if prev_close <= 0 or close_price <= 0:
                return loc, close_price, "normal_close"

            day_ret = (close_price - prev_close) / prev_close
            is_limit_down_close = day_ret <= -limit_down_pct

            if not is_limit_down_close:
                if waited_one_word:
                    return loc, close_price, "wait_after_one_word_ld"
                return loc, close_price, "normal_close"

            if self._is_one_price_bar(bar):
                waited_one_word = True
                continue

            # 带量跌停：按触板价（近似为跌停价）卖出
            limit_down_price = prev_close * (1 - limit_down_pct)
            return loc, float(limit_down_price), "touch_limit_down"

        # 若直到样本结束仍被一字跌停锁死，只能用最后一根收盘近似结算
        return len(df) - 1, float(df.iloc[-1]["close"]), "end_of_data_locked_ld"

    def simulate_with_stop(
        self,
        signals: pd.DataFrame,
        stop_method: str = "atr",
        stop_params: dict = None,
        rr_ratio: float = 2.0,
        max_hold_days: int = 10,
        buy_offset: int = 1,
    ) -> list[Trade]:
        """
        多日持仓模拟：动态止损止盈出场。

        参数
        ----
        buy_offset : 买入日相对信号日的偏移
            1（默认）— 信号日次日开盘买入（收盘信号的标准模式）
            0        — 信号日当天开盘买入（开盘信号模式）

        出场优先级（每日按顺序检查）：
          1. 开盘跳空跌破止损 → 以开盘价止损
          2. 开盘跳空超过止盈 → 以开盘价止盈
          3. 当日最低价触及止损 → 以止损价成交
          4. 当日最高价触及止盈 → 以止盈价成交
          5. 持仓满 max_hold_days → 以收盘价强制离场（时间止损）

        参数
        ----
        stop_method  : 'atr' | 'swing' | 'ma'
        stop_params  : 止损方法的参数（见 strategies/stops.py）
        rr_ratio     : 盈亏比（止盈距离 / 止损距离）
        max_hold_days: 最大持仓天数
        """
        if signals.empty:
            return []
        if stop_method not in STOP_METHODS:
            raise ValueError(f"stop_method 须为 {list(STOP_METHODS)}")

        stop_params = stop_params or {}
        signals = self._add_stop_prices(signals, stop_method, stop_params, rr_ratio)
        return self._simulate_trades_multiday(signals, max_hold_days, buy_offset)

    # ------------------------------------------------------------------
    # 内部：止损价计算 + 多日模拟
    # ------------------------------------------------------------------

    def _add_stop_prices(
        self,
        signals: pd.DataFrame,
        stop_method: str,
        stop_params: dict,
        rr_ratio: float,
    ) -> pd.DataFrame:
        """为每条信号计算止损价和止盈价，追加为列。"""
        stop_fn = STOP_METHODS[stop_method]
        stop_prices = []
        target_prices = []
        risk_pcts = []

        for row in signals.itertuples():
            symbol = row.symbol
            df = self._data.get(symbol)
            buy_price = float(row.buy_price)

            try:
                stop = stop_fn(df, row.signal_date, **stop_params)
            except Exception as e:
                logger.debug("止损计算失败 {} {}: {}", symbol, row.signal_date, e)
                stop = buy_price * 0.95

            stop = min(stop, buy_price * 0.999)
            target = calc_target(buy_price, stop, rr_ratio)

            stop_prices.append(stop)
            target_prices.append(target)
            risk_pcts.append((buy_price - stop) / buy_price)

        signals = signals.copy()
        signals["stop_price"] = stop_prices
        signals["target_price"] = target_prices
        signals["risk_pct"] = risk_pcts

        logger.info(
            "止损计算完成 | 平均风险敞口 {:.2%} | 平均止盈目标 {:.2%}",
            float(pd.Series(risk_pcts).mean()),
            float(((pd.Series(target_prices) - signals["buy_price"]) / signals["buy_price"]).mean()),
        )
        return signals

    def _simulate_trades_multiday(
        self,
        signals: pd.DataFrame,
        max_hold_days: int,
        buy_offset: int = 1,
    ) -> list[Trade]:
        """逐日模拟多日持仓。"""
        trades = []
        cost = self._total_cost()

        for row in signals.itertuples():
            symbol = row.symbol
            df = self._data.get(symbol)
            if df is None:
                continue

            try:
                sig_loc = df.index.get_loc(row.signal_date)
            except KeyError:
                continue

            buy_loc = sig_loc + buy_offset
            if buy_loc >= len(df):
                continue

            buy_date = df.index[buy_loc]
            buy_open = float(df.iloc[buy_loc]["open"])
            if buy_open <= 0:
                continue

            # 用实际开盘价重算止盈（保持风险金额不变）
            original_buy = float(row.buy_price)
            stop_price = float(row.stop_price)
            risk_amount = original_buy - stop_price
            target_price = buy_open + risk_amount * (
                (float(row.target_price) - original_buy) / max(original_buy - stop_price, 1e-6)
            )

            # 买入当日：先检查跳空
            if buy_open <= stop_price:
                self._append_trade(trades, row, buy_date, buy_date,
                                   buy_open, buy_open, stop_price, target_price,
                                   "gap_stop", 0, cost)
                continue
            if buy_open >= target_price:
                self._append_trade(trades, row, buy_date, buy_date,
                                   buy_open, buy_open, stop_price, target_price,
                                   "gap_target", 0, cost)
                continue

            # 逐日检查
            exit_reason = "time"
            sell_date = df.index[min(buy_loc + max_hold_days - 1, len(df) - 1)]
            sell_price = float(df.loc[sell_date, "close"])
            actual_hold = max_hold_days

            for i in range(max_hold_days):
                day_loc = buy_loc + i
                if day_loc >= len(df):
                    sell_date = df.index[day_loc - 1]
                    sell_price = float(df.iloc[day_loc - 1]["close"])
                    actual_hold = i
                    exit_reason = "time"
                    break

                bar = df.iloc[day_loc]
                day_low = float(bar["low"])
                day_high = float(bar["high"])
                day_open = float(bar["open"])
                bar_date = df.index[day_loc]

                if i > 0:
                    if day_open <= stop_price:
                        sell_date, sell_price = bar_date, day_open
                        actual_hold, exit_reason = i, "gap_stop"
                        break
                    if day_open >= target_price:
                        sell_date, sell_price = bar_date, day_open
                        actual_hold, exit_reason = i, "gap_target"
                        break

                if day_low <= stop_price:
                    sell_date, sell_price = bar_date, stop_price
                    actual_hold, exit_reason = i + 1, "stop"
                    break

                if day_high >= target_price:
                    sell_date, sell_price = bar_date, target_price
                    actual_hold, exit_reason = i + 1, "target"
                    break

            self._append_trade(trades, row, buy_date, sell_date,
                               buy_open, sell_price, stop_price, target_price,
                               exit_reason, actual_hold, cost)

        return trades

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _total_cost(self) -> float:
        """单笔交易总成本率（买卖双边，快速估算不含最低佣金）。"""
        return self._cost.round_trip_cost_rate

    @staticmethod
    def _append_trade(
        trades: list,
        row,
        buy_date,
        sell_date,
        buy_price: float,
        sell_price: float,
        stop_price: float,
        target_price: float,
        exit_reason: str,
        hold_days: int,
        cost: float,
    ) -> None:
        gross = (sell_price - buy_price) / buy_price
        trades.append(Trade(
            symbol=row.symbol,
            signal_date=str(row.signal_date.date()),
            buy_date=str(buy_date.date()),
            sell_date=str(sell_date.date()),
            buy_price=buy_price,
            sell_price=sell_price,
            return_pct=gross,
            net_return_pct=gross - cost,
            stop_price=stop_price,
            target_price=target_price,
            exit_reason=exit_reason,
            hold_days=hold_days,
        ))

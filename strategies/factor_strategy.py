"""
因子策略：把因子预测分数变成可回测的交易策略。

━━━ 桥接因子分析与交易系统 ━━━

当前:
  FactorAnalyzer → MultiFactorSynthesizer → SynthesisResult.alpha (预测分数)
  SignalGenerator → TradeSimulator → PortfolioManager → PortfolioResult

缺失:
  alpha 分数 → 交易信号 → TradeSimulator → PortfolioManager

本模块补上这个缺失:

  SynthesisResult.alpha
      ↓
  FactorStrategy (每日选 top K 只, 每 N 天换仓)
      ↓
  TradeSimulator (止损止盈出场)
      ↓
  PortfolioManager (资金分配 + 流动性约束)
      ↓
  PortfolioResult (净值曲线 + 夏普 + 回撤)

━━━ 两种模式 ━━━

  1. factor_select + with_stop:
     因子选股 + ATR止损止盈
     每 rebalance_days 天重新选 top K 只, 用止损止盈管理持仓

  2. factor_select_only:
     因子选股 + 固定持有 N 天
     每 rebalance_days 天选 top K, 持有 hold_days 天后卖出
     更纯粹的因子收益, 不混入止损止盈的贡献

用法:
    from strategies.factor_strategy import FactorStrategy

    strategy = FactorStrategy(
        alpha=result.alpha,     # SynthesisResult 的合成因子
        data=fa._data,           # FactorAnalyzer 的行情数据
        top_k=10,                # 每次选10只
        rebalance_days=5,        # 每5个交易日换仓
    )

    # 模式1: 因子选股 + 止损止盈
    pf_result = strategy.run_with_stop(
        stop_method='atr', n_atr=2.0, rr_ratio=2.0, max_hold_days=15,
        initial_capital=1_000_000, position_size=0.1, max_positions=10,
    )
    PerformanceAnalyzer.print_portfolio_result(pf_result)

    # 模式2: 因子选股 + 固定持有
    pf_result = strategy.run_fixed_hold(
        hold_days=10,
        initial_capital=1_000_000, position_size=0.1, max_positions=10,
    )
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger

from strategies.cost_model import CostModel
from strategies.portfolio_manager import PortfolioManager
from strategies.types import PortfolioResult, Trade


@dataclass
class ExecutionSpec:
    """
    统一的“成交时点/价格/持有规则”规格，用于把不同的买卖逻辑收敛到同一个执行入口。

    典型组合（与 IC return_basis 对应）：
      - 收盘买卖       ：buy_offset=0, buy_price='close', sell_price='close'
      - 开盘买/收盘卖   ：buy_offset=1, buy_price='open',  sell_price='close'
      - 次日开盘买卖   ：buy_offset=1, buy_price='open',  sell_price='open'

    字段
    ----
    buy_offset   : 买入相对信号日偏移（0=信号日, 1=次日）
    buy_price    : 买入价取自哪列，'open'|'close'（配合 buy_offset）
    sell_offset  : 卖出相对买入日的偏移天数（固定持有）
    sell_price   : 到期卖出价取自哪列，'open'|'close'
    sell_rule    : 卖出规则
                   'fixed'       固定到期卖（用 sell_price 列）
                   'limit'       到期日限价卖（锚定 buy_price × (1+sell_limit_pct)）
    sell_limit_pct: sell_rule='limit' 时的止盈幅度
    fallback_to_close: sell_rule='limit' 且未触价时，是否回退到收盘卖
    skip_limit_up : 买入日开盘涨停时跳过（买不进），默认 True
    """
    buy_offset: int = 1
    buy_price: str = 'open'
    sell_offset: int = 1
    sell_price: str = 'close'
    sell_rule: str = 'fixed'
    sell_limit_pct: float = 0.0
    fallback_to_close: bool = True
    skip_limit_up: bool = True
    stop_loss_pct: float = None  # >0 时逐日检查止损
    stop_exit: str = 'fixed'     # 'fixed'=乐观(触发按止损价成交) | 'next_open'=现实(买入日破位次日开盘卖,跳空按开盘价)


class FactorStrategy:
    """因子选股策略：alpha 分数 → 交易信号 → 回测。"""

    def __init__(
        self,
        alpha: pd.Series,        # MultiIndex (date, symbol) → 预测分数
        data: dict[str, pd.DataFrame],  # {symbol: DataFrame} 行情数据
        top_k: int = 10,          # 每次选 top K 只
        rebalance_days: int = 5,  # 每 N 个交易日换仓
        candidate_pool: int = 50, # 候选池大小（买不进就顺延到下一个）
        market_filter: pd.Series = None,  # 市场择时过滤：index=date, value=bool(True=允许开仓)
        universe_mask: pd.Series = None,  # 动态股票池过滤：MultiIndex(date, symbol) -> bool
    ):
        self._alpha = alpha
        self._data = data
        self._top_k = top_k
        self._rebalance_days = rebalance_days
        self._candidate_pool = candidate_pool
        self._market_filter = market_filter
        self._universe_mask = universe_mask

        self._daily_selections = self._build_selections()

    def _build_selections(self) -> dict:
        """每个换仓日选 top K 只股票。"""
        selections = {}

        # alpha 的 index 是 MultiIndex (date, symbol)
        if not isinstance(self._alpha.index, pd.MultiIndex):
            logger.warning("alpha 的 index 不是 MultiIndex (date, symbol)")
            return selections

        dates = self._alpha.index.get_level_values('date').unique().sort_values()

        skipped_by_filter = 0
        for i, date in enumerate(dates):
            # 每 rebalance_days 天换仓
            if i % self._rebalance_days != 0:
                continue

            # 市场择时过滤：filter 为 False 的日子跳过选股（旧持仓自然到期卖出）
            if self._market_filter is not None:
                ts = pd.Timestamp(date)
                # 在 filter index 中找最近的一天（处理非对齐）
                if ts in self._market_filter.index:
                    allow = bool(self._market_filter.loc[ts])
                else:
                    # 用最近的过往日期
                    valid = self._market_filter.index[self._market_filter.index <= ts]
                    allow = bool(self._market_filter.loc[valid[-1]]) if len(valid) > 0 else True
                if not allow:
                    skipped_by_filter += 1
                    continue

            # 当日所有股票的 alpha 分数
            mask = self._alpha.index.get_level_values('date') == date
            day_alpha = self._alpha[mask].sort_values(ascending=False)

            # 动态股票池过滤（仅保留当日允许交易的标的）
            if self._universe_mask is not None and not day_alpha.empty:
                allowed = self._universe_mask.reindex(day_alpha.index).fillna(False).astype(bool)
                day_alpha = day_alpha[allowed.values]

            # 选候选池（top candidate_pool, 买不进就顺延）
            selected = []
            for sym in day_alpha.index.get_level_values('symbol'):
                if sym in self._data:
                    selected.append(sym)
                if len(selected) >= self._candidate_pool:
                    break

            if selected:
                selections[date] = selected

        logger.info("因子选股 | {} 个换仓日 | 候选池{}只, 实选{}只 | 择时跳过{}日",
                    len(selections), self._candidate_pool, self._top_k, skipped_by_filter)
        return selections

    def generate_trades_with_stop(
        self,
        stop_method: str = 'atr',
        stop_params: dict = None,
        rr_ratio: float = 2.0,
        max_hold_days: int = 15,
    ) -> list[Trade]:
        """
        因子选股 + 止损止盈出场。

        流程:
          1. 每个换仓日选 top K 只
          2. 对每只选中的股票, 用止损止盈规则模拟交易
          3. 下一换仓日重新选股(旧持仓不自动平, 由止损止盈管理)
        """
        from strategies.cost_model import CostModel
        from strategies.stops import STOP_METHODS, calc_target

        stop_params = stop_params or {}
        cost = CostModel()
        trades = []

        all_dates = sorted(self._daily_selections.keys())

        for i, rebalance_date in enumerate(all_dates):
            selected = self._daily_selections[rebalance_date]

            for symbol in selected:
                df = self._data.get(symbol)
                if df is None:
                    continue

                # 找到换仓日在 df 中的位置
                try:
                    sig_loc = df.index.get_loc(rebalance_date)
                except KeyError:
                    continue

                buy_loc = sig_loc + 1  # T+1 开盘买入
                if buy_loc >= len(df):
                    continue

                buy_date = df.index[buy_loc]
                buy_open = float(df.iloc[buy_loc]['open'])
                if buy_open <= 0:
                    continue

                # 计算止损价
                stop_fn = STOP_METHODS.get(stop_method)
                if stop_fn is None:
                    continue
                try:
                    stop_price = stop_fn(df, rebalance_date, **stop_params)
                except Exception:
                    stop_price = buy_open * 0.95
                stop_price = min(stop_price, buy_open * 0.999)
                target_price = calc_target(buy_open, stop_price, rr_ratio)

                # 逐日检查出场
                sell_date = buy_date
                sell_price = buy_open
                exit_reason = 'time'
                actual_hold = 1

                for j in range(max_hold_days):
                    day_loc = buy_loc + j
                    if day_loc >= len(df):
                        break

                    bar = df.iloc[day_loc]
                    bar_date = df.index[day_loc]
                    day_low = float(bar['low'])
                    day_high = float(bar['high'])
                    day_open = float(bar['open'])
                    day_close = float(bar['close'])

                    if j > 0:
                        if day_open <= stop_price:
                            sell_date, sell_price = bar_date, day_open
                            actual_hold, exit_reason = j, 'gap_stop'
                            break
                        if day_open >= target_price:
                            sell_date, sell_price = bar_date, day_open
                            actual_hold, exit_reason = j, 'gap_target'
                            break

                    if day_low <= stop_price:
                        sell_date, sell_price = bar_date, stop_price
                        actual_hold, exit_reason = j + 1, 'stop'
                        break

                    if day_high >= target_price:
                        sell_date, sell_price = bar_date, target_price
                        actual_hold, exit_reason = j + 1, 'target'
                        break

                    # 检查是否到了下个换仓日 → 强制平仓换股
                    next_rebalance = all_dates[i + 1] if i + 1 < len(all_dates) else None
                    if next_rebalance is not None and bar_date >= next_rebalance:
                        sell_date, sell_price = bar_date, day_close
                        actual_hold, exit_reason = j + 1, 'rebalance'
                        break
                else:
                    # 时间止损
                    day_loc = min(buy_loc + max_hold_days - 1, len(df) - 1)
                    sell_date = df.index[day_loc]
                    sell_price = float(df.iloc[day_loc]['close'])
                    actual_hold = max_hold_days

                gross = (sell_price - buy_open) / buy_open
                trades.append(Trade(
                    symbol=symbol,
                    signal_date=str(rebalance_date.date()),
                    buy_date=str(buy_date.date()),
                    sell_date=str(sell_date.date()),
                    buy_price=buy_open,
                    sell_price=sell_price,
                    return_pct=gross,
                    net_return_pct=gross - cost.round_trip_cost_rate,
                    stop_price=stop_price,
                    target_price=target_price,
                    exit_reason=exit_reason,
                    hold_days=actual_hold,
                ))

        logger.info("因子策略生成 {} 笔交易", len(trades))
        return trades

    def generate_trades(
        self,
        spec: ExecutionSpec,
    ) -> list[Trade]:
        """
        通用交易生成入口：接收 ExecutionSpec，按指定买卖时点/价格/卖出规则生成交易。

        集中实现买卖时点的所有组合，现有各 generate_trades_* 方法委托到本方法。
        """
        cost = CostModel()
        trades = []

        buy_off = int(spec.buy_offset)
        all_dates = sorted(self._daily_selections.keys())

        for rebalance_date in all_dates:
            candidates = self._daily_selections[rebalance_date]
            picked = 0

            for symbol in candidates:
                if picked >= self._top_k:
                    break

                df = self._data.get(symbol)
                if df is None:
                    continue

                try:
                    sig_loc = df.index.get_loc(rebalance_date)
                except KeyError:
                    continue

                buy_loc = sig_loc + buy_off
                if buy_loc >= len(df) or buy_loc < 0:
                    continue

                buy_date = df.index[buy_loc]
                buy_bar = df.iloc[buy_loc]
                buy_price = float(buy_bar[spec.buy_price])
                buy_volume = float(buy_bar.get('volume', 0))

                # 停牌 / 无效价格
                if buy_volume <= 0 or buy_price <= 0:
                    continue

                # 涨停买不进（仅开盘价买入时才有意义）
                if spec.skip_limit_up and spec.buy_price == 'open' and buy_loc > 0:
                    prev_close = float(df.iloc[buy_loc - 1]['close'])
                    if prev_close > 0 and buy_price >= prev_close * 1.099:
                        continue

                # 卖出（固定到期 / 限价两种规则）
                sell_loc = min(buy_loc + int(spec.sell_offset), len(df) - 1)
                if sell_loc <= buy_loc:
                    continue
                sell_date = df.index[sell_loc]
                sell_bar = df.iloc[sell_loc]

                stop_price = 0.0
                actual_hold = int(spec.sell_offset)
                exit_reason = 'fixed_hold'
                sell_price = float(sell_bar[spec.sell_price])

                # 可选止损（逐日检查）
                # stop_exit='fixed'（默认，乐观口径）：从买入次日开始逐日检查，触发即按止损价成交
                # stop_exit='next_open'（现实口径）：
                #   - 买入当日盘中跌破止损 → 次日开盘卖出（当日无法卖出）
                #   - 后续交易日开盘已低于止损价（跳空）→ 按开盘价成交（实际会更差）
                #   - 盘中才跌破 → 按止损价成交
                if spec.stop_loss_pct is not None and spec.stop_loss_pct > 0:
                    stop_price = buy_price * (1 - spec.stop_loss_pct)
                    if spec.stop_exit == 'next_open':
                        for j in range(0, int(spec.sell_offset) + 1):
                            loc = buy_loc + j
                            if loc >= len(df):
                                break
                            bar = df.iloc[loc]
                            if float(bar['low']) <= stop_price:
                                if j == 0:
                                    nloc = loc + 1  # 买入当日破位：次日开盘卖
                                    if nloc >= len(df):
                                        break
                                    sell_loc = nloc
                                    sell_date = df.index[nloc]
                                    sell_bar = df.iloc[nloc]
                                    sell_price = float(sell_bar['open'])
                                    exit_reason = 'stop_next_open'
                                else:
                                    sell_loc = loc
                                    sell_date = df.index[loc]
                                    sell_bar = df.iloc[loc]
                                    o = float(sell_bar['open'])
                                    sell_price = o if o <= stop_price else stop_price
                                    exit_reason = 'stop_gap' if o <= stop_price else 'stop_loss'
                                actual_hold = sell_loc - buy_loc
                                break
                    else:
                        for j in range(1, int(spec.sell_offset) + 1):
                            loc = buy_loc + j
                            if loc >= len(df):
                                break
                            if float(df.iloc[loc]['low']) <= stop_price:
                                sell_loc = loc
                                sell_date = df.index[loc]
                                sell_bar = df.iloc[loc]
                                sell_price = stop_price
                                exit_reason = 'stop_loss'
                                actual_hold = j
                                stop_price = stop_price
                                break

                filled = True

                # 限价卖出：锚定买入价（仅固定持有到期时生效；止损已出场则不再覆盖）
                if spec.sell_rule == 'limit' and exit_reason == 'fixed_hold':
                    sell_limit = buy_price * (1 + spec.sell_limit_pct)
                    sell_open = float(sell_bar['open'])
                    sell_high = float(sell_bar['high'])
                    if sell_open >= sell_limit:
                        sell_price = sell_open
                        exit_reason = 'limit_sell_gap'
                    elif sell_high >= sell_limit:
                        sell_price = sell_limit
                        exit_reason = 'limit_sell_touch'
                    else:
                        if not spec.fallback_to_close:
                            continue
                        sell_price = float(sell_bar['close'])
                        exit_reason = 'close_fallback'
                        filled = False

                gross = (sell_price - buy_price) / buy_price
                trades.append(Trade(
                    symbol=symbol,
                    signal_date=str(rebalance_date.date()),
                    buy_date=str(buy_date.date()),
                    sell_date=str(sell_date.date()),
                    buy_price=buy_price,
                    sell_price=sell_price,
                    return_pct=gross,
                    net_return_pct=gross - cost.round_trip_cost_rate,
                    hold_days=actual_hold,
                    exit_reason=exit_reason,
                    stop_price=stop_price,
                    sell_pct=spec.sell_limit_pct if spec.sell_rule == 'limit' else 0.0,
                    filled=filled,
                ))
                picked += 1

        logger.info("统一交易生成 | spec={} | 生成 {} 笔",
                    spec, len(trades))
        return trades

    def generate_trades_fixed_hold(
        self,
        hold_days: int = 10,
        stop_loss_pct: float = None,
    ) -> list[Trade]:
        """
        因子选股 + 固定持有 N 天 + 涨停顺延 + 可选止损。

        从候选池中按排名依次检查可交易性:
          - 涨停(open >= prev_close * 1.099): 跳过, 换下一只
          - 停牌(volume == 0): 跳过, 换下一只
          - 可交易: 生成交易, 计入已选数
        直到选满 top_k 只或候选池耗尽。

        参数
        ----
        hold_days      : 固定持有天数
        stop_loss_pct  : 止损比例（如 0.08 表示跌 8% 止损）；None=不止损
                         买入后逐日检查当日最低价，触及止损价则提前卖出
        """
        spec = ExecutionSpec(
            buy_offset=1,
            buy_price='open',
            sell_offset=hold_days,
            sell_price='close',
            sell_rule='fixed',
            stop_loss_pct=stop_loss_pct,
        )
        return self.generate_trades(spec)

    def generate_trades_fixed_hold_limit(
        self,
        hold_days: int = 1,
        buy_limit_pct: float = 0.0,
        sell_limit_pct: float = 0.0,
        fallback_to_close: bool = True,
    ) -> list[Trade]:
        """
        因子选股 + 限价单执行 + 固定持有。

        规则（单日调仓推荐用法）
        ----------------------
        1. 信号日 T 生成候选名单
        2. T+1 以限价买入：buy_limit = prev_close × (1 + buy_limit_pct)
           - 若当日 low <= buy_limit，则视为成交，成交价=buy_limit
           - 否则跳过该笔交易
        3. 持有 hold_days 天后卖出
           - 若设置 sell_limit_pct，则卖出日限价：sell_limit = prev_close × (1 + sell_limit_pct)
           - 若当日 high >= sell_limit，则按 sell_limit 成交（filled=True）
           - 否则：fallback_to_close=True 时按收盘卖出；否则跳过卖出信号

        默认 buy_limit_pct=0 / sell_limit_pct=0：
          - 买入价 = 次日昨收平价挂单，要求日内回落到昨收才成交
          - 卖出价 = 持有到期日昨收平价挂单，若冲到昨收则成交，否则收盘卖出

        备注
        ----
        这是一个工程上简洁、无未来函数的日线近似，不模拟盘口排队。
        """
        cost = CostModel()
        trades = []

        all_dates = sorted(self._daily_selections.keys())

        for rebalance_date in all_dates:
            candidates = self._daily_selections[rebalance_date]
            picked = 0

            for symbol in candidates:
                if picked >= self._top_k:
                    break

                df = self._data.get(symbol)
                if df is None:
                    continue

                try:
                    sig_loc = df.index.get_loc(rebalance_date)
                except KeyError:
                    continue

                buy_loc = sig_loc + 1
                if buy_loc >= len(df) or sig_loc < 0:
                    continue

                prev_close = float(df.iloc[sig_loc]['close'])
                buy_bar = df.iloc[buy_loc]
                buy_date = df.index[buy_loc]
                buy_open = float(buy_bar['open'])
                buy_high = float(buy_bar['high'])
                buy_low = float(buy_bar['low'])
                buy_volume = float(buy_bar.get('volume', 0))

                if prev_close <= 0 or buy_volume <= 0:
                    continue

                # 限价买入：用前收为锚点
                buy_limit = prev_close * (1 + buy_limit_pct)

                # 若开盘已低于限价，按开盘成交；否则盘中触价按限价成交
                if buy_open <= buy_limit:
                    actual_buy = buy_open
                    buy_filled = True
                elif buy_low <= buy_limit <= buy_high or buy_low <= buy_limit:
                    actual_buy = buy_limit
                    buy_filled = True
                else:
                    continue

                # 卖出日
                sell_loc = min(buy_loc + hold_days, len(df) - 1)
                if sell_loc <= 0:
                    continue
                sell_bar = df.iloc[sell_loc]
                sell_date = df.index[sell_loc]
                sell_prev_close = float(df.iloc[sell_loc - 1]['close'])
                sell_high = float(sell_bar['high'])
                sell_close = float(sell_bar['close'])
                sell_limit = sell_prev_close * (1 + sell_limit_pct) if sell_prev_close > 0 else sell_close

                if sell_high >= sell_limit:
                    sell_price = sell_limit
                    sell_filled = True
                    exit_reason = 'limit_sell'
                else:
                    if not fallback_to_close:
                        continue
                    sell_price = sell_close
                    sell_filled = False
                    exit_reason = 'close_fallback'

                gross = (sell_price - actual_buy) / actual_buy
                trades.append(Trade(
                    symbol=symbol,
                    signal_date=str(rebalance_date.date()),
                    buy_date=str(buy_date.date()),
                    sell_date=str(sell_date.date()),
                    buy_price=actual_buy,
                    sell_price=sell_price,
                    return_pct=gross,
                    net_return_pct=gross - cost.round_trip_cost_rate,
                    hold_days=hold_days,
                    exit_reason=exit_reason,
                    filled=bool(buy_filled and sell_filled),
                ))
                picked += 1

        logger.info(
            "因子策略(固定持有{}天, 限价买{:+.1%}, 限价卖{:+.1%}) 生成 {} 笔交易",
            hold_days, buy_limit_pct, sell_limit_pct, len(trades)
        )
        return trades

    def generate_trades_fixed_hold_sell_limit(
        self,
        hold_days: int = 1,
        sell_limit_pct: float = 0.0,
        fallback_to_close: bool = True,
    ) -> list[Trade]:
        """
        因子选股 + T+1 开盘买入 + 到期日限价卖出。

        适用场景
        --------
        当前多因子单日调仓基线的真实时序是：
          - T 日收盘后得到信号
          - T+1 开盘买入
          - 持有 hold_days 天后，在到期日卖出

        当 hold_days=1 时：
          - 买点 = T+1 开盘
          - 卖点 = T+2 日

        本方法将原先“到期日收盘直接卖出”改为：
          1. 到期日开盘前挂一个基于买入价的止盈限价单
             sell_limit = buy_price × (1 + sell_limit_pct)
          2. 若到期日开盘价已高于限价，则按开盘价成交（更优成交）
          3. 否则若日内最高价触及限价，则按限价成交
          4. 若整天未触发，则尾盘按收盘价卖出（fallback_to_close=True）

        注意
        ----
        - 这里的 N% 锚定在“买入价”上，而不是昨收。
        - 这是日线级近似，不模拟盘口排队与撮合细节。
        """
        spec = ExecutionSpec(
            buy_offset=1,
            buy_price='open',
            sell_offset=hold_days,
            sell_price='close',
            sell_rule='limit',
            sell_limit_pct=sell_limit_pct,
            fallback_to_close=fallback_to_close,
        )
        return self.generate_trades(spec)

    # ------------------------------------------------------------------
    # 完整回测入口
    # ------------------------------------------------------------------

    def run_with_stop(
        self,
        stop_method: str = 'atr',
        stop_params: dict = None,
        rr_ratio: float = 2.0,
        max_hold_days: int = 15,
        initial_capital: float = 1_000_000,
        position_size: float = 0.1,
        max_positions: int = 10,
    ) -> 'PortfolioResult':
        """因子选股 + 止损止盈 + 组合管理 → 完整回测"""
        trades = self.generate_trades_with_stop(
            stop_method=stop_method,
            stop_params=stop_params,
            rr_ratio=rr_ratio,
            max_hold_days=max_hold_days,
        )

        pm = PortfolioManager(
            self._data,
            initial_capital=initial_capital,
            position_size=position_size,
            max_positions=max_positions,
        )
        return pm.allocate(trades)

    def run_fixed_hold(
        self,
        hold_days: int = 10,
        initial_capital: float = 1_000_000,
        position_size: float = 0.1,
        max_positions: int = 10,
    ) -> 'PortfolioResult':
        """因子选股 + 固定持有 + 组合管理 → 完整回测"""
        trades = self.generate_trades_fixed_hold(hold_days=hold_days)

        pm = PortfolioManager(
            self._data,
            initial_capital=initial_capital,
            position_size=position_size,
            max_positions=max_positions,
        )
        return pm.allocate(trades)

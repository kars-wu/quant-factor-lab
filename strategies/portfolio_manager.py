"""
组合管理层：PortfolioManager。

职责：接收 TradeSimulator 的"理想交易列表"（假设无限资金），
逐日过滤 + 分配仓位，输出真实净值曲线。

━━━ 数据流 ━━━

  SignalGenerator     → 信号
  TradeSimulator      → 理想交易列表（每笔交易含买卖价格/日期/收益率）
  PortfolioManager    → 过滤+仓位分配 → PortfolioResult（净值曲线+组合指标）
  PerformanceAnalyzer → 打印/绘图

核心算法（逐日遍历）：
  1. 先处理卖出（释放资金）
  2. 再处理买入（消耗资金，受 max_positions 和 cash 约束）
  3. Mark to market → 记录当日 NAV

用法：
    pm = PortfolioManager(gen.data, initial_capital=1_000_000,
                          position_size=0.1, max_positions=10)
    result = pm.allocate(trades)
    PerformanceAnalyzer.print_portfolio_result(result)
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd
from loguru import logger

from strategies.cost_model import CostModel
from strategies.types import PortfolioResult, PortfolioTrade, Trade


class PortfolioManager:
    """组合管理器：理想交易 → 真实组合净值曲线。"""

    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        initial_capital: float = 1_000_000,
        position_size: float = 0.1,
        max_positions: int = 10,
        position_sizing_mode: str = 'fixed_initial',
        daily_cash_split_warmup_days: int = 0,
        daily_cash_split_warmup_fraction: float = 0.5,
        allow_duplicate: bool = False,
        cost: CostModel = None,
        max_participation: float = 0.10,
        impact_coeff: float = 0.1,
        max_drawdown_limit: float = 0.15,
        cooldown_days: int = 10,
    ):
        """
        参数
        ----
        data               : {symbol: DataFrame}，来自 SignalGenerator.data
        initial_capital    : 初始总资金
        position_size      : 每笔仓位比例
        max_positions      : 最大同时持仓数
        position_sizing_mode: 仓位基准
                             - fixed_initial: 按初始资金 × position_size 固定仓位
                             - equity: 按当前组合权益 × position_size 做复利仓位
                             - cash: 按当前可用现金 × position_size（更保守）
                             - daily_cash_split: 按当日开盘前可用现金，平均分配给当日待买入股票
        daily_cash_split_warmup_days: daily_cash_split 模式下，起始前 N 个买入日采用暖启动
        daily_cash_split_warmup_fraction: 暖启动时，每个买入日最多使用 initial_capital × 此比例
        allow_duplicate    : 是否允许同一标的多次建仓
        cost               : 交易成本模型
        max_participation  : 买入金额不超过当日成交额的此比例（默认10%）
        impact_coeff       : 冲击成本系数，impact = coeff × (order_amount / daily_amount)
        max_drawdown_limit : 组合回撤超过此值触发熔断（默认15%）
        cooldown_days      : 熔断后冷却天数（不开新仓）
        """
        self._data = data
        self._initial_capital = initial_capital
        self._position_size = position_size
        self._max_positions = max_positions
        self._position_sizing_mode = position_sizing_mode
        self._daily_cash_split_warmup_days = daily_cash_split_warmup_days
        self._daily_cash_split_warmup_fraction = daily_cash_split_warmup_fraction
        self._allow_duplicate = allow_duplicate
        self._cost = cost or CostModel()
        self._max_participation = max_participation
        self._impact_coeff = impact_coeff
        self._max_drawdown_limit = max_drawdown_limit
        self._cooldown_days = cooldown_days

        if self._position_sizing_mode not in {'fixed_initial', 'equity', 'cash', 'daily_cash_split'}:
            raise ValueError(f'未知 position_sizing_mode: {self._position_sizing_mode}')

    def _estimate_total_value(self, cur_date: str, positions: dict, cash: float, price_field: str = 'open') -> float:
        """估算当前时点组合总权益。"""
        total_value = cash
        ts = pd.Timestamp(cur_date)
        for _, pos in positions.items():
            symbol = pos['symbol']
            df = self._data.get(symbol)
            px = pos['avg_cost']
            if df is not None and ts in df.index:
                row = df.loc[ts]
                if price_field in row.index and pd.notna(row[price_field]) and float(row[price_field]) > 0:
                    px = float(row[price_field])
                elif 'close' in row.index and pd.notna(row['close']) and float(row['close']) > 0:
                    px = float(row['close'])
            total_value += px * pos['quantity']
        return float(total_value)

    def _calc_target_amount(
        self,
        cur_date: str,
        positions: dict,
        cash: float,
        remaining_slots: int | None = None,
        buy_day_no: int | None = None,
        split_budget_remaining: float | None = None,
    ) -> float:
        """计算当前买入批次的目标下单金额。"""
        if self._position_sizing_mode == 'fixed_initial':
            base = self._initial_capital
        elif self._position_sizing_mode == 'cash':
            base = cash
        elif self._position_sizing_mode == 'daily_cash_split':
            slots = max(int(remaining_slots or 1), 1)
            if split_budget_remaining is not None:
                return float(max(split_budget_remaining, 0.0) / slots)
            if buy_day_no is not None and buy_day_no <= self._daily_cash_split_warmup_days:
                warmup_budget = self._initial_capital * self._daily_cash_split_warmup_fraction
                return float(min(max(cash, 0.0), warmup_budget) / slots)
            return float(max(cash, 0.0) * 0.95 / slots)
        else:  # equity
            # 买入发生在开盘，优先用当日开盘价估算持仓权益；缺失时回退到 close/avg_cost
            base = self._estimate_total_value(cur_date, positions, cash, price_field='open')
        return float(max(base, 0.0) * self._position_size)

    def _calc_daily_split_budget(self, cash: float, buy_day_no: int | None = None) -> float:
        """daily_cash_split 模式下，当日总买入预算。"""
        available = max(float(cash), 0.0)
        if buy_day_no is not None and buy_day_no <= self._daily_cash_split_warmup_days:
            return float(min(available, self._initial_capital * self._daily_cash_split_warmup_fraction))
        return float(available * 0.95)

    @staticmethod
    def _trade_key(trade: Trade) -> tuple:
        return (trade.symbol, trade.signal_date, trade.buy_date, trade.sell_date, int(trade.hold_days), trade.exit_reason)

    def allocate(self, trades: list[Trade]) -> PortfolioResult:
        """
        逐日遍历，过滤信号 + 分配仓位 → PortfolioResult。

        返回的 PortfolioResult 包含：
        - nav_series: 逐日净值（起始 1.0）
        - executed: 实际执行的交易（带数量和绝对盈亏）
        - skipped: 因资金/持仓上限被跳过的交易
        """
        if not trades:
            empty_nav = pd.Series(dtype=float)
            return PortfolioResult.from_nav_and_trades(
                empty_nav, [], [], self._initial_capital,
            )

        # 1. 构建交易日历（所有交易涉及日期的并集 + 数据中的日期）
        trade_dates = set()
        for t in trades:
            trade_dates.add(t.buy_date)
            trade_dates.add(t.sell_date)

        # 从 data 中获取完整交易日历（Timestamp index）
        all_dates = set()
        for df in self._data.values():
            all_dates.update(df.index)
        # 统一为字符串比较
        all_dates_str = {str(d.date()) if hasattr(d, 'date') else str(d) for d in all_dates}

        min_date = min(trade_dates)
        max_date = max(trade_dates)
        calendar = sorted(d for d in all_dates_str if min_date <= d <= max_date)

        if not calendar:
            logger.warning("无法构建交易日历")
            empty_nav = pd.Series(dtype=float)
            return PortfolioResult.from_nav_and_trades(
                empty_nav, [], trades, self._initial_capital,
            )

        # 2. 按买入日/卖出日分组
        trades_by_buy: dict[str, list[Trade]] = defaultdict(list)
        trades_by_sell: dict[str, list[Trade]] = defaultdict(list)
        for t in trades:
            trades_by_buy[t.buy_date].append(t)
            trades_by_sell[t.sell_date].append(t)

        # 若某交易日同时存在买入与卖出，默认视为：
        #   开盘买入（新信号） + 收盘卖出（旧持仓）
        # 因此该日不能“先卖后买”，否则会把收盘卖出的现金错误地
        # 提前用于当日开盘买入，形成资金前视。
        same_day_buy_sell_dates = set(trades_by_buy.keys()) & set(trades_by_sell.keys())

        # 3. 逐日遍历
        cash = self._initial_capital
        # positions: {trade_key: {'symbol': str, 'trade': Trade, 'quantity': int, 'avg_cost': float}}
        positions: dict = {}
        nav_records = []
        executed: list[PortfolioTrade] = []
        skipped: list[Trade] = []
        peak_nav = 1.0           # 历史最高净值
        cooldown_until = None     # 熔断冷却期截止日
        buy_day_counter = 0       # 已发生买入的交易日数量（用于 daily_cash_split 暖启动）

        def _process_sells(cur_date: str):
            nonlocal cash
            for trade in trades_by_sell.get(cur_date, []):
                trade_key = self._trade_key(trade)
                if trade_key not in positions:
                    continue
                pos = positions[trade_key]
                sell_amount = trade.sell_price * pos['quantity']
                sell_cost = sell_amount * self._cost.sell_cost_rate
                cash += sell_amount - sell_cost

                buy_amount = pos['avg_cost'] * pos['quantity']
                buy_cost = buy_amount * self._cost.buy_cost_rate
                pnl = (trade.sell_price - pos['avg_cost']) * pos['quantity'] - sell_cost - buy_cost

                executed.append(PortfolioTrade(
                    trade=trade,
                    quantity=pos['quantity'],
                    invested=buy_amount,
                    pnl=pnl,
                ))
                del positions[trade_key]

        def _process_buys(cur_date: str):
            nonlocal cash, buy_day_counter
            candidates = trades_by_buy.get(cur_date, [])
            if len(candidates) > 1:
                candidates = self._sort_by_score(candidates, cur_date)

            if self._position_sizing_mode == 'daily_cash_split':
                eligible_candidates = []
                for trade in candidates:
                    if len(positions) + len(eligible_candidates) >= self._max_positions:
                        skipped.append(trade)
                        continue

                    if cooldown_until and pd.Timestamp(cur_date) <= cooldown_until:
                        skipped.append(trade)
                        continue

                    if not self._allow_duplicate and any(pos['symbol'] == trade.symbol for pos in positions.values()):
                        skipped.append(trade)
                        continue

                    eligible_candidates.append(trade)

                candidates = eligible_candidates
                if candidates:
                    buy_day_counter += 1
                    day_split_budget_remaining = self._calc_daily_split_budget(cash, buy_day_counter)
                else:
                    day_split_budget_remaining = 0.0

            for i, trade in enumerate(candidates):
                if self._position_sizing_mode != 'daily_cash_split':
                    if len(positions) >= self._max_positions:
                        skipped.append(trade)
                        continue

                    if cooldown_until and pd.Timestamp(cur_date) <= cooldown_until:
                        skipped.append(trade)
                        continue

                    if not self._allow_duplicate and any(pos['symbol'] == trade.symbol for pos in positions.values()):
                        skipped.append(trade)
                        continue

                    target_amount = self._calc_target_amount(cur_date, positions, cash)
                else:
                    slots_left = len(candidates) - i
                    target_amount = self._calc_target_amount(
                        cur_date,
                        positions,
                        cash,
                        remaining_slots=slots_left,
                        buy_day_no=buy_day_counter,
                        split_budget_remaining=day_split_budget_remaining,
                    )

                df_sym = self._data.get(trade.symbol)
                daily_amount = 0
                if df_sym is not None:
                    ts_buy = pd.Timestamp(cur_date)
                    if ts_buy in df_sym.index and 'amount' in df_sym.columns:
                        daily_amount = float(df_sym.loc[ts_buy, 'amount'])

                if daily_amount > 0:
                    max_by_liquidity = daily_amount * self._max_participation
                else:
                    max_by_liquidity = target_amount

                actual_amount = min(target_amount, cash * 0.95, max_by_liquidity)
                if actual_amount < trade.buy_price * 100:
                    skipped.append(trade)
                    continue

                base_slippage = self._cost.slippage_rate
                if daily_amount > 0:
                    impact = base_slippage + self._impact_coeff * (actual_amount / daily_amount)
                else:
                    impact = base_slippage
                effective_buy_price = trade.buy_price * (1 + impact)

                quantity = int(actual_amount / effective_buy_price / 100) * 100
                if quantity < 100:
                    skipped.append(trade)
                    continue

                buy_amount = effective_buy_price * quantity
                commission = max(buy_amount * self._cost.commission_rate, self._cost.min_commission)
                buy_cost = commission + buy_amount * self._cost.slippage_rate
                cash -= buy_amount + buy_cost
                if self._position_sizing_mode == 'daily_cash_split':
                    day_split_budget_remaining -= buy_amount + buy_cost

                positions[self._trade_key(trade)] = {
                    'symbol': trade.symbol,
                    'trade': trade,
                    'quantity': quantity,
                    'avg_cost': effective_buy_price,
                }

        for date in calendar:
            # date 已经是字符串 'YYYY-MM-DD'

            if date in same_day_buy_sell_dates:
                # 开盘买 → 收盘卖
                _process_buys(date)
                _process_sells(date)
            else:
                # 常规日频：先卖后买
                _process_sells(date)
                _process_buys(date)

            # ------ c. Mark to market ------
            total_value = cash
            ts = pd.Timestamp(date)
            for _, pos in positions.items():
                symbol = pos['symbol']
                df = self._data.get(symbol)
                if df is not None and ts in df.index:
                    close = float(df.loc[ts, 'close'])
                else:
                    close = pos['avg_cost']
                total_value += close * pos['quantity']

            nav = total_value / self._initial_capital
            nav_records.append((ts, nav))

            # ------ d. 组合回撤熔断 ------
            if nav > peak_nav:
                peak_nav = nav
            drawdown = (nav - peak_nav) / peak_nav if peak_nav > 0 else 0

            if drawdown < -self._max_drawdown_limit and positions:
                logger.warning(
                    "触发组合熔断！回撤 {:.1%} < -{:.0%}，清仓 {} 只，冷却 {} 天",
                    drawdown, self._max_drawdown_limit, len(positions), self._cooldown_days,
                )
                for pos_key in list(positions.keys()):
                    pos = positions[pos_key]
                    sym = pos['symbol']
                    df_pos = self._data.get(sym)
                    sell_price = pos['avg_cost']
                    if df_pos is not None and ts in df_pos.index:
                        sell_price = float(df_pos.loc[ts, 'close'])
                    sell_amount = sell_price * pos['quantity']
                    sell_cost = max(sell_amount * self._cost.commission_rate, self._cost.min_commission) \
                        + sell_amount * self._cost.stamp_duty_rate \
                        + sell_amount * self._cost.slippage_rate
                    cash += sell_amount - sell_cost
                    del positions[pos_key]
                cooldown_until = pd.Timestamp(date) + pd.Timedelta(days=self._cooldown_days)

        # 4. 构建净值序列
        nav_series = pd.Series(
            [v for _, v in nav_records],
            index=pd.DatetimeIndex([d for d, _ in nav_records]),
            name='nav',
        )

        logger.info(
            "组合管理完成 | 执行 {} 笔 / 跳过 {} 笔 | "
            "终值 {:.0f} / 净值 {:.3f}",
            len(executed), len(skipped),
            nav_series.iloc[-1] * self._initial_capital,
            nav_series.iloc[-1],
        )

        return PortfolioResult.from_nav_and_trades(
            nav_series, executed, skipped, self._initial_capital,
        )

    def _sort_by_score(self, candidates: list, date: str) -> list:
        """
        按信号质量排序同日信号。

        打分逻辑：换手率 + 成交额（流动性优先）。
        未来可扩展为按策略类型选择不同打分函数。
        """
        from strategies.scoring import score_by_liquidity

        ts = pd.Timestamp(date)
        scored = []
        for trade in candidates:
            df = self._data.get(trade.symbol)
            score = 0.0
            if df is not None and ts in df.index:
                scores = score_by_liquidity(df, [ts])
                score = scores.get(ts, scores.get(date, 0.0))
            scored.append((trade, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored]

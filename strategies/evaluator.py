"""
向量化策略评测框架（门面层）。

━━━ 架构说明 ━━━

本模块是三层架构的门面入口，内部委托给三个独立类：

  SignalGenerator      选股层 → 生成信号（strategies/signal_generator.py）
  TradeSimulator       交易层 → 模拟交易（strategies/trade_simulator.py）
  PerformanceAnalyzer  评测层 → 计算指标（strategies/analyzer.py）

数据类 Trade / EvalResult 定义在 strategies/types.py。

StrategyEvaluator 是旧版接口，保留用于兼容外部调用。
新代码建议直接使用三个独立类：

    gen = SignalGenerator(db, universe, start, end)
    gen.load_data()
    signals = gen.generate(signal_fn, params, screener)

    sim = TradeSimulator(gen.data)
    trades = sim.simulate_with_stop(signals, 'atr', {'n_atr': 2.0}, 2.0, 10)

    result = PerformanceAnalyzer.analyze(trades, None, start, end)
    PerformanceAnalyzer.print_result(result)
"""

from __future__ import annotations

from typing import Callable, Optional

from loguru import logger

from strategies.analyzer import PerformanceAnalyzer
from strategies.cost_model import CostModel
from strategies.screener import Screener
from strategies.signal_generator import SignalGenerator
from strategies.trade_simulator import TradeSimulator
from strategies.types import EvalResult


class StrategyEvaluator:
    """
    门面类：组合 SignalGenerator + TradeSimulator + PerformanceAnalyzer。

    旧接口保持不变，外部调用方无需修改。
    """

    # 交易成本（向后兼容，引用统一 CostModel）
    COMMISSION_RATE = CostModel().commission_rate
    STAMP_DUTY_SELL = CostModel().stamp_duty_rate
    SLIPPAGE = CostModel().slippage_rate

    def __init__(
        self,
        db,
        universe: list[str],
        start_date: str,
        end_date: str,
        min_amount: float = 5e7,
        exclude_st: bool = True,
        min_listed_days: int = 250,
        open_stop_loss_pct: float = None,
    ):
        self.db = db
        self.universe = universe
        self.start_date = start_date
        self.end_date = end_date
        self.min_amount = min_amount
        self.open_stop_loss_pct = open_stop_loss_pct

        self._gen = SignalGenerator(
            db=db, universe=universe,
            start_date=start_date, end_date=end_date,
            min_amount=min_amount, exclude_st=exclude_st,
            min_listed_days=min_listed_days,
        )
        self._sim: Optional[TradeSimulator] = None

    # ------------------------------------------------------------------
    # 数据加载（委托给 SignalGenerator）
    # ------------------------------------------------------------------

    @property
    def _data(self):
        return self._gen.data

    @property
    def _valid_symbols(self):
        return self._gen.valid_symbols

    def load_data(self, lookback_days: int = 60):
        self._gen.load_data(lookback_days)
        self._sim = TradeSimulator(self._gen.data)

    # ------------------------------------------------------------------
    # 模式 1：次日限价单出场（旧模式）
    # ------------------------------------------------------------------

    def evaluate(
        self,
        signal_fn: Callable,
        signal_params: dict,
        sell_pct_grid: list[float],
        screener: Optional[Screener] = None,
        position_size: float = 0.1,
    ) -> dict[float, EvalResult]:
        """对所有 sell_pct 跑评测，返回 {sell_pct: EvalResult}。"""
        if self._sim is None:
            raise RuntimeError("请先调用 load_data()")

        signals = self._gen.generate(signal_fn, signal_params, screener)
        if signals.empty:
            return {}

        results = {}
        for sell_pct in sell_pct_grid:
            trades = self._sim.simulate_next_day(
                signals, sell_pct, self.open_stop_loss_pct,
            )
            results[sell_pct] = PerformanceAnalyzer.analyze(
                trades, sell_pct, self.start_date, self.end_date,
            )
            logger.debug("sell_pct={:.1%}: {} 笔交易", sell_pct, len(trades))

        return results

    # ------------------------------------------------------------------
    # 模式 2：多日止损止盈出场（新模式）
    # ------------------------------------------------------------------

    def evaluate_with_stop(
        self,
        signal_fn: Callable,
        signal_params: dict,
        stop_method: str = "atr",
        stop_params: dict = None,
        rr_ratio: float = 2.0,
        max_hold_days: int = 10,
        screener: Optional[Screener] = None,
    ) -> EvalResult:
        """多日止损止盈评测。"""
        if self._sim is None:
            raise RuntimeError("请先调用 load_data()")

        signals = self._gen.generate(signal_fn, signal_params, screener)
        if signals.empty:
            return PerformanceAnalyzer.analyze([], None, self.start_date, self.end_date)

        trades = self._sim.simulate_with_stop(
            signals, stop_method, stop_params, rr_ratio, max_hold_days,
        )
        return PerformanceAnalyzer.analyze(
            trades, None, self.start_date, self.end_date,
        )

    # ------------------------------------------------------------------
    # 输出（委托给 PerformanceAnalyzer）
    # ------------------------------------------------------------------

    def print_summary(self, results: dict[float, EvalResult]):
        PerformanceAnalyzer.print_summary(results)

    def print_result(self, result: EvalResult):
        PerformanceAnalyzer.print_result(result)

    def plot(self, results: dict[float, EvalResult], save_path: Optional[str] = None):
        PerformanceAnalyzer.plot(results, save_path)

    # ------------------------------------------------------------------
    # 内部方法（兼容旧版调用）
    # ------------------------------------------------------------------

    def _generate_signals(self, signal_fn, signal_params, screener=None):
        return self._gen.generate(signal_fn, signal_params, screener)

    def _add_stop_prices(self, signals, stop_method, stop_params, rr_ratio):
        return self._sim._add_stop_prices(signals, stop_method, stop_params, rr_ratio)

    def _simulate_trades(self, signals, sell_pct, open_stop_loss_pct=None):
        return self._sim.simulate_next_day(signals, sell_pct, open_stop_loss_pct)

    def _simulate_trades_multiday(self, signals, max_hold_days):
        return self._sim._simulate_trades_multiday(signals, max_hold_days)

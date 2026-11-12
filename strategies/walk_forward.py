"""
Walk-Forward 参数优化器。

━━━ 背景 ━━━

静态模型必然衰退，Alpha 的半衰期通常在 6-18 个月。
在完整区间上扫网格选最优，本质是过拟合。

Walk-Forward 的思路：
  1. 用前 train_window 天数据选最优参数
  2. 用该参数在接下来 test_window 天做交易
  3. 滑动窗口，重复以上过程
  4. 把所有 test_window 拼起来 = 样本外真实表现

━━━ 用法 ━━━

    from strategies.walk_forward import WalkForwardOptimizer
    from strategies.signal_generator import SignalGenerator
    from strategies.trade_simulator import TradeSimulator
    from strategies.portfolio_manager import PortfolioManager

    opt = WalkForwardOptimizer(
        gen=gen, sim=sim, pm=pm,
        train_window=252, test_window=63,
    )
    result = opt.optimize(
        signal_fn=signal_fn,
        signal_params=signal_params,
        stop_methods=['atr', 'swing'],
        stop_grid={'atr': [{'n_atr': 1.5}, {'n_atr': 2.0}],
                   'swing': [{'lookback': 3}, {'lookback': 5}]},
        rr_ratios=[1.5, 2.0],
        max_holds=[5, 10],
    )
    # 输出每个窗口的最优参数 + 样本外累计表现
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger

from strategies.types import EvalResult


@dataclass
class WalkForwardResult:
    """Walk-Forward 优化结果。"""
    window_results: list = field(default_factory=list)  # 每个窗口的结果
    oos_trades: list = field(default_factory=list)      # 样本外交易列表
    oos_sharpe: float = 0.0
    oos_annual: float = 0.0
    oos_total_return: float = 0.0
    avg_param_consistency: float = 0.0  # 参数稳定性（相邻窗口参数变化率）

    @property
    def n_windows(self) -> int:
        return len(self.window_results)

    def summary(self) -> str:
        if not self.window_results:
            return "无结果"
        return (f"Walk-Forward {self.n_windows} 个窗口 | "
                f"样本外 Sharpe={self.oos_sharpe:.2f} | "
                f"年化={self.oos_annual:+.1%} | "
                f"参数稳定性={self.avg_param_consistency:.1%}")


class WalkForwardOptimizer:
    """滚动窗口参数优化器。"""

    def __init__(
        self,
        gen,           # SignalGenerator
        sim,           # TradeSimulator
        pm,            # PortfolioManager
        train_window: int = 252,    # 1年训练
        test_window: int = 63,      # 3个月测试
        step: int = None,           # 滑动步长（默认=test_window）
    ):
        self._gen = gen
        self._sim = sim
        self._pm = pm
        self._train_window = train_window
        self._test_window = test_window
        self._step = step or test_window

    def optimize(
        self,
        signal_fn: Callable,
        signal_params: dict,
        stop_grid: dict,        # {method: [params, ...]}
        rr_ratios: list,
        max_holds: list,
        start_date: str,
        end_date: str,
        screener=None,
        buy_offset: int = 1,
        position_size: float = 0.1,
        max_positions: int = 10,
        top_k: int = 1,         # 每个窗口选前k个参数组合
    ) -> WalkForwardResult:
        """
        Walk-Forward 优化。

        流程：
        1. 按 train_window/test_window 分割时间段
        2. 在训练窗口上扫所有参数组合，选最优
        3. 用最优参数在测试窗口上生成交易
        4. 拼接所有测试窗口的交易 = 样本外表现
        """
        # 构建时间窗口
        dates = pd.date_range(start_date, end_date, freq='B')
        n = len(dates)
        windows = []
        i = 0
        while i + self._train_window + self._test_window <= n:
            train_start = str(dates[i].date())
            train_end = str(dates[i + self._train_window - 1].date())
            test_start = str(dates[i + self._train_window].date())
            test_end = str(dates[min(i + self._train_window + self._test_window - 1, n - 1)].date())
            windows.append((train_start, train_end, test_start, test_end))
            i += self._step

        if not windows:
            logger.warning("Walk-Forward: 区间太短，需要至少 {}+{} 天",
                           self._train_window, self._test_window)
            return WalkForwardResult()

        logger.info("Walk-Forward: {} 个窗口 (训练{}天/测试{}天)",
                    len(windows), self._train_window, self._test_window)

        # 构建参数组合网格
        all_combos = []
        for method, param_list in stop_grid.items():
            for params in param_list:
                for rr in rr_ratios:
                    for hold in max_holds:
                        all_combos.append((method, params, rr, hold))

        result = WalkForwardResult()
        prev_best_params = None
        all_oos_trades = []
        param_changes = []

        for win_idx, (tr_start, tr_end, te_start, te_end) in enumerate(windows):
            # 训练窗口：生成信号 → 模拟交易 → 选最优参数
            best_sharpe = -999
            best_combo = None

            for method, params, rr, hold in all_combos:
                try:
                    tr_signals = self._gen.generate(
                        signal_fn, signal_params, screener,
                        buy_price_field=getattr(signal_fn, 'buy_price_field', 'close'),
                    )
                    tr_signals = tr_signals[
                        (tr_signals['signal_date'] >= pd.Timestamp(tr_start)) &
                        (tr_signals['signal_date'] <= pd.Timestamp(tr_end))
                    ]
                    if tr_signals.empty:
                        continue

                    trades = self._sim.simulate_with_stop(
                        tr_signals, stop_method=method, stop_params=params,
                        rr_ratio=rr, max_hold_days=hold, buy_offset=buy_offset,
                    )
                    if not trades:
                        continue

                    eval = EvalResult.from_trades(trades, None, tr_start, tr_end)
                    if eval.sharpe > best_sharpe:
                        best_sharpe = eval.sharpe
                        best_combo = (method, params, rr, hold)
                except Exception as e:
                    logger.debug("Walk-Forward 训练失败 {}: {}", method, e)

            if best_combo is None:
                continue

            # 参数稳定性
            if prev_best_params is not None:
                change = self._param_change_rate(prev_best_params, best_combo)
                param_changes.append(change)
            prev_best_params = best_combo

            # 测试窗口：用训练窗口选出的最优参数做交易
            method, params, rr, hold = best_combo
            try:
                te_signals = self._gen.generate(
                    signal_fn, signal_params, screener,
                    buy_price_field=getattr(signal_fn, 'buy_price_field', 'close'),
                )
                te_signals = te_signals[
                    (te_signals['signal_date'] >= pd.Timestamp(te_start)) &
                    (te_signals['signal_date'] <= pd.Timestamp(te_end))
                ]
                if te_signals.empty:
                    continue

                te_trades = self._sim.simulate_with_stop(
                    te_signals, stop_method=method, stop_params=params,
                    rr_ratio=rr, max_hold_days=hold, buy_offset=buy_offset,
                )
                all_oos_trades.extend(te_trades)

                te_eval = EvalResult.from_trades(te_trades, None, te_start, te_end)
                result.window_results.append({
                    'window': f'{te_start}~{te_end}',
                    'best_params': f'{method}/{params} rr={rr} hold={hold}',
                    'train_sharpe': best_sharpe,
                    'test_sharpe': te_eval.sharpe,
                    'test_annual': te_eval.annual_return,
                    'test_trades': len(te_trades),
                })
                logger.info("窗口 {} | 训练Sharpe={:.2f} → 测试Sharpe={:.2f}",
                            f'{te_start}~{te_end}', best_sharpe, te_eval.sharpe)
            except Exception as e:
                logger.debug("Walk-Forward 测试失败: {}", e)

        # 汇总
        result.oos_trades = all_oos_trades
        if all_oos_trades:
            oos_eval = EvalResult.from_trades(all_oos_trades, None, start_date, end_date)
            result.oos_sharpe = oos_eval.sharpe
            result.oos_annual = oos_eval.annual_return
            result.oos_total_return = float(np.prod([1 + t.net_return_pct for t in all_oos_trades]) - 1)

        if param_changes:
            result.avg_param_consistency = 1.0 - float(np.mean(param_changes))

        logger.info("Walk-Forward 完成 | {}", result.summary())
        return result

    @staticmethod
    def _param_change_rate(prev, curr) -> float:
        """计算参数变化率（0=完全一致，1=完全不同）。"""
        # 简化：比较方法名和关键参数
        if prev[0] != curr[0]:
            return 1.0  # 方法变了 = 100%变化
        # 方法相同，比较参数dict
        p1, p2 = prev[1], curr[1]
        if p1 == p2 and prev[2] == curr[2] and prev[3] == curr[3]:
            return 0.0
        return 0.5  # 部分变化

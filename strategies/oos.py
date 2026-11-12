"""
样本外验证（OOS）+ Walk-Forward 分析。

━━━ 背景 ━━━

参数网格扫描本质是过拟合：在完整区间上扫几十个组合选最优，
往往只是"恰好在历史数据上表现好"。

本模块实现两种验证方式：
  1. 简单样本内外分割：前 60% 训练（选参数），后 40% 测试（验证泛化）
  2. Walk-Forward：滚动窗口，每 train_window 天选参数，test_window 天验证

━━━ 用法 ━━━

    from strategies.oos import WalkForwardValidator

    validator = WalkForwardValidator()
    splits = validator.split('2020-01-01', '2026-05-14', train_ratio=0.6)
    # 返回 [(train_start, train_end, test_start, test_end), ...]

    result = validator.validate(
        signals, trades, splits,
        sim=TradeSimulator(gen.data),
        pm=PortfolioManager(gen.data),
    )
    # 返回 OOSResult：样本内夏普 vs 样本外夏普、衰减率
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from loguru import logger

from strategies.types import EvalResult, Trade


@dataclass
class OOSResult:
    """样本外验证结果。"""
    in_sample_sharpe: float = 0.0          # 样本内夏普
    out_of_sample_sharpe: float = 0.0       # 样本外夏普
    decay_ratio: float = 0.0               # 衰减率 = OOS/IS（<0.5 说明过拟合）
    in_sample_annual: float = 0.0           # 样本内年化
    out_of_sample_annual: float = 0.0       # 样本外年化
    in_sample_trades: int = 0
    out_of_sample_trades: int = 0
    wf_results: list = field(default_factory=list)  # Walk-Forward 各窗口结果

    @property
    def is_overfit(self) -> bool:
        """是否过拟合（样本外夏普 < 样本内的 50%）。"""
        if self.in_sample_sharpe <= 0:
            return False
        return self.decay_ratio < 0.5

    def summary(self) -> str:
        return (f"样本内 Sharpe={self.in_sample_sharpe:.2f} "
                f"vs 样本外 Sharpe={self.out_of_sample_sharpe:.2f} "
                f"| 衰减率={self.decay_ratio:.1%}"
                f"{' ⚠ 过拟合风险' if self.is_overfit else ' ✓'}")


class WalkForwardValidator:
    """样本外验证器。"""

    def split(
        self,
        start_date: str,
        end_date: str,
        train_ratio: float = 0.6,
    ) -> list[tuple[str, str, str, str]]:
        """
        简单样本内外分割。

        返回 [(train_start, train_end, test_start, test_end)]
        """
        dates = pd.date_range(start_date, end_date, freq='B')
        n = len(dates)
        split_point = int(n * train_ratio)

        train_start = str(dates[0].date())
        train_end = str(dates[split_point - 1].date())
        test_start = str(dates[split_point].date())
        test_end = str(dates[-1].date())

        return [(train_start, train_end, test_start, test_end)]

    def split_walk_forward(
        self,
        start_date: str,
        end_date: str,
        train_window: int = 252,    # 1年训练
        test_window: int = 63,     # 3个月测试
    ) -> list[tuple[str, str, str, str]]:
        """
        Walk-Forward 滚动窗口分割。

        每 train_window 个交易日训练，test_window 个交易日测试，
        滑动窗口直到覆盖完整回测区间。
        """
        dates = pd.date_range(start_date, end_date, freq='B')
        n = len(dates)
        splits = []

        i = 0
        while i + train_window + test_window <= n:
            train_start = str(dates[i].date())
            train_end = str(dates[i + train_window - 1].date())
            test_start = str(dates[i + train_window].date())
            test_end = str(dates[min(i + train_window + test_window - 1, n - 1)].date())
            splits.append((train_start, train_end, test_start, test_end))
            i += test_window  # 滑动步长 = test_window

        return splits

    def validate(
        self,
        trades: list[Trade],
        splits: list[tuple[str, str, str, str]],
    ) -> OOSResult:
        """
        对交易列表按时间分割，计算样本内外指标。

        注意：这是一种"事后验证"——交易列表已由 TradeSimulator 在完整
        区间上生成。真正的 Walk-Forward 应该在每个训练窗口上重新选参数
        再在测试窗口上跑策略，但那需要完整的信号→模拟流程，这里做的是
        简化版：把已有交易按时间分到训练/测试集，对比指标。
        """
        if not trades:
            return OOSResult()

        result = OOSResult()

        for split in splits:
            train_start, train_end, test_start, test_end = split

            # 按买入日期分割交易
            in_trades = [t for t in trades if train_start <= t.buy_date <= train_end]
            out_trades = [t for t in trades if test_start <= t.buy_date <= test_end]

            if in_trades:
                in_result = EvalResult.from_trades(
                    in_trades, None, train_start, train_end,
                )
                result.in_sample_sharpe += in_result.sharpe
                result.in_sample_annual += in_result.annual_return
                result.in_sample_trades += len(in_trades)

            if out_trades:
                out_result = EvalResult.from_trades(
                    out_trades, None, test_start, test_end,
                )
                result.out_of_sample_sharpe += out_result.sharpe
                result.out_of_sample_annual += out_result.annual_return
                result.out_of_sample_trades += len(out_trades)

                result.wf_results.append({
                    'test_period': f'{test_start} ~ {test_end}',
                    'sharpe': out_result.sharpe,
                    'annual': out_result.annual_return,
                    'trades': len(out_trades),
                })

        # 平均化
        n_splits = len(splits)
        if n_splits > 0:
            result.in_sample_sharpe /= n_splits
            result.in_sample_annual /= n_splits
            result.out_of_sample_sharpe /= n_splits
            result.out_of_sample_annual /= n_splits

        if result.in_sample_sharpe > 0:
            result.decay_ratio = result.out_of_sample_sharpe / result.in_sample_sharpe

        logger.info("OOS验证 | {}", result.summary())

        return result

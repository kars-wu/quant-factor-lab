"""
数据类：Trade 和 EvalResult。

纯数据容器 + from_trades 统计逻辑，不依赖任何其他策略模块。
三层架构（SignalGenerator / TradeSimulator / PerformanceAnalyzer）
之间通过 Trade 列表传递数据。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 交易记录
# ------------------------------------------------------------------

@dataclass
class Trade:
    """单笔交易记录（信号 → 买入 → 卖出 全过程）。"""
    symbol: str
    signal_date: str        # 信号产生日
    buy_date: str           # 实际买入日
    sell_date: str          # 卖出日
    buy_price: float        # 买入价
    sell_price: float       # 卖出价
    return_pct: float       # 收益率（扣费前）
    net_return_pct: float   # 净收益率（扣费后）
    # 次日出场模式字段
    sell_pct: float = 0.0           # 目标涨幅超参数
    filled: bool = False            # 限价单是否成交
    # 多日止损止盈模式字段
    stop_price: float = 0.0         # 止损价
    target_price: float = 0.0       # 止盈价
    exit_reason: str = ''           # 'target' | 'stop' | 'time' | 'gap_stop' | 'gap_target'
    hold_days: int = 0              # 实际持仓天数


# ------------------------------------------------------------------
# 评测结果
# ------------------------------------------------------------------

@dataclass
class EvalResult:
    """策略评测汇总指标。"""
    sell_pct: Optional[float]
    total_trades: int
    fill_rate: float                # 限价单达标率（次日出场模式）
    win_rate: float                 # 盈利比例（净收益>0）
    avg_gross_return: float
    avg_net_return: float
    median_return: float
    max_gain: float
    max_loss: float
    annual_return: float
    sharpe: float
    trades: list = field(default_factory=list, repr=False)
    # 多日止损止盈模式新增字段
    avg_hold_days: float = 0.0      # 平均持仓天数
    actual_rr: float = 0.0          # 实际盈亏比（avg_win / |avg_loss|）
    avg_risk_pct: float = 0.0       # 平均风险敞口（止损距离/买入价）
    exit_stats: dict = field(default_factory=dict)  # 出场原因分布

    @classmethod
    def from_trades(
        cls,
        trades: list[Trade],
        sell_pct: Optional[float],
        start_date: str,
        end_date: str,
    ) -> "EvalResult":
        """从交易记录列表计算全部评测指标。"""
        if not trades:
            return cls(
                sell_pct=sell_pct, total_trades=0, fill_rate=0,
                win_rate=0, avg_gross_return=0, avg_net_return=0,
                median_return=0, max_gain=0, max_loss=0,
                annual_return=0, sharpe=0, trades=[],
            )

        net_returns = np.array([t.net_return_pct for t in trades])
        gross_returns = np.array([t.return_pct for t in trades])
        filled = np.array([t.filled for t in trades])

        # ------ 年化收益 & 夏普（统一使用全量日序列，含空仓日收益0）------
        df_trades = pd.DataFrame({
            "sell_date": pd.to_datetime([t.sell_date for t in trades]),
            "net_return": net_returns,
        })
        daily_ret = (
            df_trades.groupby("sell_date")["net_return"]
            .mean()
            .reindex(
                pd.date_range(start_date, end_date, freq="B"),
                fill_value=0.0,
            )
        )
        total_days = len(daily_ret)
        years = max(total_days / 252, 0.1)
        cum = (1 + daily_ret).prod()
        annual_return = float(cum ** (1 / years) - 1)

        # 夏普比率：用全量日收益率序列（含空仓日0收益），
        # 与 PortfolioResult 和 report/metrics.py 保持一致
        if len(daily_ret) > 1 and daily_ret.std() > 0:
            sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
        else:
            sharpe = 0.0

        # ------ 多日止损止盈模式新增指标 ------
        avg_hold_days = float(np.mean([t.hold_days for t in trades]))

        wins = net_returns[net_returns > 0]
        losses = net_returns[net_returns <= 0]
        if len(wins) > 0 and len(losses) > 0 and losses.mean() != 0:
            actual_rr = float(wins.mean() / abs(losses.mean()))
        else:
            actual_rr = 0.0

        risk_pcts = [
            (t.buy_price - t.stop_price) / t.buy_price
            for t in trades if t.stop_price > 0 and t.buy_price > 0
        ]
        avg_risk_pct = float(np.mean(risk_pcts)) if risk_pcts else 0.0

        exit_stats = dict(Counter(t.exit_reason for t in trades if t.exit_reason))

        return cls(
            sell_pct=sell_pct,
            total_trades=len(trades),
            fill_rate=float(filled.mean()),
            win_rate=float((net_returns > 0).mean()),
            avg_gross_return=float(gross_returns.mean()),
            avg_net_return=float(net_returns.mean()),
            median_return=float(np.median(net_returns)),
            max_gain=float(net_returns.max()),
            max_loss=float(net_returns.min()),
            annual_return=annual_return,
            sharpe=sharpe,
            trades=trades,
            avg_hold_days=avg_hold_days,
            actual_rr=actual_rr,
            avg_risk_pct=avg_risk_pct,
            exit_stats=exit_stats,
        )


# ------------------------------------------------------------------
# 带仓位的交易记录（PortfolioManager 输出）
# ------------------------------------------------------------------

@dataclass
class PortfolioTrade:
    """带仓位信息的交易记录。"""
    trade: Trade              # 原始交易（收益率、价格、日期）
    quantity: int             # 实际买入股数
    invested: float           # 实际投入金额（buy_price × quantity）
    pnl: float                # 绝对盈亏（扣费后）

    @property
    def pnl_pct(self) -> float:
        """投入资金收益率。"""
        if self.invested <= 0:
            return 0.0
        return self.pnl / self.invested


# ------------------------------------------------------------------
# 组合级回测结果
# ------------------------------------------------------------------

@dataclass
class PortfolioResult:
    """组合级回测结果（含净值曲线和真实资金管理）。"""
    nav_series: pd.Series                 # 逐日净值（起始1.0）
    executed: list = field(default_factory=list)  # list[PortfolioTrade]
    skipped: list = field(default_factory=list)   # list[Trade] 因资金不足被跳过
    initial_capital: float = 1_000_000
    final_value: float = 0.0
    total_return: float = 0.0             # (final - initial) / initial
    annual_return: float = 0.0            # CAGR，基于 NAV 曲线
    max_drawdown: float = 0.0             # 最大回撤
    max_drawdown_duration: int = 0        # 最大回撤持续天数
    sharpe: float = 0.0                   # 基于每日 NAV 收益
    calmar: float = 0.0                   # annual_return / max_drawdown
    capital_utilization: float = 0.0      # 平均资金利用率
    total_trades: int = 0
    skipped_count: int = 0
    win_rate: float = 0.0                 # 盈利交易占比
    avg_pnl: float = 0.0                  # 平均绝对盈亏
    turnover_rate: float = 0.0           # 年换手率（双边交易额/平均资产）

    @classmethod
    def from_nav_and_trades(
        cls,
        nav_series: pd.Series,
        executed: list,
        skipped: list,
        initial_capital: float,
    ) -> "PortfolioResult":
        """从净值序列和交易列表计算全部组合级指标。"""
        if nav_series.empty:
            return cls(
                nav_series=nav_series,
                executed=executed, skipped=skipped,
                initial_capital=initial_capital,
            )

        final_value = float(nav_series.iloc[-1]) * initial_capital
        total_return = (final_value - initial_capital) / initial_capital

        # 年化（CAGR）
        n_days = len(nav_series)
        years = max(n_days / 252, 0.1)
        cum = float(nav_series.iloc[-1])
        annual_return = cum ** (1 / years) - 1 if cum > 0 else -1.0

        # 每日收益率
        daily_ret = nav_series.pct_change().dropna()
        if len(daily_ret) > 1 and daily_ret.std() > 0:
            sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
        else:
            sharpe = 0.0

        # 最大回撤
        peak = nav_series.cummax()
        drawdown = (nav_series - peak) / peak
        max_drawdown = float(drawdown.min())
        # 回撤持续天数（从peak到trough）
        is_dd = drawdown < 0
        if is_dd.any():
            # 找最长连续回撤区间
            dd_groups = (~is_dd).cumsum()
            dd_durations = is_dd.groupby(dd_groups).sum()
            max_drawdown_duration = int(dd_durations.max())
        else:
            max_drawdown_duration = 0

        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

        # 资金利用率（平均持仓市值 / 初始资金）
        # 简化：用 1 - 平均现金占比 估算
        # 这里 nav_series 已经是总资产/初始资金，无法直接拆分
        # 用 executed 交易的平均持仓天数 × 投资金额 估算
        if executed:
            total_invested_days = sum(
                pt.trade.hold_days * pt.invested for pt in executed
            )
            capital_utilization = total_invested_days / (initial_capital * n_days)
        else:
            capital_utilization = 0.0

        # 交易级统计
        total_trades = len(executed)
        skipped_count = len(skipped)
        if executed:
            pnls = np.array([pt.pnl for pt in executed])
            win_rate = float((pnls > 0).mean())
            avg_pnl = float(pnls.mean())
            # 换手率：双边交易额 / 平均资产 / 年数
            total_volume = sum(pt.invested * 2 for pt in executed)  # 买入+卖出
            avg_assets = initial_capital * (1 + total_return) / 2  # 粗略平均资产
            turnover_rate = total_volume / avg_assets / years if avg_assets > 0 else 0.0
        else:
            win_rate = 0.0
            avg_pnl = 0.0
            turnover_rate = 0.0

        return cls(
            nav_series=nav_series,
            executed=executed,
            skipped=skipped,
            initial_capital=initial_capital,
            final_value=final_value,
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            sharpe=sharpe,
            calmar=calmar,
            capital_utilization=capital_utilization,
            total_trades=total_trades,
            skipped_count=skipped_count,
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            turnover_rate=turnover_rate,
        )

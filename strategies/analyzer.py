"""
评测层：PerformanceAnalyzer。

职责：接收 Trade 列表 → 计算指标 → 输出 EvalResult / 打印表格 / 绘图。

完全不关心交易怎么产生的（由 TradeSimulator 负责），也不关心信号怎么
来的（由 SignalGenerator 负责）。只做"给定交易记录，算指标"。

用法：
    result = PerformanceAnalyzer.analyze(trades, sell_pct, start, end)
    PerformanceAnalyzer.print_summary(results)    # 多参数对比
    PerformanceAnalyzer.print_result(result)      # 单结果详情
    PerformanceAnalyzer.print_portfolio_result(pf_result)  # 组合级结果
    PerformanceAnalyzer.plot(results, save_path)  # 绘图
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

from strategies.types import EvalResult


class PerformanceAnalyzer:
    """交易记录 → 评测指标。"""

    # ------------------------------------------------------------------
    # 核心分析
    # ------------------------------------------------------------------

    @staticmethod
    def analyze(
        trades: list,
        sell_pct: Optional[float],
        start_date: str,
        end_date: str,
    ) -> EvalResult:
        """从 Trade 列表计算全部指标。"""
        return EvalResult.from_trades(trades, sell_pct, start_date, end_date)

    # ------------------------------------------------------------------
    # 输出：次日出场模式多参数对比表
    # ------------------------------------------------------------------

    @staticmethod
    def print_summary(results: dict[float, EvalResult]) -> None:
        """打印超参数对比表（次日出场模式，sell_pct → EvalResult）。"""
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="策略超参数评测对比", box=box.ROUNDED)

        table.add_column("卖出目标涨幅", justify="center", style="cyan")
        table.add_column("交易次数", justify="right")
        table.add_column("达标率", justify="right", style="green")
        table.add_column("胜率", justify="right")
        table.add_column("均收益(毛)", justify="right")
        table.add_column("均收益(净)", justify="right", style="yellow")
        table.add_column("最大单笔亏损", justify="right", style="red")
        table.add_column("组合年化", justify="right", style="green")
        table.add_column("夏普", justify="right")

        for sell_pct, r in sorted(results.items()):
            if r.total_trades == 0:
                continue
            annual_str = (f"{r.annual_return:+.1%}"
                          if abs(r.annual_return) < 100
                          else f"{r.annual_return:+.0%}")
            table.add_row(
                f"{sell_pct:+.1%}" if sell_pct != 0 else "次日收盘",
                str(r.total_trades),
                f"{r.fill_rate:.1%}",
                f"{r.win_rate:.1%}",
                f"{r.avg_gross_return:+.2%}",
                f"{r.avg_net_return:+.2%}",
                f"{r.max_loss:.2%}",
                annual_str,
                f"{r.sharpe:.2f}",
            )

        console.print(table)

    # ------------------------------------------------------------------
    # 输出：多日止损止盈模式单结果详情
    # ------------------------------------------------------------------

    @staticmethod
    def print_result(result: EvalResult) -> None:
        """打印单个评测结果（多日止损止盈模式）。"""
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        r = result

        if r.total_trades == 0:
            console.print("[red]无交易记录[/red]")
            return

        wins = [t.net_return_pct for t in r.trades if t.net_return_pct > 0]
        losses = [t.net_return_pct for t in r.trades if t.net_return_pct <= 0]
        avg_win = float(pd.Series(wins).mean()) if wins else 0.0
        avg_loss = float(pd.Series(losses).mean()) if losses else 0.0

        exit_counts = r.exit_stats
        exit_str = "  ".join(
            f"{k}:{v}" for k, v in sorted(exit_counts.items(), key=lambda x: -x[1])
        )

        table = Table(title="多日止损止盈评测结果", box=box.ROUNDED)
        table.add_column("指标", style="cyan")
        table.add_column("数值", justify="right")

        rows = [
            ("交易次数",       str(r.total_trades)),
            ("胜率",           f"{r.win_rate:.1%}"),
            ("均净收益",       f"{r.avg_net_return:+.2%}"),
            ("均盈利",         f"{avg_win:+.2%}"),
            ("均亏损",         f"{avg_loss:+.2%}"),
            ("实际盈亏比",     f"{r.actual_rr:.2f}x" if r.actual_rr else "N/A"),
            ("平均风险敞口",   f"{r.avg_risk_pct:.2%}"),
            ("平均持仓天数",   f"{r.avg_hold_days:.1f}天"),
            ("最大单笔亏损",   f"{r.max_loss:.2%}"),
            ("组合年化",       f"{r.annual_return:+.1%}"),
            ("夏普",           f"{r.sharpe:.2f}"),
            ("出场原因分布",   exit_str),
        ]
        for k, v in rows:
            table.add_row(k, v)

        console.print(table)

    # ------------------------------------------------------------------
    # 输出：组合级结果（PortfolioResult）
    # ------------------------------------------------------------------

    @staticmethod
    def print_portfolio_result(result) -> None:
        """打印组合级回测结果（含净值曲线、回撤、资金利用率）。"""
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        r = result

        if r.total_trades == 0 and r.skipped_count == 0:
            console.print("[red]无交易记录[/red]")
            return

        table = Table(title="组合级回测结果（含仓位管理）", box=box.ROUNDED)
        table.add_column("指标", style="cyan")
        table.add_column("数值", justify="right")

        rows = [
            ("初始资金",       f"{r.initial_capital:,.0f}"),
            ("最终市值",       f"{r.final_value:,.0f}"),
            ("总收益率",       f"{r.total_return:+.2%}"),
            ("年化(CAGR)",     f"{r.annual_return:+.2%}"),
            ("最大回撤",       f"{r.max_drawdown:.2%}"),
            ("回撤持续天数",   f"{r.max_drawdown_duration}天"),
            ("夏普",           f"{r.sharpe:.2f}"),
            ("Calmar",         f"{r.calmar:.2f}" if r.calmar else "N/A"),
            ("资金利用率",     f"{r.capital_utilization:.1%}"),
            ("年换手率",       f"{r.turnover_rate:.1%}"),
            ("执行交易数",     str(r.total_trades)),
            ("跳过信号数",     str(r.skipped_count)),
            ("信号执行率",     f"{r.total_trades / (r.total_trades + r.skipped_count):.1%}"
                              if r.total_trades + r.skipped_count > 0 else "N/A"),
            ("胜率",           f"{r.win_rate:.1%}"),
            ("平均盈亏",       f"{r.avg_pnl:+,.0f}"),
        ]
        for k, v in rows:
            table.add_row(k, v)

        console.print(table)

    # ------------------------------------------------------------------
    # 绘图
    # ------------------------------------------------------------------

    @staticmethod
    def plot(
        results: dict[float, EvalResult],
        save_path: Optional[str] = None,
    ) -> None:
        """绘制超参数对比图（次日出场模式）。"""
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        matplotlib.rcParams["font.sans-serif"] = [
            "Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"
        ]
        matplotlib.rcParams["axes.unicode_minus"] = False

        sell_pcts = sorted(results.keys())
        valid = [p for p in sell_pcts if results[p].total_trades > 0]
        if not valid:
            return

        metrics = {
            "均净收益率": [results[p].avg_net_return * 100 for p in valid],
            "胜率": [results[p].win_rate * 100 for p in valid],
            "达标率(限价成交)": [results[p].fill_rate * 100 for p in valid],
            "交易次数": [results[p].total_trades for p in valid],
        }
        xlabels = [f"{p:+.1%}" if p != 0 else "收盘" for p in valid]

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("超参数扫描：卖出目标涨幅 vs 策略表现",
                     fontsize=13, fontweight="bold")

        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
        ylabels = ["%", "%", "%", "次"]

        for ax, (title, vals), color, ylabel in zip(
            axes.flat, metrics.items(), colors, ylabels
        ):
            ax.plot(xlabels, vals, "o-", color=color, linewidth=2, markersize=7)
            ax.fill_between(range(len(valid)), vals, alpha=0.15, color=color)
            ax.set_title(title, fontsize=10)
            ax.set_ylabel(ylabel)
            ax.set_xticks(range(len(valid)))
            ax.set_xticklabels(xlabels, rotation=30, fontsize=8)
            ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda y, _: f"{y:.1f}")
            )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("图表已保存: {}", save_path)
        else:
            plt.show()
        plt.close(fig)

"""
图表生成模块。
生成4张图：净值曲线、回撤水下图、月度收益热力图、交易散点图。
"""

import warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


class Plotter:

    def __init__(self, result):
        self.result = result

    def plot_all(self, save_path: str = None):
        nav = self.result.nav_series
        bm = self.result.benchmark_series
        fills = self.result.fills
        metrics = self.result.metrics

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(
            f"{self.result.strategy_name}  |  {self.result.start_date} ~ {self.result.end_date}",
            fontsize=14, fontweight="bold", y=0.98
        )

        gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :])     # 净值曲线（横跨两列）
        ax2 = fig.add_subplot(gs[1, :])     # 回撤水下图
        ax3 = fig.add_subplot(gs[2, 0])     # 月度热力图
        ax4 = fig.add_subplot(gs[2, 1])     # 交易散点图

        self._plot_nav(ax1, nav, bm, metrics)
        self._plot_drawdown(ax2, nav)
        self._plot_monthly_heatmap(ax3, nav)
        self._plot_trade_scatter(ax4, fills)

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"图表已保存: {save_path}")
        else:
            plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------

    def _plot_nav(self, ax, nav: pd.Series, bm: pd.Series, metrics: dict):
        ax.plot(nav.index, nav.values, color="#2196F3", linewidth=1.5, label="策略净值")

        if bm is not None and not bm.empty:
            # 基准对齐到策略的时间范围
            bm_aligned = bm.reindex(nav.index, method="ffill")
            ax.plot(bm_aligned.index, bm_aligned.values, color="#FF9800",
                    linewidth=1.2, linestyle="--", alpha=0.8, label="基准")

        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title("净值曲线", fontsize=11)
        ax.set_ylabel("净值")

        # 指标文本框
        info = (
            f"总收益: {metrics.get('total_return', 0):.2%}  "
            f"年化: {metrics.get('annual_return', 0):.2%}  "
            f"夏普: {metrics.get('sharpe_ratio', 0):.3f}  "
            f"最大回撤: {metrics.get('max_drawdown', 0):.2%}"
        )
        ax.text(0.01, 0.02, info, transform=ax.transAxes,
                fontsize=8.5, color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", alpha=0.8))

        ax.legend(loc="upper left", fontsize=9)
        self._format_date_axis(ax, nav.index)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}"))

    def _plot_drawdown(self, ax, nav: pd.Series):
        from report.metrics import PerformanceMetrics
        dd = PerformanceMetrics.drawdown_series(nav)

        ax.fill_between(dd.index, dd.values, 0, color="#F44336", alpha=0.4, label="回撤")
        ax.plot(dd.index, dd.values, color="#F44336", linewidth=0.8)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_title("回撤（水下图）", fontsize=11)
        ax.set_ylabel("回撤幅度")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1%}"))
        self._format_date_axis(ax, dd.index)

    def _plot_monthly_heatmap(self, ax, nav: pd.Series):
        """月度收益热力图"""
        if len(nav) < 20:
            ax.set_title("月度收益（数据不足）")
            return

        # 计算月度收益
        nav.index = pd.to_datetime(nav.index)
        monthly = nav.resample("ME").last().pct_change().dropna()

        pivot = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "ret": monthly.values,
        }).pivot(index="year", columns="month", values="ret")

        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot.columns = [month_labels[m - 1] for m in pivot.columns]

        vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 0.01)
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=7)
        ax.set_title("月度收益热力图", fontsize=11)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                            fontsize=6, color="black")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     format=mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))

    def _plot_trade_scatter(self, ax, fills):
        """每笔交易：持有天数 vs 收益率"""
        sell_fills = [f for f in fills if f.direction == "sell"]

        if not sell_fills:
            ax.set_title("交易分布（暂无成交）")
            return

        pnls = [f.pnl for f in sell_fills]
        holding = []
        for f in sell_fills:
            try:
                from datetime import date
                d1 = date.fromisoformat(f.order_date)
                d2 = date.fromisoformat(f.trade_date)
                holding.append((d2 - d1).days)
            except Exception:
                holding.append(0)

        colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnls]
        ax.scatter(holding, pnls, c=colors, alpha=0.6, s=20, edgecolors="none")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("持有天数")
        ax.set_ylabel("净盈亏（元）")
        ax.set_title(f"交易分布（共{len(sell_fills)}笔）", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:,.0f}"))

    @staticmethod
    def _format_date_axis(ax, index):
        """简化日期轴显示"""
        n = len(index)
        step = max(1, n // 8)
        ticks = range(0, n, step)
        ax.set_xticks([list(index)[i] for i in ticks])
        ax.set_xticklabels([list(index)[i] for i in ticks], rotation=30, fontsize=7)

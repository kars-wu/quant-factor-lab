"""
BacktestEngine：低频量化回测主引擎。

执行流程：
  Phase 1 [向量化] 数据加载 + 技术指标预计算
  Phase 2 [顺序]  逐日模拟：T日信号 → T+1开盘成交
  Phase 3 [向量化] 绩效计算 + 图表生成

防未来函数保障：
  ctx.data[symbol] = df.loc[:current_date]（截至当日收盘的历史切片）
  订单放入 pending，下一日开盘成交
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from loguru import logger

from config.settings import (
    DEFAULT_BENCHMARK,
    DEFAULT_COMMISSION_RATE,
    DEFAULT_SLIPPAGE_RATE,
    DEFAULT_STAMP_DUTY_RATE,
)
from data.database import Database
from data.loader import DataLoader
from engine.broker import Broker, Order
from engine.context import BarContext
from engine.portfolio import Fill, Portfolio


@dataclass
class BacktestResult:
    run_id: str
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    nav_series: pd.Series               # index=trade_date, values=净值（初始=1.0）
    benchmark_series: pd.Series         # 基准净值序列
    fills: list[Fill]
    metrics: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    def print_summary(self):
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=f"回测结果 — {self.strategy_name}", box=box.ROUNDED)
        table.add_column("指标", style="cyan", no_wrap=True)
        table.add_column("策略", justify="right", style="green")
        table.add_column("基准(买持)", justify="right", style="yellow")

        bm_return = float(self.benchmark_series.iloc[-1] - 1) if len(self.benchmark_series) > 0 else 0

        rows = [
            ("总收益率", f"{self.metrics.get('total_return', 0):.2%}", f"{bm_return:.2%}"),
            ("年化收益率", f"{self.metrics.get('annual_return', 0):.2%}", "—"),
            ("年化波动率", f"{self.metrics.get('annual_volatility', 0):.2%}", "—"),
            ("夏普比率", f"{self.metrics.get('sharpe_ratio', 0):.3f}", "—"),
            ("最大回撤", f"{self.metrics.get('max_drawdown', 0):.2%}", "—"),
            ("卡玛比率", f"{self.metrics.get('calmar_ratio', 0):.3f}", "—"),
            ("胜率", f"{self.metrics.get('win_rate', 0):.2%}", "—"),
            ("盈亏比", f"{self.metrics.get('profit_factor', 0):.2f}", "—"),
            ("总交易次数", str(self.metrics.get('total_trades', 0)), "—"),
            ("平均持仓天数", f"{self.metrics.get('avg_holding_days', 0):.1f}", "—"),
        ]
        for r in rows:
            table.add_row(*r)
        console.print(table)

    def plot(self, save_path: Optional[str] = None):
        """绘制净值曲线、回撤图、月度热力图"""
        from report.plotter import Plotter
        p = Plotter(self)
        p.plot_all(save_path=save_path)

    def export_trades(self, path: Optional[str] = None) -> pd.DataFrame:
        """导出逐笔交易记录"""
        if not self.fills:
            return pd.DataFrame()
        records = [
            {
                "trade_date": f.trade_date,
                "order_date": f.order_date,
                "symbol": f.symbol,
                "direction": f.direction,
                "price": f.price,
                "quantity": f.quantity,
                "commission": f.commission,
                "stamp_duty": f.stamp_duty,
                "pnl": f.pnl,
            }
            for f in self.fills
        ]
        df = pd.DataFrame(records)
        if path:
            df.to_csv(path, index=False, encoding="utf-8-sig")
        return df


class BacktestEngine:

    def __init__(
        self,
        strategy,
        db: Optional[Database] = None,
        initial_capital: float = 1_000_000.0,
        commission_rate: float = DEFAULT_COMMISSION_RATE,
        stamp_duty_rate: float = DEFAULT_STAMP_DUTY_RATE,
        slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
        benchmark: str = DEFAULT_BENCHMARK,
        adjust: str = "qfq",
        freq: str = "D",
    ):
        self.strategy = strategy
        self.db = db or Database()
        self.loader = DataLoader(self.db)
        self.broker = Broker()
        self.initial_capital = initial_capital
        self.benchmark = benchmark
        self.adjust = adjust
        self.freq = freq

    def run(
        self,
        start_date: str,
        end_date: str,
        universe: list[str],
    ) -> BacktestResult:
        """
        执行回测。

        start_date / end_date : 回测区间（含两端）
        universe              : 标的代码列表
        """
        run_id = str(uuid.uuid4())[:8]
        logger.info(
            "回测开始 [{}] {} | {} ~ {} | 标的:{} | 策略:{}",
            run_id, self.strategy.name, start_date, end_date,
            len(universe), self.strategy.name
        )

        # ── Phase 1: 数据加载 ──────────────────────────────────────────
        logger.info("  加载行情数据...")
        data = self.loader.load_ohlcv(
            universe, start_date, end_date,
            adjust=self.adjust, freq=self.freq,
        )
        if not data:
            raise RuntimeError("没有有效的行情数据，请先初始化数据库")

        benchmark_raw = self.loader.load_benchmark(self.benchmark, start_date, end_date)

        trade_dates = self.loader.get_trade_dates(start_date, end_date)
        if not trade_dates:
            raise RuntimeError("无交易日数据，请确认交易日历已初始化")

        # ── Phase 2: 策略预计算（可选） ───────────────────────────────
        logger.info("  策略预计算...")
        self.strategy.prepare(data)

        # ── Phase 3: 逐日回测循环 ─────────────────────────────────────
        logger.info("  开始逐日回测（共{}个交易日）...", len(trade_dates))
        portfolio = Portfolio(self.initial_capital)
        pending_orders: list[Order] = []

        # 用于获取次日开盘价（不复权）的原始数据
        raw_data = self.loader.load_ohlcv(
            universe, start_date, end_date, adjust="", freq=self.freq
        )

        for i, date in enumerate(trade_dates):
            close_prices: dict[str, float] = {}
            for symbol, df in data.items():
                if date in df.index:
                    close_prices[symbol] = float(df.loc[date, "close"])

            # 3.1 处理昨日挂单（T+1 开盘成交）
            if pending_orders and i > 0:
                for order in pending_orders:
                    symbol = order.symbol
                    raw_df = raw_data.get(symbol)
                    if raw_df is None or date not in raw_df.index:
                        logger.debug("  {} 在 {} 无行情，订单取消", symbol, date)
                        continue
                    next_open = float(raw_df.loc[date, "open"])
                    self.broker.execute(order, next_open, date, portfolio)
                pending_orders.clear()

            # 3.2 构建上下文（截至当日数据切片，防未来函数）
            sliced_data = {}
            for symbol, df in data.items():
                # 取 <= current_date 的行，df.index 为字符串，直接比较
                mask = df.index <= date
                sliced_data[symbol] = df.loc[mask]

            ctx = BarContext(
                current_date=date,
                data=sliced_data,
                portfolio=portfolio.get_readonly_view(close_prices),
                universe=[s for s in universe if s in data],
            )

            # 3.3 调用策略
            try:
                new_orders = self.strategy.on_bar(ctx)
            except Exception as e:
                logger.error("策略 on_bar 异常 [{}]: {}", date, e)
                new_orders = []

            # 验证订单格式
            for order in (new_orders or []):
                if isinstance(order, Order) and order.quantity > 0:
                    order.order_date = date
                    pending_orders.append(order)

            # 3.4 按收盘价快照组合市值
            portfolio.snapshot(date, close_prices)

        # ── Phase 4: 绩效计算 ──────────────────────────────────────────
        nav_series = portfolio.get_nav_series()
        nav_series = nav_series[nav_series.index >= start_date]

        # 基准净值（买持归一化）
        bm_series = self._calc_benchmark_nav(benchmark_raw, start_date, end_date)

        from report.metrics import PerformanceMetrics
        metrics = PerformanceMetrics.calculate(nav_series, bm_series)
        metrics.update(PerformanceMetrics.trade_stats(portfolio.get_fills()))

        logger.info(
            "回测完成 [{}] 总收益:{:.2%} 夏普:{:.3f} 最大回撤:{:.2%}",
            run_id,
            metrics.get("total_return", 0),
            metrics.get("sharpe_ratio", 0),
            metrics.get("max_drawdown", 0),
        )

        return BacktestResult(
            run_id=run_id,
            strategy_name=self.strategy.name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            nav_series=nav_series,
            benchmark_series=bm_series,
            fills=portfolio.get_fills(),
            metrics=metrics,
            params=getattr(self.strategy, "params", {}),
        )

    def _calc_benchmark_nav(
        self,
        benchmark_df: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        """将基准收盘价序列归一化为净值（初始=1.0）"""
        if benchmark_df.empty:
            return pd.Series(dtype=float)
        df = benchmark_df[benchmark_df.index >= start_date]
        df = df[df.index <= end_date]
        if df.empty:
            return pd.Series(dtype=float)
        close = df["close"].astype(float)
        return close / close.iloc[0]

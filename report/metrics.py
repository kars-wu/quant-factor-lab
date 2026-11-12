"""
绩效指标计算（全部向量化）。
"""

import numpy as np
import pandas as pd


class PerformanceMetrics:

    @staticmethod
    def calculate(
        nav: pd.Series,
        benchmark: pd.Series = None,
        risk_free_rate: float = 0.03,
        trading_days_per_year: int = 252,
    ) -> dict:
        """
        计算全量绩效指标。

        nav       : 净值序列（初始值=1.0），index 为日期字符串
        benchmark : 基准净值序列，可为 None
        """
        if nav is None or len(nav) < 2:
            return {}

        nav = nav.sort_index().astype(float)
        daily_returns = nav.pct_change().dropna()

        if len(daily_returns) == 0:
            return {}

        # 基础统计
        total_days = len(nav)
        years = total_days / trading_days_per_year
        total_return = float(nav.iloc[-1] / nav.iloc[0] - 1)
        annual_return = float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0

        # 波动率
        annual_vol = float(daily_returns.std() * np.sqrt(trading_days_per_year))

        # 夏普比率
        daily_rf = risk_free_rate / trading_days_per_year
        excess_returns = daily_returns - daily_rf
        sharpe = float(excess_returns.mean() / excess_returns.std() * np.sqrt(trading_days_per_year)) if excess_returns.std() > 0 else 0.0

        # 索提诺比率（只考虑下行波动）
        downside = daily_returns[daily_returns < daily_rf]
        downside_std = float(downside.std() * np.sqrt(trading_days_per_year)) if len(downside) > 1 else 1e-6
        sortino = float((annual_return - risk_free_rate) / downside_std) if downside_std > 0 else 0.0

        # 最大回撤
        rolling_max = nav.cummax()
        drawdown_series = (nav - rolling_max) / rolling_max
        max_drawdown = float(drawdown_series.min())

        # 最大回撤持续天数
        in_drawdown = drawdown_series < 0
        max_dd_duration = 0
        current_duration = 0
        for in_dd in in_drawdown:
            if in_dd:
                current_duration += 1
                max_dd_duration = max(max_dd_duration, current_duration)
            else:
                current_duration = 0

        # 卡玛比率
        calmar = float(annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0

        # 基准相关指标
        alpha, beta, information_ratio = 0.0, 0.0, 0.0
        if benchmark is not None and len(benchmark) >= 2:
            bm_returns = benchmark.sort_index().astype(float).pct_change().dropna()
            # 对齐
            common_idx = daily_returns.index.intersection(bm_returns.index)
            if len(common_idx) >= 2:
                r = daily_returns.loc[common_idx]
                rb = bm_returns.loc[common_idx]
                cov = np.cov(r, rb)
                beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else 0.0
                alpha = float(annual_return - risk_free_rate - beta * (
                    (1 + rb.mean()) ** trading_days_per_year - 1 - risk_free_rate
                ))
                active_returns = r - rb
                if active_returns.std() > 0:
                    information_ratio = float(
                        active_returns.mean() / active_returns.std() * np.sqrt(trading_days_per_year)
                    )

        metrics = {
            "total_return": round(total_return, 6),
            "annual_return": round(annual_return, 6),
            "annual_volatility": round(annual_vol, 6),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "calmar_ratio": round(calmar, 4),
            "max_drawdown": round(max_drawdown, 6),
            "max_drawdown_duration": max_dd_duration,
            "alpha": round(alpha, 6),
            "beta": round(beta, 4),
            "information_ratio": round(information_ratio, 4),
        }

        return metrics

    @staticmethod
    def trade_stats(fills) -> dict:
        """从成交记录计算胜率、盈亏比等交易统计"""
        if not fills:
            return {
                "total_trades": 0, "win_rate": 0.0,
                "profit_factor": 0.0, "avg_holding_days": 0.0,
            }

        sell_fills = [f for f in fills if f.direction == "sell"]
        if not sell_fills:
            return {
                "total_trades": len(fills), "win_rate": 0.0,
                "profit_factor": 0.0, "avg_holding_days": 0.0,
            }

        pnls = [f.pnl for f in sell_fills]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_rate = len(wins) / len(pnls) if pnls else 0.0
        profit_factor = (
            sum(wins) / abs(sum(losses))
            if losses and sum(losses) != 0
            else float("inf")
        )

        # 平均持仓天数（买入成交日 → 卖出成交日，按标的顺序配对）
        from datetime import date as _date
        buy_fills = [f for f in fills if f.direction == "buy"]
        # 建立每个标的的买入队列 {symbol: [buy_trade_date, ...]}
        buy_queue: dict = {}
        for f in buy_fills:
            buy_queue.setdefault(f.symbol, []).append(f.trade_date)

        holding_days = []
        used_buys: dict = {s: 0 for s in buy_queue}  # 每个标的已配对的买入索引
        for f in sell_fills:
            s = f.symbol
            idx = used_buys.get(s, 0)
            buys = buy_queue.get(s, [])
            if idx < len(buys):
                try:
                    d1 = _date.fromisoformat(buys[idx])
                    d2 = _date.fromisoformat(f.trade_date)
                    holding_days.append((d2 - d1).days)
                    used_buys[s] = idx + 1
                except Exception:
                    pass
        avg_holding = float(np.mean(holding_days)) if holding_days else 0.0

        return {
            "total_trades": len(sell_fills),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "avg_holding_days": round(avg_holding, 1),
        }

    @staticmethod
    def drawdown_series(nav: pd.Series) -> pd.Series:
        """返回每日回撤序列（用于可视化水下图）"""
        rolling_max = nav.cummax()
        return (nav - rolling_max) / rolling_max

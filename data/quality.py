"""
数据质量校验管道。

━━━ 背景 ━━━

策略最常见的死法之一是数据污染。
某团队数据提供商将价格除以100（日元错标美元），系统以为暴跌99%疯狂做多。

脏数据（价格为0/负数、OHLC逻辑矛盾、缺失日期、复权因子异常）会
直接进入信号计算产生虚假信号。本模块在 SignalGenerator.load_data() 后
自动运行，检测并报告数据质量问题。

━━━ 检查项 ━━━

  1. OHLC 逻辑：high >= max(open, close), low <= min(open, close)
  2. 价格范围：价格 <= 0 或日变动 > 50%（异常跳变）
  3. 缺失日期：与交易日历对比，检测缺失的交易日
  4. 成交量异常：volume = 0 或 > 均量的 20 倍
  5. 复权因子单调性：hfq_factor 应非递减
  6. 停牌检测：成交额 = 0 且 volume = 0 的日期

用法：
    from data.quality import DataQualityChecker

    checker = DataQualityChecker()
    report = checker.check_symbol(df, symbol, calendar_dates)
    if report.has_issues:
        logger.warning("数据质量问题: {}", report.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class Issue:
    """单个数据质量问题。"""
    check: str          # 检查项名称
    date: str           # 问题日期
    detail: str         # 问题描述
    severity: str = 'warning'   # 'warning' | 'error'


@dataclass
class DataQualityReport:
    """单只股票的数据质量报告。"""
    symbol: str
    issues: list[Issue] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == 'error')

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == 'warning')

    def summary(self) -> str:
        return (f"{self.symbol}: {self.error_count} 个错误, "
                f"{self.warning_count} 个警告")


class DataQualityChecker:
    """数据质量校验器。"""

    def __init__(
        self,
        max_daily_change: float = 0.50,      # 日变动超过50%视为异常
        volume_anomaly_ratio: float = 20.0,   # 成交量超过均量20倍视为异常
    ):
        self._max_daily_change = max_daily_change
        self._volume_anomaly_ratio = volume_anomaly_ratio

    def check_symbol(
        self,
        df: pd.DataFrame,
        symbol: str,
        calendar_dates: Optional[set] = None,
    ) -> DataQualityReport:
        """
        对单只股票的 DataFrame 运行所有检查。

        参数
        ----
        df             : 行情数据，index=trade_date，含 open/high/low/close/volume
        symbol         : 股票代码
        calendar_dates : 交易日历日期集合（用于检测缺失日期），可选
        """
        report = DataQualityReport(symbol=symbol)

        if df.empty:
            report.issues.append(Issue('empty', '', '数据为空', 'error'))
            return report

        # 1. OHLC 逻辑校验
        self._check_ohlc_logic(df, report)

        # 2. 价格范围 + 异常跳变
        self._check_price_range(df, report)

        # 3. 缺失日期
        if calendar_dates:
            self._check_missing_dates(df, calendar_dates, report)

        # 4. 成交量异常
        self._check_volume_anomaly(df, report)

        # 5. 复权因子单调性
        if 'hfq_factor' in df.columns:
            self._check_adj_factor(df, report)

        return report

    def check_batch(
        self,
        data: dict[str, pd.DataFrame],
        calendar_dates: Optional[set] = None,
        max_report: int = 50,
    ) -> list[DataQualityReport]:
        """
        批量检查多只股票，只返回有问题的报告。

        参数
        ----
        max_report : 最多报告多少只股票的问题（避免日志刷屏）
        """
        reports_with_issues = []
        for symbol, df in data.items():
            report = self.check_symbol(df, symbol, calendar_dates)
            if report.has_issues:
                reports_with_issues.append(report)
                if len(reports_with_issues) >= max_report:
                    break

        if reports_with_issues:
            total_errors = sum(r.error_count for r in reports_with_issues)
            total_warnings = sum(r.warning_count for r in reports_with_issues)
            logger.warning(
                "数据质量检查完成 | {} 只有问题标的 / {} 只总计 | "
                "{} 个错误, {} 个警告",
                len(reports_with_issues), len(data),
                total_errors, total_warnings,
            )
            # 打印前5个问题的详情
            for r in reports_with_issues[:5]:
                logger.warning("  {} → {}", r.symbol, r.summary())
                for issue in r.issues[:3]:
                    logger.warning("    {} {}: {}", issue.date, issue.check, issue.detail)

        return reports_with_issues

    # ------------------------------------------------------------------
    # 具体检查方法
    # ------------------------------------------------------------------

    def _check_ohlc_logic(self, df: pd.DataFrame, report: DataQualityReport):
        """high >= max(open, close), low <= min(open, close)"""
        for col in ['open', 'high', 'low', 'close']:
            if col not in df.columns:
                report.issues.append(Issue('missing_column', '', f'缺少 {col} 列', 'error'))
                return

        # 向量化检查
        max_oc = df[['open', 'close']].max(axis=1)
        min_oc = df[['open', 'close']].min(axis=1)

        bad_high = df['high'] < max_oc
        bad_low = df['low'] > min_oc
        bad_hl = df['high'] < df['low']

        for date in df.index[bad_high]:
            report.issues.append(Issue(
                'ohlc_logic', str(date.date()) if hasattr(date, 'date') else str(date),
                f"high({df.loc[date,'high']:.2f}) < max(open,close)({max_oc.loc[date]:.2f})",
                'error',
            ))
        for date in df.index[bad_low]:
            report.issues.append(Issue(
                'ohlc_logic', str(date.date()) if hasattr(date, 'date') else str(date),
                f"low({df.loc[date,'low']:.2f}) > min(open,close)({min_oc.loc[date]:.2f})",
                'error',
            ))
        for date in df.index[bad_hl]:
            report.issues.append(Issue(
                'ohlc_logic', str(date.date()) if hasattr(date, 'date') else str(date),
                f"high({df.loc[date,'high']:.2f}) < low({df.loc[date,'low']:.2f})",
                'error',
            ))

    def _check_price_range(self, df: pd.DataFrame, report: DataQualityReport):
        """价格 <= 0 或日变动异常"""
        for col in ['open', 'high', 'low', 'close']:
            bad = df[col] <= 0
            for date in df.index[bad]:
                report.issues.append(Issue(
                    'price_range', str(date.date()) if hasattr(date, 'date') else str(date),
                    f"{col}={df.loc[date, col]} <= 0",
                    'error',
                ))

        # 日变动超过阈值
        daily_ret = df['close'].pct_change().abs()
        abnormal = daily_ret > self._max_daily_change
        for date in df.index[abnormal]:
            if not df.loc[date, 'close'] or pd.isna(daily_ret.loc[date]):
                continue
            report.issues.append(Issue(
                'price_jump', str(date.date()) if hasattr(date, 'date') else str(date),
                f"日变动 {daily_ret.loc[date]:.1%} > {self._max_daily_change:.0%}",
                'warning',
            ))

    def _check_missing_dates(
        self,
        df: pd.DataFrame,
        calendar_dates: set,
        report: DataQualityReport,
    ):
        """对比交易日历，检测缺失的交易日"""
        df_dates = {str(d.date()) if hasattr(d, 'date') else str(d) for d in df.index}
        # 只检查 df 日期范围内的交易日
        min_date = min(df_dates)
        max_date = max(df_dates)
        expected = {d for d in calendar_dates if min_date <= d <= max_date}
        missing = expected - df_dates

        if missing:
            missing_sorted = sorted(missing)
            # 只报告前10个缺失日期
            for date in missing_sorted[:10]:
                report.issues.append(Issue(
                    'missing_date', date, '交易日数据缺失', 'warning',
                ))
            if len(missing_sorted) > 10:
                report.issues.append(Issue(
                    'missing_date', '', f'共 {len(missing_sorted)} 个交易日缺失（仅显示前10个）',
                    'warning',
                ))

    def _check_volume_anomaly(self, df: pd.DataFrame, report: DataQualityReport):
        """成交量 = 0 或异常放大"""
        if 'volume' not in df.columns:
            return

        # 成交量 = 0（可能停牌）
        zero_vol = (df['volume'] == 0) | df['volume'].isna()
        for date in df.index[zero_vol]:
            report.issues.append(Issue(
                'zero_volume', str(date.date()) if hasattr(date, 'date') else str(date),
                '成交量为0（可能停牌）', 'warning',
            ))

        # 成交量异常放大
        vol_ma = df['volume'].rolling(20, min_periods=5).mean()
        vol_ratio = df['volume'] / vol_ma.replace(0, np.nan)
        abnormal = vol_ratio > self._volume_anomaly_ratio
        for date in df.index[abnormal]:
            if pd.isna(vol_ratio.loc[date]):
                continue
            report.issues.append(Issue(
                'volume_anomaly', str(date.date()) if hasattr(date, 'date') else str(date),
                f"成交量 {vol_ratio.loc[date]:.1f}x > {self._volume_anomaly_ratio}x 均量",
                'warning',
            ))

    def _check_adj_factor(self, df: pd.DataFrame, report: DataQualityReport):
        """复权因子应非递减"""
        hfq = df['hfq_factor'].dropna()
        if len(hfq) < 2:
            return

        # 检测递减点（允许浮点微小波动 < 1e-3，AkShare 因子有已知微小波动）
        diff = hfq.diff()
        decreasing = diff < -1e-3
        for date in hfq.index[decreasing]:
            report.issues.append(Issue(
                'adj_factor', str(date.date()) if hasattr(date, 'date') else str(date),
                f"复权因子下降 {diff.loc[date]:.6f}",
                'warning',
            ))

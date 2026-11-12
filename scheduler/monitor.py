"""
监控告警 + 异常汇总。

━━━ 背景 ━━━

监控分四层：基础设施 → 服务 → 应用 → 业务。
数据更新失败只 logger.error 不通知用户，长时间任务无异常汇总。

━━━ 功能 ━━━

1. 摘要报告：任务结束后统计成功/失败/跳过数
2. Webhook 通知：飞书/钉钉/Slack（可选）
3. 异常汇总：回测运行中信号计算失败的标的汇总

━━━ 用法 ━━━

    from scheduler.monitor import TaskMonitor

    monitor = TaskMonitor(name="每日增量更新")
    # ... 执行任务 ...
    monitor.record_success("600519")
    monitor.record_failure("000001", "API超时")
    monitor.record_skip("ST股票", "ST标记")

    report = monitor.finish()
    print(report.summary())
    # 可选：发送 Webhook 通知
    monitor.notify(webhook_url="https://...")
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import requests
from loguru import logger


@dataclass
class TaskMonitor:
    """任务执行监控器。"""
    name: str
    start_time: float = field(default_factory=time.time)
    successes: list = field(default_factory=list)   # 成功的标的/操作
    failures: list = field(default_factory=list)     # (标的, 原因)
    skips: list = field(default_factory=list)        # (标的, 原因)
    end_time: float = 0.0

    def record_success(self, item: str):
        self.successes.append(item)

    def record_failure(self, item: str, reason: str):
        self.failures.append((item, reason))

    def record_skip(self, item: str, reason: str):
        self.skips.append((item, reason))

    def finish(self) -> 'TaskReport':
        self.end_time = time.time()
        return TaskReport(
            name=self.name,
            n_success=len(self.successes),
            n_failure=len(self.failures),
            n_skip=len(self.skips),
            duration=self.end_time - self.start_time,
            failures=self.failures[:20],    # 只保留前20个
            skips=self.skips[:20],
        )

    def notify(self, webhook_url: str = None, report: 'TaskReport' = None):
        """发送 Webhook 通知（飞书/钉钉/Slack）。"""
        if not webhook_url:
            return
        if report is None:
            report = self.finish()

        # 从环境变量读取 webhook URL
        url = webhook_url or os.getenv('MONITOR_WEBHOOK_URL', '')
        if not url:
            return

        try:
            text = report.summary()
            # 飞书/钉钉通用格式
            payload = {
                "msg_type": "text",
                "content": {"text": text},
            }
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info("Webhook 通知已发送")
            else:
                logger.warning("Webhook 通知失败: HTTP {}", resp.status_code)
        except Exception as e:
            logger.warning("Webhook 通知异常: {}", e)


@dataclass
class TaskReport:
    """任务执行摘要报告。"""
    name: str
    n_success: int
    n_failure: int
    n_skip: int
    duration: float
    failures: list = field(default_factory=list)
    skips: list = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.n_success + self.n_failure + self.n_skip

    @property
    def success_rate(self) -> float:
        return self.n_success / self.total if self.total > 0 else 0

    def summary(self) -> str:
        lines = [
            f"━━━ {self.name} 摘要 ━━━",
            f"成功: {self.n_success} / 失败: {self.n_failure} / 跳过: {self.n_skip}",
            f"成功率: {self.success_rate:.1%}",
            f"耗时: {self.duration:.0f}秒 ({self.duration/60:.1f}分钟)",
        ]
        if self.failures:
            lines.append(f"\n失败详情（前{len(self.failures)}个）:")
            for item, reason in self.failures[:10]:
                lines.append(f"  {item}: {reason}")
        if self.skips:
            lines.append(f"\n跳过详情（前{len(self.skips)}个）:")
            for item, reason in self.skips[:5]:
                lines.append(f"  {item}: {reason}")
        return '\n'.join(lines)

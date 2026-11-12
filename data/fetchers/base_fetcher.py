from __future__ import annotations

import time
import threading
from abc import ABC, abstractmethod

import pandas as pd
from loguru import logger

from config.settings import AKSHARE_RPM


class RateLimiter:
    """
    令牌桶限速器（线程安全）。
    默认 50 RPM，即每次请求间隔至少 1.2 秒。
    """

    def __init__(self, rpm: int = AKSHARE_RPM):
        self._interval = 60.0 / max(rpm, 1)
        self._last_call = 0.0
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            wait = self._interval - elapsed
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()


class BaseFetcher(ABC):
    """数据源抽象基类"""

    def __init__(self, rate_limiter: RateLimiter | None = None):
        self.rate_limiter = rate_limiter or RateLimiter()

    def _throttle(self):
        self.rate_limiter.acquire()

    def _retry(self, func, *args, retries: int = 3, backoff: float = 2.0, **kwargs):
        """指数退避重试"""
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == retries - 1:
                    raise
                wait = backoff ** attempt
                logger.warning("请求失败（第{}次），{:.1f}s 后重试: {}", attempt + 1, wait, e)
                time.sleep(wait)

    @abstractmethod
    def fetch_stock_list(self) -> pd.DataFrame:
        """返回 DataFrame，列：[symbol, name, market, asset_type, list_date]"""
        ...

    @abstractmethod
    def fetch_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        返回不复权日线数据，列：[trade_date, open, high, low, close, volume, amount]
        trade_date 格式：'YYYY-MM-DD'
        """
        ...

    @abstractmethod
    def fetch_hfq_factor(self, symbol: str) -> pd.DataFrame:
        """
        返回后复权因子，列：[trade_date, hfq_factor]
        hfq_factor 定义：上市首日=1.0，后续随除权单调不减
        """
        ...

    def fetch_hfq_latest(self, symbol: str) -> pd.DataFrame:
        """返回最新后复权收盘价，默认回退到全量因子接口。"""
        return self.fetch_hfq_factor(symbol)

    @abstractmethod
    def fetch_index_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """返回指数/ETF 日线，列：[trade_date, open, high, low, close, volume, amount]"""
        ...

    @abstractmethod
    def fetch_trade_calendar(self, start_year: int = 2010) -> pd.DataFrame:
        """返回交易日历，列：[trade_date, is_open]"""
        ...

    def fetch_valuation(self, symbol: str) -> pd.DataFrame:
        """
        返回个股估值数据（总市值 + PB），百度接口。

        返回列：[trade_date, total_mv, pb]
        total_mv 单位：亿元
        pb 无量纲

        默认实现调用 AkShare 的 stock_zh_valuation_baidu，
        子类可覆盖。未实现时返回空 DataFrame。
        """
        return pd.DataFrame()

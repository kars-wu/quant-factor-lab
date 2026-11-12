import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent

# 数据库路径
DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "data" / "stock.db"))

# AkShare 限速（每分钟请求数，保守设置留余量）
AKSHARE_RPM = int(os.getenv("AKSHARE_RPM", "50"))

# Tushare token（备用数据源）
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")

# 回测默认参数
DEFAULT_COMMISSION_RATE = 0.0001   # 万一（买卖双向）
DEFAULT_STAMP_DUTY_RATE = 0.001    # 千一（仅卖出）
DEFAULT_SLIPPAGE_RATE = 0.0        # 竞价策略按开盘价成交，不额外计滑点
DEFAULT_MIN_COMMISSION = 5.0       # 最低佣金 5 元

# 数据历史起始日期
DEFAULT_START_DATE = "2015-01-01"

# 默认基准
DEFAULT_BENCHMARK = "000300"  # 沪深300

# 数据加载时额外预留的历史窗口（用于技术指标热身）
MAX_LOOKBACK_DAYS = 250

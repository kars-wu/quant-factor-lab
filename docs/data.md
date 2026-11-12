# 数据接入

## 1. 存储

单文件 SQLite（默认 `data/stock.db`，可用 `DB_PATH` 覆盖）：

- WAL 模式 + 线程本地连接：更新任务写库不阻塞研究/模拟盘读库
- 初始化时自动执行 `data/schema.sql` 并做幂等迁移
- 研究侧**不要**直接用 `data.database.Database`（构造会写库），用 `multi_agent/datasets/sqlite_view.py`

核心表：

| 表 | 内容 | 关键注意 |
|---|---|---|
| `stock_info` | 标的主数据、上市/退市、is_active | `is_active` 是"当前"状态，回溯历史会带幸存者偏差 |
| `stock_daily` | 未复权 OHLCV、换手率、PE/PB、市值 | 历史行永不修改 |
| `adj_factor` | 后复权因子（单调不减） | 前复权价 = 原始价 × hfq(t) / hfq(锚点) |
| `index_daily` | 指数/ETF 日线 | 用于择时与基准 |
| `trade_calendar` | 交易日历 | 所有日期运算以此为准 |
| `stock_shares` | 股本 | 换手/市值兜底 |
| `stock_valuation` | 总市值、PB、PE | 日期是日历日（含周末），查询需 ffill 到交易日 |
| `financial_report` | 三大报表 | 必须按 `announce_date <= T` 切片，否则是未来函数 |
| `update_log` / `data_quality_log` | 更新状态与质量问题 | 重试与审计入口 |

## 2. 初始化

```bash
# 首次全量（股票列表 + 日历 + 日线 + 复权因子 + 股本 + 估值），按网络状况可能数十分钟起
python -m scheduler.daily_update --init-all

# 每日增量（盘后）
python -m scheduler.daily_update --incremental
python -m scheduler.daily_update fetch-shares
python -m scheduler.daily_update fetch-valuation

# 质量检查
python -c "from data.database import Database; from data.quality import DataQualityChecker as C; print(C(Database()).run_all())"
```

数据源默认 akshare，限速见 `AKSHARE_RPM`（默认 50 次/分钟）。
`data/choice_client.py` 是一个带**周配额守卫**的 Choice 包装器：
请求发出前先在 SQLite 里预占额度，失败也记账——适合有调用量上限的付费源。
它需要一个已登录的 `EmQuantAPI.c` 对象，仓库不内置该 SDK 与账号。

## 3. 复权口径

```text
hfq_price(t) = raw_price(t) × hfq_factor(t)
qfq_price(t) = raw_price(t) × hfq_factor(t) / hfq_factor(anchor)
```

- 信号/因子用**前复权**（锚点取数据末端），保证价格连续
- 真实资金会计应该用**未复权价 + 当时股数 + 公司行为**；本仓库的 `engine/`/`simulation/`
  只做简化处理，没有完整账本
- 成交量不复权

## 4. 股票池

`multi_agent/datasets/sqlite_view.load_sqlite_view()`：

```python
universe='top_n_amount'   # 过去 lookback 日平均成交额 top_n，shift(1) —— 不看当天
universe='all'            # 全部有行情的标的
non_st_mask=...           # 可选：PIT 的非 ST 掩码（(date × symbol) bool）
```

**必须自己补的两件事**：

1. **ST / 风险警示**：用当前名单过滤历史 = 幸存者偏差。正确做法是从历史公告或
   逐日板块成分构造 PIT 掩码，传 `non_st_mask`
2. **退市与停牌**：缺行情不等于可以静默删除；`mask=False` 保留标的，
   让"卖不出"变成显式的执行约束
3. **行业分类**：分组中性化算子用 `industry_code` 面板。`load_sqlite_view` 默认从
   `stock_info.industry` 生成编码——这是**当前**分类套到历史上，属于轻度未来信息。
   严格研究请换成按日期生效的行业变更表，或自己构造 `industry_code` 面板注入

## 5. 只读约束

```python
from multi_agent.datasets.sqlite_view import readonly_conn, load_sqlite_view

with readonly_conn('data/stock.db') as conn:      # file:...?mode=ro
    ...
view = load_sqlite_view('data/stock.db', start='2018-01-01', universe='top_n_amount')
```

研究进程只应拿到 `DataView`。注意：`mode=ro` 只保证不写库，
**不阻止读其他路径**；真正的保留段隔离要靠容器/权限，见 `docs/multi-agent.md` 第 7 节。

## 6. 标定检查

换数据源或改 schema 后，先跑这三条：

```bash
pytest tests/test_data_quality.py -q          # OHLC/复权/缺日等基础校验
pytest tests/test_simulation_data.py -q       # 池子不看当天成交额、基准同口径
python -m multi_agent.cli validate --file examples/hypotheses.sample.jsonl
```

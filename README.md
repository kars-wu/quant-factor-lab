# quant-factor-lab

A 股日频量化研究工具包：**因子研究 → 组合回测 → 模拟盘跟踪**，外加一套**多 agent 因子挖掘与研究治理**框架。

目标不是再提供一堆"有效因子"，而是把一条**可复现、可审计**的研究流水线开源出来：
数据、表达式、执行、成本、评估口径全部显式化，任何一个结论都能追到代码和工件。

```text
数据(SQLite) → 因子表达式(受限 DSL) → 截面合成/打分 → T+1 开盘成交 → 组合回测
                                     ↘ Gate0 工程不变量 → 预算/台账/失败记忆 → 外层策略进化
```

## ⚠️ 先说清楚三件事

1. **仓库里没有任何挖掘出来的因子**。开源版本只保留教科书级示例因子（`strategies/factor_zoo.py`）
   和两个演示策略（`simulation/examples.py`）。原先研究中筛出的因子家族、参数组合、
   实验报告与一次性脚本均已移除。
2. **示例策略和合成数据的回测数字没有意义**。它们只用来验证"信号 → 成交 → 净值 → 台账"这条链路是通的，
   不能据此判断任何市场规律。
3. **不构成投资建议**。回测是日线近似，不含盘口排队、集合竞价成交明细与真实冲击模型；
   上实盘前必须自己补执行与风控。

## 能做什么

| 模块 | 能力 |
|---|---|
| `data/` | SQLite 行情库（WAL、增量更新、复权因子、数据质量检查、可选 Choice 配额守卫） |
| `strategies/` | 因子加载/IC/分层/中性化、多因子合成（IC 加权、Ridge）、走查验证、组合管理、止损、成本模型 |
| `engine/` | 事件驱动回测骨架（context / broker / portfolio） |
| `simulation/` | 日频模拟盘：T+1 开盘成交、半仓滚动、持久化状态、FastAPI + 网页工作台 |
| `multi_agent/` | 六角色研究闭环 + 受限 DSL + Gate0 不变量 + SQLite 台账 + 预算 + 失败记忆 + GEPA 式外层进化 |

## 快速开始

```bash
git clone https://github.com/kars-wu/quant-factor-lab.git
cd quant-factor-lab
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 1) 多 agent：不需要任何数据，直接跑合成数据端到端演示
python -m multi_agent.cli demo

# 2) 多 agent：校验你自己的假设文件（DSL / 字段 / 窗口 / 时间安全）
python -m multi_agent.cli validate --file examples/hypotheses.sample.jsonl

# 3) 模拟盘：先准备行情库，再回填与启动
python -m scheduler.daily_update --init-all          # 首次全量（较慢，按需）
python -m simulation.run bootstrap --start 2021-01-01
python -m simulation.run status
uvicorn simulation.api:app --host 127.0.0.1 --port 8787   # 打开 http://127.0.0.1:8787
```

`python -m multi_agent.cli demo` 的输出示例（合成数据，数字无意义）：

```text
cand_5d3c1c3fc6ef466b medium_term_momentum  pass  gate=ok  sharpe=1.827  excess=0.775  decision=keep
cand_67864ac4b66d24ad short_term_reversal   reject gate=ok sharpe=0.671  excess=-3.513 decision=reject
```

## 多 agent 怎么用

```bash
# 合成数据小规模试跑（离线、秒级）
python -m multi_agent.cli demo --symbols 200 --dates 800 --selection-top 20 --hold-days 20

# 强制"算子交互"设计：≥3 类算子 + 必须含 cs_rank / ts_corr + 禁止根部线性叠加
python -m multi_agent.cli demo --design-strict

# 接自己的 SQLite 行情库（只读打开，不会写库）
python -m multi_agent.cli run --dataset sqlite --db data/stock.db \
    --start 2018-01-01 --universe top_n_amount --top-n 500 --selection-top 50 --hold-days 20

# 留出最终保留段：dev 阶段只看到 dev，保留段只由 validator 揭盲一次
python -m multi_agent.cli run --dataset sqlite --holdout-start 2025-01-01

# 外层策略进化（配对评测，只改研究流程，不改接受门槛）
python -m multi_agent.cli evolve --rounds 3
```

要接真实 LLM，填 `.env`（任何兼容 `/chat/completions` 的服务）：

```bash
MA_LLM_BASE_URL=https://api.openai.com/v1
MA_LLM_API_KEY=sk-...
MA_LLM_MODEL=gpt-4.1-mini
```

不配置时六种角色走**确定性启发式**：流程、闸门、预算、台账、去重全部照常工作，
只是不会自动提出新机制。这条路径的存在是为了让仓库能离线跑通、能被单测覆盖。

**想把已经挖过的家族挡在门外**：把关键词写进 `multi_agent/banned_families.txt`
（每行一个，命中即拒绝），这是"剔除已挖因子"在开源版本里的可配置实现。

因子生成的角色 prompt 在 `multi_agent/prompts/02_researcher.md`：它沿用了
"强制算子交互 + 防未来函数 + 去共线性 + 向量化"这一套硬约束，但把输出从 Python 函数
改成了 DSL 表达式（因此不可执行任意代码），并把"全样本 Z-Score"改成了逐日截面标准化
（前者是隐性未来函数）。算子池对照表见 `prompts/operator_pool.md`。

## 回测口径（默认）

- 信号：T 日收盘生成，只使用 T 及以前的数据
- 成交：T+1 开盘买；持有 `hold_days` 个交易日；T+1+hold 开盘卖
- 资金：每日新开一笔 `1/hold_days` 资金的 tranche，tranche 内等权；未投入部分按现金计（收益 0，不 dropna）
- 成本：默认双边 18bp，压力情景 50bp；成本 = 换手 × 费率，换手按**新开仓名义权重**计
- 基准：同一股票池、同一 open-to-open 口径的等权组合
- 选股：`selection_top` 若超过当日可选标的的 20%，按 20% 截断（避免"买整池"被当成选股）

这些口径写在 `multi_agent/contracts.py` 与 `multi_agent/evaluation/portfolio.py` 里，
不是文档里的口头约定。

## Gate0：看收益之前先过工程闸门

```
dsl_valid / complexity / warmup        表达式合法、规模受限
prefix_invariance                      删掉 T 之后的数据，T 及以前的因子值不能变
cross_section_invariance               打乱列顺序，排序结果必须一致
zero_weight_invariance                 加一个零权重稀疏因子，合成值与候选池不能变
coverage                               覆盖率、是否系统性偏向大市值
cash_identity                          现金+持仓=净值，空仓日必须保留
```

`multi_agent/evaluation/gates.py::audit_regression_suite` 里有一组**已知缺陷夹具**，
用来度量"审计链路本身是否真的能发现问题"——外层进化用的也是这套指标。

## 目录

```text
config/         配置（DB 路径、费率、环境变量）
data/           SQLite 连接、schema、增量更新、复权、质量检查、akshare/Choice 抓取
indicators/     通用技术指标（trend / oscillator / volatility）
strategies/     因子分析、合成、策略时序、组合管理、成本、止损、走查
engine/         事件驱动回测骨架
risk/           仓位与止损
report/         绩效指标、绘图、结果存储
scheduler/      每日数据更新与任务监控
simulation/     模拟盘（数据层、引擎、状态、API、网页、示例策略）
multi_agent/    多 agent 研究系统（契约/DSL/数据视图/评估/角色/编排/进化）
docs/           架构、数据接入、多 agent 设计说明
examples/       示例假设文件
tests/          单测（111 个）
```

## 已知限制

- 日线近似执行：无排队、无盘口吃单、无集合竞价量；涨停买不进的过滤只是开盘价近似
- `simulation` 的股票池是代理池（成交额 top-N / 流通市值 top-300），不是真实指数成分
- 复权价格用于信号，真实资金会计应使用未复权价与当时股数（本仓库的 `engine/` 与 `simulation/` 只做了简化处理）
- 外层进化只优化**过程指标**（缺陷检出、误杀率），不对应收益提升；它不能替代独立样本外验证

## 测试

```bash
pip install -r requirements-dev.txt
pytest            # 111 passed
```

## License

MIT

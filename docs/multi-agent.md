# 多 agent 因子研究系统

一句话：**把"研究纪律"写成代码，而不是写在 prompt 里。**

Prompt 可以被模型忽略，代码不能。所以这个子系统的结构是：
模型只负责提机制、写表达式、解释结果；**能不能晋级由确定性程序决定**。

## 1. 六个角色

| 角色 | 输入 | 交付 | 边界 |
|---|---|---|---|
| Research Lead | 目标、协议、失败记忆、预算 | `research_plan`：研究流与配额 | 只能提议，预算由编排器锁定 |
| Researcher × 2（反转流 / 动量流） | 各自研究流、算子目录、记忆 | `hypothesis_cards`：机制 + 可否证预测 + 表达式草案 | 不读保留段；不跑网格；两条流互不通信 |
| Factor Engineer | 假设卡 | `candidate_spec`：冻结的 AST + 方向 | 只能拼装 DSL，不能写 Python |
| Data & Execution Auditor | 候选 + Gate0 结果 | `audit_report`：硬失败/待核对/分类 | 可否决；不能放宽门槛 |
| Portfolio Analyst | 指标 + 基准 + 压力情景 | `decision_record`：keep/reject/watch/repair | 不能主观覆盖硬门槛 |
| Evolution Coach | 轨迹 + 失败归因 | `policy_mutation`：单模块改动 | 不能改数据/撮合/接受标准 |

两条研究者流各有自己的 prompt：`prompts/02_researcher.md`（公共硬约束）+
`02a_reversal_researcher.md` / `02b_continuation_researcher.md`（研究范围与自查项）。
组合方式见 `agents/prompts.py::system`，`researcher_role()` 负责 track → 角色名映射。

两个"服务"是纯程序：

- **Runner / Evaluator**：编译、Gate0、tranche 回测、成本压力、统计判定
- **Registry**：不可覆盖的试验台账 + 预算事件 + 访问事件 + 失败记忆

## 2. 契约（`contracts.py`）

一个候选 = `FactorSpec + direction + DataSpec + ExecutionSpec + EvaluationSpec`。

- `CandidateSpec.key()` 只由决定结果的字段构成 → 用来去重
- `CandidateSpec.fingerprint()` 加上 run / hypothesis / policy / prompt / code hash / seed → 用来追溯血缘
- 任何一项变化都是**新候选**，不能就地改口径后复用旧的通过记录

`key` 只在同一个 run 内唯一：换 run 重跑同一公式属于"复现"，允许但要标记
（`replicated_from`），因为修 bug 后重跑是合法需求。

## 3. DSL（`dsl.py`）

```python
{"op": "ts_pct_change", "n": 5, "inputs": [{"field": "close"}]}
```

- 算子白名单：
  - 算术：`add / sub / mul / div / neg / abs / log / sign / where_pos / inverse`
  - 截面：`cs_rank / cs_zscore / cs_demean / cs_regress_residual / cs_scale / cs_winsorize`
  - 分组中性化：`cs_group_demean / cs_group_rank / cs_regress_multi`（`by` 指向
    `industry_code` 等分组字段，`cs_regress_multi` 可混合行业哑变量与 `mv` 等连续风格）
  - 时序统计：`ts_rolling(mean/std/sum/max/min/median/skew/kurt) / ts_quantile`
  - 高级时序：`ts_shift / ts_delta / ts_pct_change / ts_rank / ts_slope /
    decay_linear / ts_zscore / ts_argmax / ts_argmin / ts_corr / ts_cov`
  - 条件：`where(cond, x)`（当日截面筛选）、`trade_when(trigger, x)`（路径依赖，谨慎）
  - `ts_slope` / `decay_linear` 用前缀和实现（O(n)）；`ts_argmax/argmin` 用 stride view
    分块计算，三者都要求整窗有效才输出
- 窗口：`1..250`，正整数；`shift` 只向过去
- 字段分 `close`（T 收盘可得）与 `pit_fundamental`（需上游按公告日对齐），
  `check_time_safety` 会提示
- `structure_key` 忽略窗口数值 → 同结构仅改窗口算同一 family
- 复杂度上限：32 节点、深度 8、warmup ≤ 250

**DSL 挡的是显式越界，不挡"看起来合法但经济上无意义"的表达式**，
后者靠审计与统计门槛。

### 3.1 设计要求（可选硬门槛）

`dsl.design_report()` 检查三件事：

| 规则 | 判定 |
|---|---|
| 算子类别 ≥ 3 | 类别：arithmetic / cross_section / ts_stat / ts_dynamic |
| 必须含相对算子 | `cs_rank` / `cs_zscore` / `cs_scale` / `ts_corr` / `ts_cov` 至少一个 |
| 禁止根部线性叠加 | 根为 `add`/`sub` 且两个子树 `complexity ≥ 3` |

默认只在 Gate0 里**记录**（`design_richness` 恒 True，detail 里给建议）；
加 `--design-strict` 或 `EvaluationSpec.design_strict=True` 后变成硬门槛。
启发式模板里有两条满足该规则（量价背离反转、量价确认动量），其余会被挡下——
这正是严格模式该有的效果：**简单表达式不配进入评估**。

### 3.2 两个容易写错的点

1. **不要用全样本统计量做标准化**。`(x - x.mean()) / x.std()` 在整段历史上算均值方差，
   会把未来信息带进 T 日；要标准化用 `cs_zscore`（逐日截面）。
2. **缩尾也在截面层做**，不要写进表达式。评估器已按日 `cs_rank`，对极端值稳健。

## 4. 状态机与台账

```text
PROPOSED → REGISTERED → IMPLEMENTED → AUDITED
        → TRAIN_EVALUATED → DEV_VALIDATED → FROZEN
        → FINAL_VALIDATED → SHADOW_TRACKED
任意节点可转 REJECTED / NEEDS_DATA / REPAIR_REQUIRED
```

SQLite 表：`research_runs / hypotheses / candidates / attempts / evaluations /
decisions / memories / access_events / budget_events`。

产物按 `outputs/multi_agent/<run_id>/<candidate_id>/{spec,gate,audit,metrics,decision}.json`
落盘，不可覆盖；重试用新的 `attempt_id`。

## 5. 预算

```python
Budget(max_proposals=24, max_variants=24, max_train_evaluations=48,
       max_full_validations=8, max_llm_calls=400)
```

- 跨 agent、跨轮次、跨 prompt 版本累计，不因换角色或换版本重置
- 每次扣减都写 `budget_events`，报告里可对账
- 重复候选、实现失败、Gate0 失败**都算消耗**——否则"多试几次总能中"就成立了

## 6. 晋级闸门

**Gate0（工程不变量，看收益之前）**

| 检查 | 触发的问题 |
|---|---|
| `prefix_invariance` | 用到了 T 之后的数据 |
| `cross_section_invariance` | 依赖列顺序/批次 |
| `zero_weight_invariance` | 零权重稀疏项改变候选池 |
| `coverage` | 覆盖率过低 / 系统性偏大市值 |
| `cash_identity` | 日序列缺天、净值恒等式不成立 |

**开发评估**：`excess_sharpe ≥ 0.3`、`active_annual_return ≥ 0`、
`max_drawdown ≥ -45%`、逐年正超额占比 ≥ 60%、覆盖率 ≥ 60%。

**成本压力**：成本抬到 50bp 后结论翻转 → 决策降为 reject（记录为 `cost_stress`）。

**保留段**：只有开发阶段 keep 的候选，才由 validator 揭盲一次；
失败后该段降级为开发历史，下一轮换新的保留段。

这些阈值都在 `EvaluationSpec` 里，开跑前固定，不因结果修改。
想调阈值请新建 `EvaluationSpec`（会体现在 hash 里）。

> 未显式传 `--dev-start` 时，开发窗口默认从数据的前 35% 处开始——
> 前面那段留给因子 warmup（最长 250 个交易日），不参与统计。
> 真实研究建议显式指定，并把 `--holdout-start` 留给最后揭盲。

## 7. 访问隔离的现实边界

仓库提供的是**登记与结构**，不是物理隔离：

- `datasets/sqlite_view.py` 用 `mode=ro` 打开数据库，且不经过会写库的 `Database` 类
- 研究进程只接收 `DataView`（内存宽表），开发窗口与保留窗口是两次不同 `slice`
- 所有保留段读取写 `access_events`

但**同一 OS 用户的进程仍可读绝对路径**，所以"独立保留隔离"需要你自己用容器 /
子进程 / 权限做到。没做隔离时，报告里必须标"隔离未建立"，不能声称独立验证。

## 8. 外层进化（`evolution.py`）

```text
轨迹 → 归因失败 → 选一个模块 → 最小改动 → 配对评测（同视图/同种子/同预算）→ 接受或回滚
```

可进化模块：

- `proposal`：研究流顺序、提案配额
- `audit_checklist`：启用哪些 Gate0 检查（默认全开；缺项时优先补）
- `budget_allocation`：配额分配

评分只看**过程指标**：`detection`（能否抓住已知缺陷夹具）、
`false_positive`（会不会误杀干净候选）、`check_cost`。
`score = detection - 0.5 × false_positive`，提升超过 0.05 且误杀不增加才接受。

**这不是"改 prompt 让因子更赚钱"**。收益证据只能来自冻结后的新数据。

## 9. 换真实 LLM

```bash
export MA_LLM_BASE_URL=https://api.openai.com/v1
export MA_LLM_API_KEY=sk-...
export MA_LLM_MODEL=gpt-4.1-mini
```

角色 prompt 在 `multi_agent/prompts/*.md`，输出要求是单个 JSON 对象
（`artifact_type / status / summary / evidence_refs / payload`）。

模型输出非法 JSON、表达式过不了 DSL、字段越界时，**自动回退到启发式**并记账；
`gate` 失败时模型也不能把结论改成通过（`roles.audit` 里做了强制覆盖）。

## 10. 接入你已经挖过的家族

把关键词写进 `multi_agent/banned_families.txt`：

```text
# 每行一个，命中即拒绝
short_term_reversal_5d
dry_volume
notice_share
```

命中判定看 `family + 表达式文本 + 机制描述`；被拦下的提案会计入
`duplicate_family` 记忆，下次检索时会提醒模型别再提。

这是开源版本里"剔除已挖因子"的落点：**清单不进仓库，机制进仓库**。

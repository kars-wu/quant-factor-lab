# 算子池

DSL 只允许下表算子。写表达式时按"类别"思考，但落笔必须是 DSL 名。

| 类别 | 常用说法 | DSL 算子 | 说明 |
|---|---|---|---|
| 基础算术 | add / sub / mul / div / inverse | `add` `sub` `mul` `div` `neg` `inverse` | `inverse` 即 `1/x`，分母 0 → NaN |
| | abs / log / sign | `abs` `log` `sign` `where_pos` | `log` 只对正值生效，`where_pos` 先屏蔽非正值 |
| 横截面 | rank | `cs_rank` | 逐日截面百分位排名，`[0,1]` |
| | zscore / scale | `cs_zscore` `cs_scale` | 逐日截面标准化 / 除以当日绝对值之和 |
| | demean | `cs_demean` | 逐日截面去均值 |
| | neutralize | `cs_regress_residual` | 逐日截面回归取残差，`y` 在前、`x` 在后 |
| | 行业中性 | `cs_group_demean(by='industry_code')` `cs_group_rank(by=...)` | 逐日组内去均值 / 组内排名 |
| | 多风格中性 | `cs_regress_multi(by=['industry_code','mv'])` | 行业哑变量 + 连续风格一起回归取残差，需要 ≥10 只有效标的 |
| | winsorize | `cs_winsorize(q)` | 逐日截面缩尾，默认 `q=0.01` |
| 时序统计 | ts_mean / ts_std / ts_max / ts_min | `ts_rolling(n, agg=mean/std/max/min)` | 另有 `sum/median/skew/kurt` |
| | ts_quantile | `ts_quantile(n, q)` | 过去 n 日分位值，`0<q<1` |
| 高级时序 | ts_slope | `ts_slope(n)` | 过去 n 日对时间回归的斜率（单位/日） |
| | decay_linear | `decay_linear(n)` | 线性衰减加权均值，近端权重高 |
| | ts_zscore | `ts_zscore(n)` | 当前值在过去 n 日的 z 分数 |
| | delay / delta / pct_change | `ts_shift` `ts_delta` `ts_pct_change` | 只接受正窗口 |
| | ts_rank | `ts_rank(n)` | 当前值在过去 n 日中的分位 |
| | ts_argmax / ts_argmin | `ts_argmax(n)` `ts_argmin(n)` | 极值在窗口内的相对时刻，1=今天，0=n-1 天前 |
| | ts_corr / ts_cov | `ts_corr` `ts_cov(n)` | 双输入 + 窗口，量价背离常用 |
| 条件 | where | `where(cond, x)` | `cond>0` 才保留，否则缺失（当日截面筛选） |
| | trade_when | `trade_when(trigger, x)` | `trigger>0` 时更新信号，否则沿用上一值（**路径依赖**） |
| 组合 | 加权求和 | `add` `sub` | **根节点这样用会被设计规则拦下**（线性叠加） |

## 设计规则（`dsl.design_report`）

| 规则 | 判定 | 说明 |
|---|---|---|
| 算子类别 ≥ 3 | 统计去重后的类别数 | 类别见上表 |
| 必须含相对算子 | 出现 `cs_rank` / `cs_zscore` / `cs_scale` / `ts_corr` / `ts_cov` 之一 | 避免纯绝对量因子 |
| 禁止根部线性叠加 | 根为 `add`/`sub` 且两个子树 `complexity ≥ 3` | 用 `mul`/`div`/`ts_corr` 表达交互 |

默认只在报告里提示；加 `--design-strict`（或 `EvaluationSpec.design_strict=True`）后变成硬门槛。

## 规模限制

| 项目 | 上限 |
|---|---|
| 单窗口 `n` | 1..250（`ts_rolling` ≥ 2，`ts_rank`/`ts_slope`/`ts_zscore`/`ts_quantile` ≥ 5，`ts_corr/cov` ≥ 10） |
| 节点数 | 32 |
| 嵌套深度 | 8 |
| `warmup` | 250 个交易日 |
| `min_periods` | `max(2, n//2)`（窗口的 50%） |

## 三个容易写错的地方

1. **不要用全样本统计量做标准化**：`(x - x.mean()) / x.std()` 在整段历史上算均值方差
   会把未来信息带进 T 日。要标准化请用 `cs_zscore`（逐日截面），
   时序标准化用 `ts_zscore`（只看过去 n 日）。
2. **缩尾也在截面层做**：表达式里不要写 `clip(pct=0.01)`，评估器已按日做 `cs_rank`，
   并对极端值稳健；如需额外缩尾用 `cs_winsorize`，不要用全样本分位。
3. **`trade_when` 不是"状态择时"许可证**：它只允许"用 T 日已知的触发条件决定是否更新信号"，
   不能用它把历史上表现差的区间剔掉。用了它就要在假设里写清楚触发条件的金融含义。

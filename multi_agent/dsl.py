"""因子 DSL：唯一允许的算子与字段白名单。

为什么要有 DSL：让 agent 只能「拼装」已审计的算子，而不是生成任意 Python。
表达式树形如::

    {"op": "ts_pct_change", "n": 5, "inputs": [{"field": "close"}]}

约束：

* 时间序列算子只接受正整数窗口 ``1 <= n <= 250``；
* ``shift`` 只允许向过去取数，因此不存在用未来数据的写法；
* 每个字段声明 ``available`` 时点，编译前会检查「因子可用时点 <= 信号时点」；
* 结构相同的表达式（仅窗口不同）会被归入同一 family，用于去重与预算计数。

DSL 只能挡住显式越界，不能替代数据与执行审计 —— 见 evaluation/gates.py。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

MAX_WINDOW = 250
MAX_DEPTH = 8
MAX_NODES = 32

# ─────────────────────────── 字段白名单 ───────────────────────────

CLOSE_FIELDS = [
    'open', 'high', 'low', 'close', 'volume', 'amount',
    'turnover_rate', 'mv', 'pb', 'pe_ttm',
]

FUNDAMENTAL_FIELDS = [
    'net_profit_ttm', 'operating_cf_ttm', 'total_assets', 'revenue',
    'equity', 'interest_bearing_debt', 'accrual_ratio', 'leverage_ratio',
]

# 分类/分组字段：只能作为 by 参数传给分组中性化算子，不能当连续值参与运算
GROUP_FIELDS = ['industry_code']

# available='close'  → 当日收盘后可用（可作为 T 日信号）
# available='pit_fundamental' → 需上游按公告日对齐后再注入面板
FIELD_AVAILABILITY: Dict[str, str] = {
    **{f: 'close' for f in CLOSE_FIELDS},
    **{f: 'pit_fundamental' for f in FUNDAMENTAL_FIELDS},
    **{f: 'static' for f in GROUP_FIELDS},
}

PANEL_FIELDS = CLOSE_FIELDS + FUNDAMENTAL_FIELDS


# ─────────────────────────── 算子实现 ───────────────────────────

def _check_window(n: Any, lo: int = 1) -> int:
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError(f'窗口必须是整数，收到 {n!r}')
    if not (lo <= n <= MAX_WINDOW):
        raise ValueError(f'窗口 {n} 超出范围 [{lo}, {MAX_WINDOW}]')
    return n


def _ts_shift(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.shift(_check_window(n))


def _ts_delta(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x - x.shift(_check_window(n))


def _ts_pct_change(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x / x.shift(_check_window(n)) - 1


def _ts_rolling(x: pd.DataFrame, n: int, agg: str) -> pd.DataFrame:
    n = _check_window(n, lo=2)
    roll = x.rolling(n, min_periods=max(2, n // 2))
    if agg not in ('mean', 'std', 'sum', 'max', 'min', 'median', 'skew', 'kurt'):
        raise ValueError(f'不支持的聚合方式: {agg}')
    return getattr(roll, agg)()


def _ts_rank(x: pd.DataFrame, n: int) -> pd.DataFrame:
    n = _check_window(n, lo=5)
    return x.rolling(n, min_periods=max(5, n // 2)).apply(
        lambda w: pd.Series(w).rank(pct=True).iloc[-1], raw=False
    )


def _ts_corr(a: pd.DataFrame, b: pd.DataFrame, n: int) -> pd.DataFrame:
    n = _check_window(n, lo=10)
    return a.rolling(n, min_periods=max(10, n // 2)).corr(b)


def _ts_cov(a: pd.DataFrame, b: pd.DataFrame, n: int) -> pd.DataFrame:
    n = _check_window(n, lo=10)
    return a.rolling(n, min_periods=max(10, n // 2)).cov(b)


def _cs_rank(x: pd.DataFrame) -> pd.DataFrame:
    return x.rank(axis=1, pct=True)


def _cs_zscore(x: pd.DataFrame) -> pd.DataFrame:
    mu = x.mean(axis=1)
    sd = x.std(axis=1).replace(0, np.nan)
    return x.sub(mu, axis=0).div(sd, axis=0)


def _cs_demean(x: pd.DataFrame) -> pd.DataFrame:
    return x.sub(x.mean(axis=1), axis=0)


def _window_sums(vals: np.ndarray, n: int):
    """返回窗口内的 Σx、Σk·x（k=0..n-1）与有效计数，全部 O(n) 前缀和实现。

    缺失值按 0 求和，靠有效计数筛选窗口；只有整窗有效才输出，避免半窗偏差。
    """
    rows, cols = vals.shape
    valid = ~np.isnan(vals)
    x = np.where(valid, vals, 0.0)
    pos = np.arange(rows, dtype=float)[:, None]

    def _wsum(arr: np.ndarray) -> np.ndarray:
        cum = np.cumsum(arr, axis=0)
        out = np.empty((rows - n + 1, cols), dtype=float)
        out[0] = cum[n - 1]
        out[1:] = cum[n:] - cum[:-n]
        return out

    sum_x = _wsum(x)
    sum_kx = _wsum(x * pos) - (np.arange(rows - n + 1, dtype=float)[:, None]) * sum_x
    count = _wsum(valid.astype(float))
    return sum_x, sum_kx, count


def _fill_window_result(vals: np.ndarray, n: int, result: np.ndarray,
                        count: np.ndarray) -> pd.DataFrame:
    out = np.full(vals.shape, np.nan)
    ok = count >= n
    block = np.where(ok, result, np.nan)
    out[n - 1:] = block
    return pd.DataFrame(out, index=None)


def _ts_slope(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """滚动线性回归的斜率（对交易日序号 k=0..n-1 回归），单位：每天。"""
    n = _check_window(n, lo=5)
    vals = x.to_numpy(dtype=float, copy=False)
    rows, cols = vals.shape
    if rows < n:
        return pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    sum_x, sum_kx, count = _window_sums(vals, n)
    t_sum = n * (n - 1) / 2.0
    t_sq = (n - 1) * n * (2 * n - 1) / 6.0
    denom = n * t_sq - t_sum ** 2
    slope = (n * sum_kx - t_sum * sum_x) / denom
    out = _fill_window_result(vals, n, slope, count)
    out.index = x.index
    out.columns = x.columns
    return out


def _decay_linear(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """线性衰减加权均值：近端权重高（权重 1..n），整窗有效才输出。"""
    n = _check_window(n, lo=2)
    vals = x.to_numpy(dtype=float, copy=False)
    rows, cols = vals.shape
    if rows < n:
        return pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    sum_x, sum_kx, count = _window_sums(vals, n)
    weighted = (sum_kx + sum_x) / (n * (n + 1) / 2.0)
    out = _fill_window_result(vals, n, weighted, count)
    out.index = x.index
    out.columns = x.columns
    return out


def _ts_quantile(x: pd.DataFrame, n: int, q: float) -> pd.DataFrame:
    n = _check_window(n, lo=5)
    if not isinstance(q, (int, float)) or not (0.0 < float(q) < 1.0):
        raise ValueError(f'分位参数 q 必须在 (0,1) 内，收到 {q!r}')
    return x.rolling(n, min_periods=max(2, n // 2)).quantile(float(q))


def _ts_zscore(x: pd.DataFrame, n: int) -> pd.DataFrame:
    n = _check_window(n, lo=5)
    roll = x.rolling(n, min_periods=max(2, n // 2))
    return (x - roll.mean()) / roll.std().replace(0, np.nan)


def _cs_winsorize(x: pd.DataFrame, q: float) -> pd.DataFrame:
    """逐日截面缩尾：把当日分布的两端各截掉 q（默认 1%）。"""
    q = float(q)
    if not (0.0 <= q < 0.5):
        raise ValueError(f'缩尾比例必须在 [0, 0.5) 内，收到 {q}')
    if q == 0:
        return x
    lo = x.quantile(q, axis=1)
    hi = x.quantile(1 - q, axis=1)
    return x.clip(lower=lo, upper=hi, axis=0)


def _cs_scale(x: pd.DataFrame) -> pd.DataFrame:
    """逐日截面缩放：除以当日绝对值之和（保留符号）。"""
    denom = x.abs().sum(axis=1).replace(0, np.nan)
    return x.div(denom, axis=0)


def _inverse(x: pd.DataFrame) -> pd.DataFrame:
    return 1.0 / x.replace(0, np.nan)


def _ts_extreme_pos(x: pd.DataFrame, n: int, mode: str) -> pd.DataFrame:
    """窗口内极值位置的相对时刻：0 = 极值在 n-1 天前，1 = 极值就在今天。

    分块用 stride view + argmax 计算，不复制整块窗口；整窗有效才输出。
    """
    n = _check_window(n, lo=5)
    vals = x.to_numpy(dtype=float, copy=False)
    rows, cols = vals.shape
    out = np.full((rows, cols), np.nan)
    if rows < n:
        return pd.DataFrame(out, index=x.index, columns=x.columns)

    valid = ~np.isnan(vals)
    cum = np.cumsum(valid.astype(np.int64), axis=0)
    zero = np.zeros((1, cols), dtype=np.int64)
    full = (cum[n - 1:] - np.vstack([zero, cum[:-n]])) == n

    chunk = max(1, int(64e6 / max(1, rows * n * 8)))
    for start in range(0, cols, chunk):
        stop = min(cols, start + chunk)
        block = vals[:, start:stop]
        windows = np.lib.stride_tricks.sliding_window_view(block, n, axis=0)
        idx = windows.argmax(axis=2) if mode == 'max' else windows.argmin(axis=2)
        pos = idx.astype(float) / (n - 1)
        out[n - 1:, start:stop] = np.where(full[:, start:stop], pos, np.nan)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def _group_matrix(groups: pd.DataFrame, x: pd.DataFrame) -> np.ndarray:
    g = groups.reindex(index=x.index, columns=x.columns).to_numpy(dtype=float)
    if np.isnan(g).all():
        raise ValueError('分组字段全为缺失，无法做分组中性化')
    return g


def _cs_group_demean(x: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """逐日按组去均值（行业内中性化）。"""
    vals = x.to_numpy(dtype=float, copy=False)
    g = _group_matrix(groups, x)
    out = np.full_like(vals, np.nan)
    n_groups = int(np.nanmax(g)) + 1
    for i in range(vals.shape[0]):
        row, gi = vals[i], g[i]
        ok = np.isfinite(row) & np.isfinite(gi)
        if not ok.any():
            continue
        codes = gi[ok].astype(int)
        sums = np.bincount(codes, weights=row[ok], minlength=n_groups)
        counts = np.bincount(codes, minlength=n_groups)
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        out[i, ok] = row[ok] - means[codes]
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def _cs_group_rank(x: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """逐日组内百分位排名（组内选相对强弱，而不是全市场排名）。"""
    vals = x.to_numpy(dtype=float, copy=False)
    g = _group_matrix(groups, x)
    out = np.full_like(vals, np.nan)
    for i in range(vals.shape[0]):
        ok = np.isfinite(vals[i]) & np.isfinite(g[i])
        if not ok.any():
            continue
        series = pd.Series(vals[i][ok])
        keys = g[i][ok].astype(int)
        ranks = series.groupby(keys).rank(pct=True)
        out[i, np.flatnonzero(ok)] = ranks.to_numpy()
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def _cs_regress_multi(x: pd.DataFrame, panels: Dict[str, pd.DataFrame],
                      by: List[str]) -> pd.DataFrame:
    """逐日截面回归取残差：行业哑变量 + 连续风格（如 mv / pb / 波动）。

    ``by`` 里的分组字段按哑变量处理（丢弃第一档），其余字段按连续值处理。
    """
    vals = x.to_numpy(dtype=float, copy=False)
    rows, cols = vals.shape
    group_cols = [f for f in by if f in GROUP_FIELDS]
    cont_cols = [f for f in by if f not in GROUP_FIELDS]

    g_mats = {f: _group_matrix(panels[f], x) for f in group_cols}
    c_mats = {f: panels[f].reindex(index=x.index, columns=x.columns)
              .to_numpy(dtype=float, copy=False) for f in cont_cols}

    out = np.full_like(vals, np.nan)
    for i in range(rows):
        y = vals[i]
        mask = np.isfinite(y)
        for mat in list(c_mats.values()):
            mask &= np.isfinite(mat[i])
        if mask.sum() < 10:
            continue
        cols_design = [np.ones(mask.sum())]
        for f, mat in g_mats.items():
            codes = mat[i][mask]
            uniq = np.unique(codes[np.isfinite(codes)])
            for level in uniq[1:]:
                cols_design.append((codes == level).astype(float))
        cont = [mat[i][mask] for mat in c_mats.values()]
        if cont:
            block = np.column_stack(cont)
            # 市值/价格量级差异极大，先按当日截面标准化再回归，避免病态矩阵
            mu = block.mean(axis=0)
            sd = block.std(axis=0)
            sd[sd == 0] = 1.0
            block = (block - mu) / sd
            cols_design.extend(block[:, k] for k in range(block.shape[1]))
        design = np.column_stack(cols_design)
        y_ok = y[mask]
        beta, *_ = np.linalg.lstsq(design, y_ok, rcond=None)
        resid = y_ok - design @ beta
        out[i, mask] = resid
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def _where(cond: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    """条件取值：cond > 0 时保留 x，否则置为缺失（纯截面筛选，不看未来）。"""
    return x.where(cond > 0)


def _trade_when(trigger: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    """trigger > 0 时更新信号，否则沿用最近一次的值（前向填充）。

    注意：这会引入路径依赖。trigger 必须只用 T 日及以前可知的信息，
    并且不能用它来"事后挑出表现好的区间"。
    """
    return x.where(trigger > 0).ffill()


def cs_regress_residual(y: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    """逐日截面回归 y ~ 1 + x 取残差（用于市值/行业中性化）。"""
    xm = x.mean(axis=1)
    ym = y.mean(axis=1)
    cov = ((x.sub(xm, axis=0)) * (y.sub(ym, axis=0))).mean(axis=1)
    var = (x.sub(xm, axis=0) ** 2).mean(axis=1).replace(0, np.nan)
    beta = cov / var
    alpha = ym - beta * xm
    return y.sub(alpha, axis=0).sub(x.mul(beta, axis=0))


def _safe_div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a / b.replace(0, np.nan)


def _log(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(x.where(x > 0))


def _sign(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x)


def _abs(x: pd.DataFrame) -> pd.DataFrame:
    return x.abs()


def _where_pos(x: pd.DataFrame) -> pd.DataFrame:
    return x.where(x > 0)


def _neg(x: pd.DataFrame) -> pd.DataFrame:
    return -x


UNARY_OPS: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    'cs_rank': _cs_rank,
    'cs_zscore': _cs_zscore,
    'cs_demean': _cs_demean,
    'log': _log,
    'sign': _sign,
    'abs': _abs,
    'where_pos': _where_pos,
    'neg': _neg,
    'inverse': _inverse,
    'cs_scale': _cs_scale,
}

WINDOW_OPS: Dict[str, Callable[..., pd.DataFrame]] = {
    'ts_shift': _ts_shift,
    'ts_delta': _ts_delta,
    'ts_pct_change': _ts_pct_change,
    'ts_rank': _ts_rank,
    'ts_slope': _ts_slope,
    'decay_linear': _decay_linear,
    'ts_zscore': _ts_zscore,
    'ts_argmax': lambda x, n: _ts_extreme_pos(x, n, 'max'),
    'ts_argmin': lambda x, n: _ts_extreme_pos(x, n, 'min'),
}

BINARY_OPS: Dict[str, Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]] = {
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'div': _safe_div,
    'cs_regress_residual': cs_regress_residual,
    'where': _where,
    'trade_when': _trade_when,
}

# 带 by 参数的分组算子：输入 1 个表达式 + 1 个或多个分组/风格字段
GROUP_OPS: Dict[str, str] = {
    'cs_group_demean': 'one',
    'cs_group_rank': 'one',
    'cs_regress_multi': 'multi',
}

BINARY_WINDOW_OPS: Dict[str, Callable[..., pd.DataFrame]] = {
    'ts_corr': _ts_corr,
    'ts_cov': _ts_cov,
}

# 算子分类：用于强制"算子交互"设计规则（详见 prompts/02_researcher.md）
OP_CATEGORY: Dict[str, str] = {
    **{op: 'arithmetic' for op in ('add', 'sub', 'mul', 'div', 'neg', 'abs',
                                   'log', 'sign', 'where_pos', 'inverse')},
    **{op: 'cross_section' for op in ('cs_rank', 'cs_zscore', 'cs_demean',
                                      'cs_regress_residual', 'cs_scale',
                                      'cs_winsorize')},
    **{op: 'ts_stat' for op in ('ts_rolling', 'ts_quantile')},
    **{op: 'conditional' for op in ('where', 'trade_when')},
    **{op: 'cross_section' for op in GROUP_OPS},
    **{op: 'ts_dynamic' for op in ('ts_shift', 'ts_delta', 'ts_pct_change',
                                   'ts_rank', 'ts_slope', 'decay_linear',
                                   'ts_zscore', 'ts_argmax', 'ts_argmin',
                                   'ts_corr', 'ts_cov')},
}

# 交叉/相对信息算子：至少出现一个，避免"纯绝对量"的因子
RELATIVE_OPS = ('cs_rank', 'cs_zscore', 'cs_scale', 'ts_corr', 'ts_cov')

# 各算子的最小窗口：窗口太小统计量没有意义（与实现里的下界保持一致）
MIN_OP_WINDOW: Dict[str, int] = {
    'ts_rolling': 2,
    'ts_rank': 5,
    'ts_slope': 5,
    'ts_zscore': 5,
    'ts_quantile': 5,
    'ts_argmax': 5,
    'ts_argmin': 5,
    'ts_corr': 10,
    'ts_cov': 10,
}

OP_REGISTRY: Tuple[str, ...] = (
    tuple(UNARY_OPS) + tuple(WINDOW_OPS) + tuple(BINARY_OPS)
    + tuple(BINARY_WINDOW_OPS) + tuple(GROUP_OPS) + ('ts_rolling',)
)

_OPS = {
    **UNARY_OPS,
    'cs_group_demean': _cs_group_demean,
    'cs_group_rank': _cs_group_rank,
    'cs_regress_multi': None,   # 需要 panels，单独分派
    **WINDOW_OPS,
    **BINARY_OPS,
    **BINARY_WINDOW_OPS,
}

# 各算子对输入的最少历史长度（不含子表达式的 warmup）
_MIN_WINDOW = {
    'ts_shift': lambda a: int(a['n']) + 1,
    'ts_delta': lambda a: int(a['n']) + 1,
    'ts_pct_change': lambda a: int(a['n']) + 1,
    'ts_rolling': lambda a: int(a['n']),
    'ts_rank': lambda a: int(a['n']),
    'ts_corr': lambda a: int(a['n']),
    'ts_cov': lambda a: int(a['n']),
    'ts_slope': lambda a: int(a['n']),
    'decay_linear': lambda a: int(a['n']),
    'ts_zscore': lambda a: int(a['n']),
    'ts_quantile': lambda a: int(a['n']),
    'ts_argmax': lambda a: int(a['n']),
    'ts_argmin': lambda a: int(a['n']),
}


# ─────────────────────────── 校验 / 编译 ───────────────────────────

def validate(spec: Dict[str, Any], path: str = 'spec') -> List[str]:
    """递归校验表达式；返回错误列表（空表示合法）。"""
    errors: List[str] = []
    if not isinstance(spec, dict):
        return [f'{path}: 表达式必须是 dict']

    if 'field' in spec:
        field = spec['field']
        if field not in PANEL_FIELDS:
            errors.append(f'{path}: 未知字段 {field}')
        if set(spec) - {'field'}:
            errors.append(f'{path}: 字段节点不能带其他键 {sorted(set(spec) - {"field"})}')
        return errors

    op = spec.get('op')
    if op not in OP_REGISTRY:
        return [f'{path}: 不允许的算子 {op}']

    inputs = spec.get('inputs')
    if not isinstance(inputs, list):
        return [f'{path}: 缺少 inputs 列表']

    arity = 2 if (op in BINARY_OPS or op in BINARY_WINDOW_OPS) else 1
    if len(inputs) != arity:
        errors.append(f'{path}: {op} 需要 {arity} 个输入，收到 {len(inputs)}')
    if op == 'ts_quantile':
        try:
            q = float(spec.get('q'))
            if not 0.0 < q < 1.0:
                errors.append(f'{path}: ts_quantile 的 q 必须在 (0,1) 内')
        except (TypeError, ValueError):
            errors.append(f'{path}: ts_quantile 需要数值参数 q')
    if op in GROUP_OPS:
        raw = spec.get('by')
        refs = raw if isinstance(raw, list) else [raw]
        if GROUP_OPS[op] == 'multi' and not isinstance(raw, list):
            errors.append(f'{path}: {op} 的 by 必须是字段列表')
        elif GROUP_OPS[op] == 'one' and isinstance(raw, list):
            errors.append(f'{path}: {op} 的 by 必须是单个字段名')
        for ref in refs:
            if ref not in GROUP_FIELDS and ref not in PANEL_FIELDS:
                errors.append(f'{path}: 未知分组/风格字段 {ref!r}')
    if op == 'cs_winsorize':
        try:
            q = float(spec.get('q', 0.01))
            if not 0.0 <= q < 0.5:
                errors.append(f'{path}: cs_winsorize 的 q 必须在 [0,0.5) 内')
        except (TypeError, ValueError):
            errors.append(f'{path}: cs_winsorize 需要数值参数 q')

    if op in _MIN_WINDOW:
        try:
            _check_window(spec.get('n'), lo=MIN_OP_WINDOW.get(op, 1))
        except ValueError as exc:
            errors.append(f'{path}: {exc}')
    if op == 'ts_rolling' and spec.get('agg') not in (
            'mean', 'std', 'sum', 'max', 'min', 'median', 'skew', 'kurt'):
        errors.append(f'{path}: ts_rolling 的 agg 非法 {spec.get("agg")!r}')

    for i, sub in enumerate(inputs):
        errors.extend(validate(sub, f'{path}.inputs[{i}]'))
    return errors


def warmup(spec: Dict[str, Any]) -> int:
    """估算表达式需要的历史长度（交易日）。"""
    if 'field' in spec:
        return 1
    base = max([warmup(s) for s in spec.get('inputs', [])], default=1)
    fn = _MIN_WINDOW.get(spec.get('op'))
    return base if fn is None else base + fn(spec) - 1


def complexity(spec: Dict[str, Any]) -> int:
    if 'field' in spec:
        return 1
    return 1 + sum(complexity(s) for s in spec.get('inputs', []))


def max_depth(spec: Dict[str, Any], depth: int = 1) -> int:
    if 'field' in spec:
        return depth
    return max([max_depth(s, depth + 1) for s in spec.get('inputs', [])] or [depth])


def compile_factor(spec: Dict[str, Any], panels: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """编译表达式为 (date × symbol) 面板。"""
    if 'field' in spec:
        field = spec['field']
        if field not in panels:
            raise KeyError(f'面板缺少字段 {field}')
        return panels[field].astype(float)

    op = spec['op']
    inputs = [compile_factor(s, panels) for s in spec.get('inputs', [])]
    if op == 'ts_rolling':
        return _ts_rolling(inputs[0], spec['n'], spec['agg'])
    if op == 'ts_quantile':
        return _ts_quantile(inputs[0], spec['n'], spec['q'])
    if op == 'cs_winsorize':
        return _cs_winsorize(inputs[0], spec.get('q', 0.01))
    if op in GROUP_OPS:
        if GROUP_OPS[op] == 'multi':
            return _cs_regress_multi(inputs[0], panels, list(spec['by']))
        return _OPS[op](inputs[0], panels[spec['by']])
    if op in WINDOW_OPS:
        return _OPS[op](inputs[0], spec['n'])
    if op in BINARY_WINDOW_OPS:
        return _OPS[op](inputs[0], inputs[1], spec['n'])
    if op in BINARY_OPS:
        return _OPS[op](inputs[0], inputs[1])
    return _OPS[op](inputs[0])


# ─────────────────────────── 规范化 / 去重 ───────────────────────────

def _args(spec: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in spec.items() if k not in ('op', 'inputs')}


def canonical(spec: Dict[str, Any]) -> str:
    """可读的规范形式，例如 ``ts_pct_change(close, n=5)``。"""
    if 'field' in spec:
        return spec['field']
    inner = ', '.join(canonical(s) for s in spec.get('inputs', []))
    args = _args(spec)
    extra = ''.join(f', {k}={args[k]}' for k in sorted(args))
    return f'{spec["op"]}({inner}{extra})'


def structure_key(spec: Dict[str, Any]) -> str:
    """结构指纹：忽略窗口数值与聚合方式，用于家族去重。"""
    if 'field' in spec:
        return f'field:{spec["field"]}'
    inner = ','.join(structure_key(s) for s in spec.get('inputs', []))
    return f'{spec["op"]}({inner})'


def exact_key(spec: Dict[str, Any]) -> str:
    """精确指纹：包含窗口数值。"""
    payload = json.dumps(spec, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def family_of(spec: Dict[str, Any]) -> str:
    return hashlib.sha256(structure_key(spec).encode()).hexdigest()[:12]


def to_ast(spec: Dict[str, Any]) -> Dict[str, Any]:
    """深拷贝的规范化 AST（去掉未知键），写入台账前使用。"""
    if 'field' in spec:
        return {'field': spec['field']}
    out: Dict[str, Any] = {'op': spec['op'],
                           'inputs': [to_ast(s) for s in spec.get('inputs', [])]}
    out.update(_args(spec))
    return out


def fields_of(spec: Dict[str, Any]) -> List[str]:
    if 'field' in spec:
        return [spec['field']]
    out: List[str] = []
    if spec.get('op') in GROUP_OPS:
        raw = spec.get('by')
        out.extend(raw if isinstance(raw, list) else [raw])
    for sub in spec.get('inputs', []):
        out.extend(fields_of(sub))
    return sorted(set(out))


def check_time_safety(spec: Dict[str, Any], signal_time: str = 'close') -> List[str]:
    """检查字段可用时点是否早于信号时点。

    面板里的 ``pit_fundamental`` 字段要求上游已按公告日对齐；
    若调用方无法保证，应在数据契约里显式降级证据等级。
    """
    errors: List[str] = []
    for field in fields_of(spec):
        available = FIELD_AVAILABILITY[field]
        if available == 'pit_fundamental' and signal_time == 'close':
            errors.append(
                f'字段 {field} 是基本面字段，需确认已按公告日对齐（available_time <= T 收盘）'
            )
    return errors


@dataclass(frozen=True)
class CompiledFactor:
    name: str
    values: pd.DataFrame
    warmup: int
    complexity: int
    fields: Tuple[str, ...]

    def coverage(self, mask: pd.DataFrame) -> float:
        m = self.values.where(mask)
        denom = mask.sum(axis=1).replace(0, np.nan)
        return float((m.notna().sum(axis=1) / denom).mean())


# ─────────────────────────── 设计规则（因子丰富度） ───────────────────────────

@dataclass(frozen=True)
class DesignReport:
    categories: Tuple[str, ...]
    has_relative: bool
    is_linear_blend: bool
    nodes: int
    depth: int
    failures: Tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_dict(self) -> Dict[str, Any]:
        return {
            'categories': list(self.categories),
            'has_relative': self.has_relative,
            'is_linear_blend': self.is_linear_blend,
            'nodes': self.nodes,
            'depth': self.depth,
            'ok': self.ok,
            'failures': list(self.failures),
        }


def operator_categories(spec: Dict[str, Any]) -> List[str]:
    if 'field' in spec:
        return []
    cats = {OP_CATEGORY[spec['op']]}
    for sub in spec.get('inputs', []):
        cats.update(operator_categories(sub))
    return sorted(cats)


def _ops_of(spec: Dict[str, Any]) -> List[str]:
    if 'field' in spec:
        return []
    out = [spec['op']]
    for sub in spec.get('inputs', []):
        out.extend(_ops_of(sub))
    return out


def _is_linear_blend(spec: Dict[str, Any]) -> bool:
    """根部就是两个"有内容的子树"相加减 → 线性叠加，视为共线性风险。

    允许 ``div``/``mul``/``ts_corr`` 这类交互，因为它们是乘积或比值关系；
    只挡"把两个因子相加"。
    """
    if spec.get('op') not in ('add', 'sub'):
        return False
    ins = spec.get('inputs', [])
    return len(ins) == 2 and all(complexity(i) >= 3 for i in ins)


def design_report(spec: Dict[str, Any], min_categories: int = 3) -> DesignReport:
    """评估表达式是否满足"算子交互 + 相对信息 + 非线性"的设计要求。"""
    cats = tuple(operator_categories(spec))
    ops = _ops_of(spec)
    failures: List[str] = []
    if len(cats) < min_categories:
        failures.append(f'算子类别 {len(cats)} < {min_categories}：{list(cats)}')
    has_relative = any(op in RELATIVE_OPS for op in ops)
    if not has_relative:
        failures.append('缺少交叉/相对算子（cs_rank / cs_zscore / ts_corr / ts_cov）')
    linear = _is_linear_blend(spec)
    if linear:
        failures.append('根节点是两个子表达式相加减，属于线性叠加，用 div/mul/ts_corr 表达交互')
    return DesignReport(categories=cats, has_relative=has_relative,
                        is_linear_blend=linear, nodes=complexity(spec),
                        depth=max_depth(spec), failures=tuple(failures))

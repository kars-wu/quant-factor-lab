"""
多因子合成器。

━━━ 原理 ━━━

单因子 IC≈0.03 看起来弱，但 N 个独立因子组合后 IC 可达 0.03×√N。
ML 不是预测涨跌，而是"从噪音中提取微弱但稳健的信号"。

━━━ 流程 ━━━

  FactorAnalyzer 的因子数据
      ↓
  构建特征矩阵 X(日期×股票 → 因子值)
  构建目标变量 y(前瞻N日收益率)
      ↓
  因子去重(相关性 > 0.8 的只保留 IC 最高的)
      ↓
  时间序列分割(前60%训练，后40%测试)
      ↓
  训练模型(线性回归 Ridge / LightGBM)
      ↓
  输出 composite alpha(每日每只股票预测得分)
      ↓
  验证:训练IC vs 测试IC + 分层回测

━━━ 防过拟合设计 ━━━

  1. 严格时间序列分割(不随机划分)
  2. 特征标准化(训练集 fit，测试集 transform)
  3. 正则化(Ridge L2 / LightGBM 限树深+学习率)
  4. IC 衰减率 > 50% = 过拟合警告
  5. 因子去重(高相关因子只保留IC最高的)

用法:
    from strategies.factor_synthesis import MultiFactorSynthesizer

    synth = MultiFactorSynthesizer(fa)
    result = synth.synthesize(model='linear', forward_days=5)
    print(result.summary())

    # 合成因子可作为 PortfolioManager 的打分函数
    # alpha 高的优先买入
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


@dataclass
class SynthesisResult:
    """多因子合成结果。"""
    model_name: str                    # 'linear' | 'lightgbm'
    forward_days: int
    train_ic: float = 0.0             # 训练集 IC
    test_ic: float = 0.0              # 测试集 IC
    decay_ratio: float = 0.0          # 衰减率 = test_ic / train_ic
    is_overfit: bool = False
    alpha: pd.Series = field(default_factory=pd.Series)  # 合成因子
    factor_weights: dict = field(default_factory=dict)    # 因子权重/重要度
    layer_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_factors: int = 0
    n_train: int = 0
    n_test: int = 0
    removed_factors: list = field(default_factory=list)
    rolling_ic: pd.Series = field(default_factory=pd.Series)  # 滚动IC序列
    rolling_ic_consecutive_neg: int = 0    # 连续IC<0天数(失效预警)
    ic_stability: float = 0.0             # IC稳定性 = IC均值/IC标准差

    @property
    def is_effective(self) -> bool:
        return abs(self.test_ic) > 0.03 and not self.is_overfit

    def summary(self) -> str:
        flag = "✓ 有效" if self.is_effective else "✗ 无效"
        overfit = " ⚠ 过拟合" if self.is_overfit else ""
        return (f"{self.model_name} ({self.n_factors}因子, 前瞻{self.forward_days}天) | "
                f"训练IC={self.train_ic:.4f} 测试IC={self.test_ic:.4f} "
                f"衰减={self.decay_ratio:.0%} | "
                f"训练{self.n_train}样本 测试{self.n_test}样本 "
                f"{flag}{overfit}")


class MultiFactorSynthesizer:
    """多因子合成器:弱因子 → composite alpha。"""

    def __init__(self, factor_analyzer):
        """
        参数
        ----
        factor_analyzer : FactorAnalyzer 实例(已加载因子)
        """
        self._fa = factor_analyzer

    def synthesize(
        self,
        model: str = 'linear',
        forward_days: int = 5,
        train_ratio: float = 0.6,
        corr_threshold: float = 0.7,
        ridge_alpha: float = 1.0,
        lgb_params: dict = None,
        ic_filter: float = 0.0,
        importance_prune: float = 0.05,
        rolling_ic_window: int = 60,
        rank_transform: bool = True,
        winsorize: bool = True,
        winsorize_pct: float = 0.01,
        factor_names: list = None,
        universe_mask: Optional[pd.Series] = None,
    ) -> SynthesisResult:
        """
        多因子合成。

        参数
        ----
        model              : 'linear' | 'lightgbm' | 'ic_weighted'
        forward_days       : 前瞻天数
        train_ratio        : 训练集比例
        corr_threshold     : 因子去重阈值(默认0.7)
        ic_filter          : IC预筛选阈值；<=0 表示关闭预筛选
        importance_prune   : 重要性剪枝阈值
        rolling_ic_window  : 滚动IC窗口天数
        rank_transform     : 截面排名变换(每日把因子值转为排名，降低极值影响)
        winsorize          : 缩尾处理(clip极值到1/99百分位)
        winsorize_pct      : 缩尾百分位
        factor_names       : 指定参与合成的因子列表；None=全部已注册因子
        universe_mask      : 动态股票池过滤，MultiIndex(date, symbol) -> bool
        """
        # 1. 获取因子矩阵和目标变量
        if factor_names is None:
            factor_names = [n for n in self._fa._factors if not n.endswith('_neutral')]
        X = self._fa.get_factor_matrix(factor_names)
        y = self._fa.get_forward_returns(forward_days)

        # 对齐回测收益口径：train_ic/test_ic 用开盘买入(close[d+1+days]/open[d+1]-1)
        # 训练目标 y 仍用 close-to-close，仅 IC 评价用对齐口径
        y_open = None
        _get_open = getattr(self._fa, 'get_forward_returns_open', None)
        if _get_open is not None:
            y_open = _get_open(forward_days)

        if X.empty or y.empty:
            logger.warning("因子矩阵或目标变量为空")
            return SynthesisResult(model_name=model, forward_days=forward_days)

        # 对齐
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        # 动态股票池过滤（防未来函数：只保留当日允许交易的标的）
        if universe_mask is not None and not universe_mask.empty:
            mask = universe_mask.reindex(common_idx).fillna(False).astype(bool)
            X = X[mask.values]
            y = y[mask.values]

        # 去掉有 NaN 的行
        valid = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid]
        y = y[valid]

        if len(X) < 100:
            logger.warning("有效样本太少: {}", len(X))
            return SynthesisResult(model_name=model, forward_days=forward_days,
                                   n_factors=len(X.columns))

        logger.info("合成数据 | {} 因子 | {} 样本 | 前瞻 {} 天",
                    len(X.columns), len(X), forward_days)

        # 1b. 截面排名变换 + 缩尾(降低极值对IC的干扰)
        if rank_transform:
            X = self._rank_transform(X)
        if winsorize:
            X = self._winsorize(X, winsorize_pct)

        # 2. 因子去重（相关性 > 0.7 视为重复）
        kept_factors, removed = self._deduplicate_factors(
            X, y, list(X.columns), corr_threshold,
        )
        X = X[kept_factors]

        # 3. IC预筛选(可选；默认关闭，避免过早丢弃弱但可组合的因子)
        if ic_filter > 0:
            kept_by_ic, ic_filtered = self._ic_prefilter(X, y, ic_filter)
            if ic_filtered:
                logger.info("IC预筛选移除: {}", ic_filtered)
                removed.extend(ic_filtered)
            X = X[kept_by_ic]
            kept_factors = list(X.columns)  # 更新 kept_factors
        else:
            logger.info("IC预筛选已关闭")

        if len(X.columns) == 0:
            logger.warning("所有因子被IC预筛选移除")
            return SynthesisResult(model_name=model, forward_days=forward_days,
                                   removed_factors=removed)

        # 3. 时间序列分割
        dates = X.index.get_level_values('date').unique().sort_values()
        split_point = int(len(dates) * train_ratio)
        train_dates = dates[:split_point]
        test_dates = dates[split_point:]

        train_mask = X.index.get_level_values('date').isin(train_dates)
        test_mask = X.index.get_level_values('date').isin(test_dates)

        X_train = X[train_mask].values
        y_train = y[train_mask].values
        X_test = X[test_mask].values
        y_test = y[test_mask].values

        # 4. 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 5. 训练模型
        if model == 'lightgbm':
            try:
                import lightgbm as lgb
            except ImportError:
                logger.warning("lightgbm 未安装，回退到 linear")
                model = 'linear'

        if model == 'lightgbm':
            params = lgb_params or {
                'objective': 'regression',
                'num_leaves': 15,
                'learning_rate': 0.05,
                'n_estimators': 100,
                'max_depth': 3,
                'min_child_samples': 50,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'verbose': -1,
            }
            reg = lgb.LGBMRegressor(**params)
            reg.fit(X_train_scaled, y_train)
            factor_weights = dict(zip(kept_factors, reg.feature_importances_.tolist()))
            predictions_train = reg.predict(X_train_scaled)
            predictions_test = reg.predict(X_test_scaled)

        elif model == 'ic_weighted':
            # IC加权: 用训练集每个因子的IC作为权重, 直接加权组合
            X_train_df = X[train_mask]
            train_ics = []
            for i, fname in enumerate(kept_factors):
                ic_vals = []
                for date in train_dates:
                    mask = X_train_df.index.get_level_values('date') == date
                    f_vals = X_train_df.loc[mask, fname].values
                    r_vals = y_train[mask]
                    valid = ~(np.isnan(f_vals) | np.isnan(r_vals))
                    if valid.sum() < 5:
                        continue
                    ic, _ = stats.spearmanr(f_vals[valid], r_vals[valid])
                    if not np.isnan(ic):
                        ic_vals.append(ic)
                mean_ic = np.mean(ic_vals) if ic_vals else 0
                train_ics.append(mean_ic)

            weights = np.array(train_ics)
            if np.abs(weights).sum() > 0:
                weights = weights / np.abs(weights).sum()
            else:
                weights = np.ones(len(kept_factors)) / len(kept_factors)

            factor_weights = dict(zip(kept_factors, weights.tolist()))
            predictions_train = X_train_scaled @ weights
            predictions_test = X_test_scaled @ weights

        else:
            # Ridge 回归
            reg = Ridge(alpha=ridge_alpha)
            reg.fit(X_train_scaled, y_train)
            factor_weights = dict(zip(kept_factors, reg.coef_.tolist()))
            predictions_train = reg.predict(X_train_scaled)
            predictions_test = reg.predict(X_test_scaled)

        # 7. IC 计算
        if y_open is not None:
            # 对齐回测口径：用开盘买入收益作为 actual；缺失值由 _calc_daily_ic 过滤
            actual_train = y_open.reindex(X[train_mask].index).values
            actual_test = y_open.reindex(X[test_mask].index).values
        else:
            actual_train, actual_test = y_train, y_test
        train_ic = self._calc_daily_ic(
            predictions_train,
            actual_train,
            X[train_mask].index,
        )
        test_ic = self._calc_daily_ic(
            predictions_test,
            actual_test,
            X[test_mask].index,
        )

        # 7b. 重要性剪枝：剔除重要性 < 5% 的因子后重训
        pruned = []
        # ic_weighted 模型跳过剪枝（已有IC权重过滤，剪枝后维度不匹配）
        if importance_prune > 0 and len(kept_factors) > 2 and model != 'ic_weighted':
            weights_arr = np.abs(list(factor_weights.values()))
            total_weight = weights_arr.sum()
            if total_weight > 0:
                low_importance = [
                    kept_factors[i] for i, w in enumerate(weights_arr)
                    if w / total_weight < importance_prune
                ]
                if low_importance:
                    pruned = low_importance
                    kept_factors = [f for f in kept_factors if f not in low_importance]
                    X = X[kept_factors]
                    logger.info("重要性剪枝移除: {}", pruned)
                    # 重新分割+训练
                    train_mask = X.index.get_level_values('date').isin(train_dates)
                    test_mask = X.index.get_level_values('date').isin(test_dates)
                    X_train = X[train_mask].values
                    X_test = X[test_mask].values
                    y_train = y[train_mask].values
                    y_test = y[test_mask].values
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    if model == 'lightgbm':
                        reg = lgb.LGBMRegressor(**params)
                    else:
                        reg = Ridge(alpha=ridge_alpha)
                    reg.fit(X_train_scaled, y_train)
                    predictions_train = reg.predict(X_train_scaled)
                    predictions_test = reg.predict(X_test_scaled)
                    train_ic = self._calc_daily_ic(predictions_train, y_train, X[train_mask].index)
                    test_ic = self._calc_daily_ic(predictions_test, y_test, X[test_mask].index)
                    if model == 'lightgbm':
                        factor_weights = dict(zip(kept_factors, reg.feature_importances_.tolist()))
                    else:
                        factor_weights = dict(zip(kept_factors, reg.coef_.tolist()))

        removed.extend([f"{f}(重要性低)" for f in pruned])

        decay_ratio = test_ic / train_ic if train_ic > 0 else 0
        is_overfit = decay_ratio < 0.5 and train_ic > 0.02

        # 8. 合成因子(全量预测)
        X_all_scaled = scaler.transform(X.values)
        if model == 'ic_weighted':
            predictions_all = X_all_scaled @ weights
        elif model == 'lightgbm':
            predictions_all = reg.predict(X_all_scaled)
        else:
            predictions_all = reg.predict(X_all_scaled)
        alpha = pd.Series(
            predictions_all,
            index=X.index,
            name='composite_alpha',
        )

        # 9. 分层回测(测试集)
        test_pred = pd.Series(predictions_test, index=X[test_mask].index)
        test_ret = pd.Series(y_test, index=X[test_mask].index)
        layer_returns = self._layer_backtest(test_pred, test_ret, n_layers=5)

        result = SynthesisResult(
            model_name=model,
            forward_days=forward_days,
            train_ic=train_ic,
            test_ic=test_ic,
            decay_ratio=decay_ratio,
            is_overfit=is_overfit,
            alpha=alpha,
            factor_weights=factor_weights,
            layer_returns=layer_returns,
            n_factors=len(kept_factors),
            n_train=len(X_train),
            n_test=len(X_test),
            removed_factors=removed,
        )

        logger.info("合成完成 | {}", result.summary())
        if removed:
            logger.info("因子移除: {}", removed)

        # 10. 滚动 IC 监控：连续 IC < 0 视为失效预警
        rolling_ic = self._calc_rolling_ic(
            predictions_test, y_test, X[test_mask].index, window=rolling_ic_window,
        )
        result.rolling_ic = rolling_ic
        # 连续IC<0的最大天数
        if not rolling_ic.empty:
            consecutive = 0
            max_consecutive = 0
            for ic_val in rolling_ic:
                if ic_val < 0:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 0
            result.rolling_ic_consecutive_neg = max_consecutive
            # IC稳定性
            if rolling_ic.std() > 0:
                result.ic_stability = float(rolling_ic.mean() / rolling_ic.std())

            if max_consecutive >= 20:
                logger.warning("⚠ 因子失效预警: 连续{}天IC<0(阈值20天)", max_consecutive)

        return result

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _rank_transform(X: pd.DataFrame) -> pd.DataFrame:
        """
        截面排名变换:每日把因子值转为截面排名百分比(0~1)。

        用 Spearman 相关而非 Pearson 的原因：排名对原始值的单调变换不敏感，
        对异常值不敏感。这里对因子也做排名变换，让IC更稳定。
        """
        result = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
        dates = X.index.get_level_values('date').unique()
        for date in dates:
            mask = X.index.get_level_values('date') == date
            day_data = X[mask]
            if len(day_data) < 2:
                continue
            result.loc[mask] = day_data.rank(pct=True).values
        return result.fillna(0.5)

    @staticmethod
    def _winsorize(X: pd.DataFrame, pct: float = 0.01) -> pd.DataFrame:
        """
        缩尾处理:clip到1/99百分位，消除极端值对回归的干扰。
        """
        result = X.copy()
        for col in result.columns:
            lo = result[col].quantile(pct)
            hi = result[col].quantile(1 - pct)
            result[col] = result[col].clip(lo, hi)
        return result

    def _ic_prefilter(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ic_threshold: float,
    ) -> tuple[list[str], list[str]]:
        """
        单变量 IC 筛选。
        训练集上计算每个因子的IC，绝对值<阈值的不入选。
        """
        # 只在训练集上算IC(避免泄漏)
        dates = X.index.get_level_values('date').unique()
        split = int(len(dates) * 0.6)
        train_dates = dates[:split]

        kept = []
        removed = []
        for col in X.columns:
            f_vals = X[col]
            # 只用训练集数据
            train_mask = X.index.get_level_values('date').isin(train_dates)
            f_train = f_vals[train_mask]
            y_train = y[train_mask]

            valid = ~(f_train.isna() | y_train.isna())
            if valid.sum() < 30:
                removed.append(f"{col}(样本不足)")
                continue

            ic, _ = stats.spearmanr(f_train[valid].values, y_train[valid].values)
            if np.isnan(ic):
                ic = 0

            if abs(ic) >= ic_threshold:
                kept.append(col)
            else:
                removed.append(f"{col}(IC={ic:.4f}<{ic_threshold})")

        return kept, removed

    @staticmethod
    def _calc_rolling_ic(
        predictions: np.ndarray,
        actual: np.ndarray,
        index: pd.MultiIndex,
        window: int = 60,
    ) -> pd.Series:
        """
        滚动 IC 监控。
        连续20天IC<0 → 因子失效预警。
        """
        pred_series = pd.Series(predictions, index=index, name='pred')
        actual_series = pd.Series(actual, index=index, name='actual')

        dates = index.get_level_values('date').unique().sort_values()
        if len(dates) < window:
            return pd.Series(dtype=float)

        rolling_ic = []
        for i in range(window, len(dates)):
            window_dates = dates[i - window:i]
            mask = pred_series.index.get_level_values('date').isin(window_dates)
            p = pred_series[mask].values
            a = actual_series[mask].values

            valid = ~(np.isnan(p) | np.isnan(a))
            if valid.sum() < 10:
                rolling_ic.append((dates[i], 0))
                continue

            ic, _ = stats.spearmanr(p[valid], a[valid])
            rolling_ic.append((dates[i], ic if not np.isnan(ic) else 0))

        return pd.Series(
            [v for _, v in rolling_ic],
            index=pd.DatetimeIndex([d for d, _ in rolling_ic]),
            name='rolling_ic',
        )

    def _deduplicate_factors(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        factor_names: list[str],
        corr_threshold: float,
    ) -> tuple[list[str], list[str]]:
        """
        高相关因子去重:相关 > threshold 的只保留 IC 最高的。
        """
        corr_matrix = X.corr().abs()

        # 计算每个因子的 IC(用于去重时选保留哪个)
        factor_ic = {}
        for name in factor_names:
            if name in X.columns:
                f_vals = X[name]
                valid = ~(f_vals.isna() | y.isna())
                if valid.sum() > 10:
                    ic, _ = stats.spearmanr(f_vals[valid], y[valid])
                    factor_ic[name] = abs(ic) if not np.isnan(ic) else 0
                else:
                    factor_ic[name] = 0

        kept = list(X.columns)
        removed = []

        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                f1, f2 = corr_matrix.columns[i], corr_matrix.columns[j]
                if f1 not in kept or f2 not in kept:
                    continue
                corr = corr_matrix.loc[f1, f2]
                if corr > corr_threshold:
                    # 移除 IC 较低的
                    ic1 = factor_ic.get(f1, 0)
                    ic2 = factor_ic.get(f2, 0)
                    if ic1 >= ic2:
                        if f2 in kept:
                            kept.remove(f2)
                            removed.append(f"{f2}(corr={corr:.2f},ic={ic2:.4f})")
                    else:
                        if f1 in kept:
                            kept.remove(f1)
                            removed.append(f"{f1}(corr={corr:.2f},ic={ic1:.4f})")

        return kept, removed

    @staticmethod
    def _calc_daily_ic(
        predictions: np.ndarray,
        actual: np.ndarray,
        index: pd.MultiIndex,
    ) -> float:
        """计算日 IC 的均值(Spearman 秩相关)。"""
        pred_series = pd.Series(predictions, index=index, name='pred')
        actual_series = pd.Series(actual, index=index, name='actual')

        dates = index.get_level_values('date').unique()
        ic_values = []

        for date in dates:
            mask = pred_series.index.get_level_values('date') == date
            p = pred_series[mask].values
            a = actual_series[mask].values

            valid = ~(np.isnan(p) | np.isnan(a))
            if valid.sum() < 5:
                continue

            p_valid = p[valid]
            a_valid = a[valid]
            if p_valid.std() == 0 or a_valid.std() == 0:
                continue

            ic, _ = stats.spearmanr(p_valid, a_valid)
            if not np.isnan(ic):
                ic_values.append(ic)

        return float(np.mean(ic_values)) if ic_values else 0.0

    @staticmethod
    def _layer_backtest(
        predictions: pd.Series,
        returns: pd.Series,
        n_layers: int = 5,
    ) -> pd.DataFrame:
        """分层回测。"""
        from collections import defaultdict
        layer_returns = defaultdict(list)

        dates = predictions.index.get_level_values('date').unique()
        for date in dates:
            mask = predictions.index.get_level_values('date') == date
            p = predictions[mask].sort_values()
            r = returns[mask]

            n = len(p)
            if n < n_layers:
                continue

            layer_size = n // n_layers
            for layer in range(n_layers):
                start = layer * layer_size
                end = (layer + 1) * layer_size if layer < n_layers - 1 else n
                layer_syms = p.index[start:end]
                layer_ret = r.loc[layer_syms].mean()
                layer_returns[f"L{layer+1}"].append(layer_ret)

        result = {}
        for layer, rets in layer_returns.items():
            result[layer] = {
                'mean_return': float(np.mean(rets)),
                'std': float(np.std(rets)),
                'n_days': len(rets),
            }

        return pd.DataFrame(result).T

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------

    @staticmethod
    def print_result(result: SynthesisResult):
        """打印合成结果。"""
        from rich.console import Console

        console = Console()

        # 概要
        console.print("\n[bold cyan]多因子合成结果[/bold cyan]")
        console.print(f"  模型: {result.model_name}")
        console.print(f"  因子数: {result.n_factors}")
        console.print(f"  训练IC: {result.train_ic:.4f}")
        console.print(f"  测试IC: {result.test_ic:.4f}")
        console.print(f"  衰减率: {result.decay_ratio:.0%}")
        console.print(f"  过拟合: {'⚠ 是' if result.is_overfit else '否'}")
        console.print(f"  判定: {'✓ 有效' if result.is_effective else '✗ 无效'}")

        # 滚动IC监控
        if not result.rolling_ic.empty:
            console.print(f"  IC稳定性: {result.ic_stability:.2f}")
            console.print(f"  连续IC<0天数: {result.rolling_ic_consecutive_neg} (阈值20天)")
            if result.rolling_ic_consecutive_neg >= 20:
                console.print(f"  [red]⚠ 因子失效预警！连续{result.rolling_ic_consecutive_neg}天IC<0[/red]")

        if result.removed_factors:
            console.print(f"  因子移除: {', '.join(result.removed_factors)}")

        # 因子权重
        if result.factor_weights:
            console.print("\n[bold]因子权重/重要度:[/bold]")
            sorted_weights = sorted(
                result.factor_weights.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            for name, weight in sorted_weights:
                console.print(f"  {name:20s} {weight:+.4f}")

        # 分层回测
        if not result.layer_returns.empty:
            console.print("\n[bold]分层回测(测试集):[/bold]")
            console.print(result.layer_returns.round(4))

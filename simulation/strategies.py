"""
通用策略注册框架。

一个「日频模拟盘策略」= 一个纯函数/类，输入最新面板，输出当日选股信号。
引擎负责 T+1 开盘成交、持仓管理、净值与交易记录。

约定（与回测口径一致）：
  - 信号在 T 日收盘生成，T+1 开盘价成交
  - 策略决定「目标持仓集合」或「开/平仓订单」，引擎执行
  - 所有收益类输入由库内因子显式 shift(1)，本框架不做二次处理
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class StrategySpec:
    """策略定义：一个策略 = 一个信号函数 + 交易规则。"""
    strategy_id: str
    name: str
    description: str
    # signal_fn(panel_df, date) -> dict[str, float] 或 pd.Series
    #   输入：最新全量面板 (date, ticker) 排序；输出 {ticker: 信号分数}
    #   或返回 pd.Series(index=(date,ticker)) 由框架取最近一日
    signal_fn: Callable
    # 交易规则
    top_k: int = 10              # 买入前 K 名（分数最小端/最大端由 direction 决定）
    direction: str = 'low'       # 'low'=买信号最小端(negative alpha) | 'high'=买最大端
    candidate_pool: int = 50     # 候选池大小
    max_positions: int = 20      # 最大持仓数
    position_size: float = 0.05  # 单仓目标占总资产比例（半仓滚动时每仓 5%）
    rebalance_days: int = 1      # 每 N 日调仓（模拟盘默认每日评估）
    # 多空支持（可选）：long_short='none' | 'decile'（D1 多 / D10 空）
    long_short: str = 'none'
    ls_decile: float = 0.10      # 多空分位
    hold_days: int = 1           # 持有交易日数：T+1 开盘买、T+2 卖出 = 持有 1 个交易日
    sell_price: str = 'close'    # 到期卖出价格：'close' | 'open'
    nav_price: str = 'close'     # 净值估值价格：默认收盘；open-open策略用'open'
    buy_gap_limit: float | None = None  # T+1开盘涨幅上限（相对T日收盘）；None=不过滤
    cash_reserve: float = 0.0  # 每日始终保留的现金比例；0=不保留
    signal_ffill: bool = True  # 信号缺失日是否回填最近一日；择时策略应设False
    # open-open每日换仓：T+1开盘先卖旧仓释放资金，再在同一开盘价买入新仓。
    # 该假设与理论回测一致，但实盘要求同一开盘竞价内完成卖旧买新。
    same_open_reinvest: bool = False
    # 持续持仓模式：只在股票进出目标组合时交易，而非每日全量轮换。
    # 用于对齐研究回测中“换手率=新增持仓数/当前持仓数”的成本口径。
    persistent_selection: bool = False
    forbid_repeat: bool = False  # True=禁止对已持仓（昨日在途）股票重复开仓，信号再选中则跳过
    market_filter: str = 'none'  # 'none' | 'hs300_ma20_ma60'
    pool: str = 'hs300_like'     # 见 simulation/data.py build_universe：all | top_n_amount | hs300_like
    initial_capital: float = 1_000_000.0
    precomputed: bool = False    # True=信号由外部预计算注入 cache, 引擎不现场调用 signal_fn
    # 低频预计算信号通常已在信号日完成universe过滤；
    # 不应再用当前交易日universe二次过滤，否则指数调出/ST变化会导致
    # 持仓在非信号日被错误强制调仓。
    filter_signal_by_universe: bool = True
    # 低频策略的有效信号日。None=每日评估；
    # 非信号日保持当前目标组合，不因信号缺失而清仓。
    signal_dates: set[str] | None = None
    # 每日对冲overlay：spec.hedge_overlay 为 {date_str: daily_return}。
    # 该收益已在 overlay 中包含对冲成本，直接乘到股票腿净值上。
    hedge_overlay: dict | None = None
    params: dict = field(default_factory=dict)

    @property
    def buy_low(self) -> bool:
        return self.direction == 'low'


# ───────────────────────────── 注册表 ─────────────────────────────

_REGISTRY: dict[str, StrategySpec] = {}


def register(spec: StrategySpec) -> StrategySpec:
    _REGISTRY[spec.strategy_id] = spec
    return spec


def get_spec(strategy_id: str) -> StrategySpec | None:
    return _REGISTRY.get(strategy_id)


def all_specs() -> list[StrategySpec]:
    return list(_REGISTRY.values())


def list_registered() -> list[dict]:
    return [
        {'strategy_id': s.strategy_id, 'name': s.name, 'description': s.description,
         'pool': s.pool, 'direction': s.direction, 'top_k': s.top_k,
         'long_short': s.long_short, 'initial_capital': s.initial_capital}
        for s in all_specs()
    ]

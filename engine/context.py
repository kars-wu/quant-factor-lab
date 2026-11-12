"""
BarContext：策略在 on_bar() 中能访问的运行时上下文。

防未来函数设计：
  ctx.data[symbol] 是 df.loc[:current_date] 的切片视图，
  仅包含截至当日收盘的历史数据，当日收盘价已知（用于信号计算），
  但当日不可成交——订单放入 pending，次日开盘价成交（T+1）。
"""

from dataclasses import dataclass, field


@dataclass
class PortfolioView:
    """持仓只读视图，避免策略直接修改组合状态"""
    cash: float
    total_value: float
    positions: dict        # {symbol: {'quantity': int, 'avg_cost': float, 'open_date': str}}
    nav: float             # 当前净值（以初始资金为 1.0 基准）


@dataclass
class BarContext:
    """
    每个交易日回调策略时传入的上下文。

    current_date : str
        当前交易日 'YYYY-MM-DD'

    data : dict[str, pd.DataFrame]
        {symbol: df.loc[:current_date]}
        df.index = trade_date（字符串）
        df.columns = [open, high, low, close, volume, amount]
        只包含截至 current_date 的历史数据，防止未来函数泄露。

    portfolio : PortfolioView
        当前持仓和资金快照（只读）

    universe : list[str]
        当日可交易的标的列表
    """
    current_date: str
    data: dict
    portfolio: PortfolioView
    universe: list = field(default_factory=list)

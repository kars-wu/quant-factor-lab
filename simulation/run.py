"""模拟盘命令行入口：历史回填 / 每日推进 / 状态查看。

用法：

    python -m simulation.run bootstrap --start 2021-01-01
    python -m simulation.run advance            # 行情已在库，只推进策略
    python -m simulation.run advance --lookback 300
    python -m simulation.run daily --fetch      # 先补行情，再推进
    python -m simulation.run status

口径（与回测一致）：
    T 日收盘生成信号 → T+1 开盘成交 → 持有 hold_days 个交易日后卖出；
    现金按当日可用资金滚动分配，成本由 strategies.cost_model 统一计算。
"""
from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from simulation import engine, state
from simulation.data import (
    benchmark_nav,
    build_universe,
    load_panel,
    trading_dates_of,
)
from simulation.strategies import all_specs

DAILY_LOOKBACK_DAYS = 420


def _setup_logger() -> None:
    logger.remove()
    logger.add(sys.stdout, level='INFO', format='{time:HH:mm:ss} | {level:<5} | {message}')


def register_examples() -> None:
    """注册示例策略；已被用户注册过的同名策略不会被覆盖。"""
    from simulation.examples import register_examples as _register
    _register()


def _context(
    lookback_days: Optional[int] = None,
    panel_start: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], List[str], Dict, Dict]:
    """加载面板并按策略用到的池构建掩码。"""
    if all_specs() == []:
        register_examples()

    start = panel_start
    if lookback_days:
        from data.database import Database
        db = Database()
        row = db.query_one(
            "SELECT trade_date FROM (SELECT DISTINCT trade_date FROM stock_daily "
            "ORDER BY trade_date DESC LIMIT ?) ORDER BY trade_date ASC LIMIT 1",
            (lookback_days,),
        )
        start = row[0] if row else start
        logger.info('面板截断到最近 {} 个交易日（起 {}）', lookback_days, start)

    panel, data = load_panel(start_date=start)
    dates = trading_dates_of(panel)

    pools: Dict[str, Dict[str, set]] = {}
    cache: Dict[str, pd.Series] = {}
    for spec in all_specs():
        if spec.pool in pools:
            continue
        if spec.precomputed:
            # 信号由外部注入（例如低频研究产物），这里不构建池
            continue
        mask = build_universe(panel, spec.pool)
        pools[spec.pool] = engine.build_pool_index(mask)
        nav = benchmark_nav(panel, mask)
        if not nav.empty:
            state.replace_benchmark(f'{spec.pool}_equal_weight', nav)
        logger.info('股票池 {}: {} 个交易日', spec.pool, len(pools[spec.pool]))
    return panel, data, dates, pools, cache


def _ensure_registered() -> None:
    for spec in all_specs():
        if state.get_strategy(spec.strategy_id) is None:
            state.register_strategy(spec.strategy_id, spec.name, spec.description,
                                    spec.initial_capital, spec.pool, {},
                                    spec.initial_capital)


def run_bootstrap(start: str = '2021-01-01') -> List[dict]:
    """从 start 起逐日回填所有策略。"""
    with state.SimLock():
        _setup_logger()
        state.mark_running()
        try:
            panel, data, dates, pools, cache = _context(panel_start=start)
            dates = [d for d in dates if d >= start]
            logger.info('回填区间 {} ~ {}，共 {} 个交易日', dates[0], dates[-1], len(dates))
            _ensure_registered()
            total = max(len(dates) - 1, 1)
            for i in range(1, len(dates)):
                engine.update_all(panel, data, dates, dates[i], cache, pools)
                if i % 20 == 0 or i == len(dates) - 1:
                    state.set_progress('advance', '历史回填', i, total,
                                       message=f'{dates[i]}（{i}/{total}）')
            state.mark_done(f'回填完成：{dates[0]} ~ {dates[-1]}')
            logger.info('回填完成：{} ~ {}', dates[0], dates[-1])
            return [{'status': 'ok', 'message': f'bootstrap {dates[0]} ~ {dates[-1]}'}]
        except Exception as exc:  # noqa: BLE001
            state.mark_error(str(exc))
            raise


def run_advance(lookback_days: int = DAILY_LOOKBACK_DAYS) -> List[dict]:
    """只推进策略（行情已入库）。"""
    with state.SimLock():
        _setup_logger()
        state.mark_running()
        try:
            panel, data, dates, pools, cache = _context(lookback_days=lookback_days)
            _ensure_registered()
            today = dates[-1]
            start_i = len(dates)
            for spec in all_specs():
                row = state.get_strategy(spec.strategy_id)
                last = (row or {}).get('last_run_date')
                if last:
                    try:
                        start_i = min(start_i, dates.index(last) + 1)
                    except ValueError:
                        start_i = 1
                else:
                    start_i = 1
            start_i = max(start_i, 1)
            pending = max(len(dates) - start_i, 1)
            logger.info('推进到 {}，待处理 {} 个交易日', today, len(dates) - start_i)
            state.set_progress('advance', '逐日推进策略', 0, pending, message=f'推进到 {today}')
            done = 0
            for i in range(start_i, len(dates)):
                engine.update_all(panel, data, dates, dates[i], cache, pools)
                done += 1
                if done % 5 == 0 or dates[i] == today:
                    state.set_progress('advance', '逐日推进策略', done, pending,
                                       message=f'{dates[i]}（{done}/{pending}）')
            state.mark_done(f'完成：已推进到 {today}')
            return [{'status': 'ok', 'message': f'advance to {today}'}]
        except Exception as exc:  # noqa: BLE001
            state.mark_error(str(exc))
            raise


def run_daily(fetch: bool = False, lookback_days: int = DAILY_LOOKBACK_DAYS) -> List[dict]:
    """每日任务：可选先增量更新行情，再推进策略。"""
    if fetch:
        _setup_logger()
        from data.database import Database
        from data.fetchers.akshare_fetcher import AkShareFetcher
        from data.updater import DataUpdater

        state.mark_running()
        state.set_progress('fetch', '增量更新行情', message='拉取最新日线/估值/股本')
        try:
            DataUpdater(Database(), AkShareFetcher()).update_incremental()
        except Exception as exc:  # noqa: BLE001
            state.mark_error(f'行情更新失败：{exc}')
            raise
    return run_advance(lookback_days=lookback_days)


def status() -> None:
    _setup_logger()
    rows = state.list_strategies()
    if not rows:
        print('尚无策略状态，先运行：python -m simulation.run bootstrap')
        return
    header = f'{"strategy_id":<24}{"名称":<30}{"最新净值":>10}{"最后推进":>14}'
    print(header)
    print('-' * len(header))
    for row in rows:
        navs = state.get_nav(row['strategy_id'])
        last_nav = float(navs['nav'].iloc[-1]) if len(navs) else float('nan')
        name = (row['name'] or '')[:28]
        last_run = row.get('last_run_date') or '-'
        print(f'{row["strategy_id"]:<24}{name:<30}{last_nav:>10.4f}{last_run:>14}')


def _arg(name: str, default: Optional[str] = None) -> Optional[str]:
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'
    if cmd == 'bootstrap':
        run_bootstrap(start=_arg('--start', '2021-01-01'))
    elif cmd == 'advance':
        run_advance(lookback_days=int(_arg('--lookback', DAILY_LOOKBACK_DAYS)))
    elif cmd == 'daily':
        run_daily(fetch='--fetch' in sys.argv,
                  lookback_days=int(_arg('--lookback', DAILY_LOOKBACK_DAYS)))
    elif cmd == 'status':
        status()
    else:
        print(__doc__)


if __name__ == '__main__':
    main()

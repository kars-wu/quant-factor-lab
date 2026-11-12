"""模拟盘数据层：股票池不看当天成交额、基准与策略同口径。"""
import numpy as np
import pandas as pd
import pytest

from simulation import data as sim_data
from simulation import engine
from simulation.strategies import StrategySpec, get_spec, register


class _FakeDB:
    """只实现 build_universe('hs300_like') 用到的 query。"""

    def __init__(self, rows):
        self._rows = rows

    def query(self, sql, params=()):
        return self._rows


def _panel(n_dates=30, symbols=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range('2020-01-01', periods=n_dates)
    cols = [f'S{i}' for i in range(symbols)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(idx), symbols)), 0)),
                         index=idx, columns=cols)
    amount = pd.DataFrame(rng.uniform(1e8, 2e8, (len(idx), symbols)), index=idx, columns=cols)
    panel = pd.concat({'close': close, 'amount': amount}, axis=1)
    panel.columns = panel.columns.set_names(['level_0', 'ticker'])
    panel = panel.stack('ticker')
    panel.index = panel.index.set_names(['date', 'ticker'])
    return panel


def test_top_n_amount_uses_previous_day_only():
    panel = _panel()
    base = sim_data.build_universe(panel, 'top_n_amount', top_n=2, lookback=5)
    # 篡改最后一天的成交额：最后一天的池成员不应发生变化（shift(1)）
    tampered = panel.copy()
    last = tampered.index.get_level_values('date').max()
    tampered.loc[(last, slice(None)), 'amount'] = 1e12
    after = sim_data.build_universe(tampered, 'top_n_amount', top_n=2, lookback=5)
    last_members = base.xs(last, level='date')
    tampered_members = after.xs(last, level='date')
    pd.testing.assert_series_equal(last_members, tampered_members)


def test_universe_membership_is_boolean_and_indexed():
    panel = _panel()
    mask = sim_data.build_universe(panel, 'all')
    assert mask.dtype == bool
    assert set(mask.index.names) == {'date', 'symbol'}
    assert mask.all()


def test_trading_dates_are_sorted_strings():
    panel = _panel()
    dates = sim_data.trading_dates_of(panel)
    assert dates == sorted(dates)
    assert all(isinstance(d, str) and len(d) == 10 for d in dates)


def test_benchmark_nav_starts_near_one():
    panel = _panel(n_dates=60, symbols=4)
    mask = sim_data.build_universe(panel, 'all')
    nav = sim_data.benchmark_nav(panel, mask)
    assert nav.iloc[0] == pytest.approx(1.0), '基准净值应从 1.0 起算'
    assert (nav > 0).all()


def test_pool_index_only_contains_selected():
    panel = _panel()
    mask = sim_data.build_universe(panel, 'top_n_amount', top_n=1, lookback=3)
    index = engine.build_pool_index(mask)
    sizes = {len(v) for v in index.values() if v}
    assert sizes and max(sizes) <= 1


def test_strategy_registry_roundtrip():
    spec = StrategySpec(strategy_id='unit_demo', name='单元测试', description='d',
                        signal_fn=lambda df: pd.Series(dtype=float))
    register(spec)
    assert get_spec('unit_demo') is spec
    assert spec.buy_low is True


def test_state_db_dir_is_created(tmp_path, monkeypatch):
    """state 模块必须自己建目录：否则首次启动 API 会 Unable to open database file。"""
    from simulation import state

    monkeypatch.setattr(state, 'DB_PATH', tmp_path / 'nested' / 'sim.db')
    monkeypatch.setattr(state, 'LOCK_PATH', tmp_path / 'nested' / 'sim.lock')
    state.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    state.init_db()
    state.register_strategy('unit_s1', '单元策略', 'd', 1e6, 'all', {}, 1e6)
    assert state.get_strategy('unit_s1')['name'] == '单元策略'
    assert [r['strategy_id'] for r in state.list_strategies()] == ['unit_s1']

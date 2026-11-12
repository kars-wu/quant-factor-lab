import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from multi_agent.datasets.synthetic import make_synthetic_view  # noqa: E402


@pytest.fixture(scope='session')
def view():
    """全样本合成视图（测试里不做真实研究，只看工程链路）。"""
    return make_synthetic_view(n_dates=400, n_symbols=80, seed=11, signal_strength=0.8)


@pytest.fixture()
def dev_view(view):
    dates = view.dates
    return view.slice(str(dates[60].date()), str(dates[320].date()))

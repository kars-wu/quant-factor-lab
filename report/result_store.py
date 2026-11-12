"""
回测结果持久化。

━━━ 背景 ━━━

当前回测结果只打印到控制台或CSV，无法后续对比。
日志是产品，回测结果也应该结构化存储。

━━━ 用法 ━━━

    from report.result_store import ResultStore

    store = ResultStore()
    run_id = store.save(
        strategy_name="连续大涨首次回调",
        config={'stop_method': 'atr', 'n_atr': 2.0, 'rr_ratio': 2.0},
        eval_result=result,
        portfolio_result=pf_result,
    )
    print(f"已保存，run_id={run_id}")

    # 对比多次回测
    df = store.compare(['run_001', 'run_002'])
    print(df)

    # 查看历史趋势
    trend = store.trend('连续大涨首次回调')
    print(trend[['run_date', 'sharpe', 'annual_return']])
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

import pandas as pd
from loguru import logger

from config.settings import DB_PATH


class ResultStore:
    """回测结果存储（SQLite）。"""

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._init_table()

    def _init_table(self):
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                run_id          TEXT PRIMARY KEY,
                run_date        TEXT NOT NULL,
                strategy_name   TEXT NOT NULL,
                config_json     TEXT NOT NULL,
                total_trades    INTEGER,
                win_rate        REAL,
                avg_net_return  REAL,
                sharpe          REAL,
                annual_return   REAL,
                max_drawdown    REAL,
                calmar          REAL,
                capital_util    REAL,
                turnover_rate   REAL,
                oos_sharpe      REAL,
                notes           TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_bt_strategy
            ON backtest_results (strategy_name, run_date)
        """)
        conn.commit()
        conn.close()

    def save(
        self,
        strategy_name: str,
        config: dict,
        eval_result=None,
        portfolio_result=None,
        oos_sharpe: float = None,
        notes: str = "",
    ) -> str:
        """保存一次回测结果，返回 run_id。"""
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        row = {
            'run_id': run_id,
            'run_date': run_date,
            'strategy_name': strategy_name,
            'config_json': json.dumps(config, ensure_ascii=False, default=str),
            'total_trades': 0,
            'win_rate': 0,
            'avg_net_return': 0,
            'sharpe': 0,
            'annual_return': 0,
            'max_drawdown': 0,
            'calmar': 0,
            'capital_util': 0,
            'turnover_rate': 0,
            'oos_sharpe': oos_sharpe,
            'notes': notes,
        }

        if eval_result:
            row.update({
                'total_trades': eval_result.total_trades,
                'win_rate': round(eval_result.win_rate, 4),
                'avg_net_return': round(eval_result.avg_net_return, 6),
                'sharpe': round(eval_result.sharpe, 4),
                'annual_return': round(eval_result.annual_return, 6),
                'max_drawdown': round(portfolio_result.max_drawdown if portfolio_result else 0, 6),
                'calmar': round(portfolio_result.calmar if portfolio_result else 0, 4),
                'capital_util': round(portfolio_result.capital_utilization if portfolio_result else 0, 4),
                'turnover_rate': round(portfolio_result.turnover_rate if portfolio_result else 0, 4),
            })

        if portfolio_result:
            row.update({
                'max_drawdown': round(portfolio_result.max_drawdown, 6),
                'calmar': round(portfolio_result.calmar, 4),
                'capital_util': round(portfolio_result.capital_utilization, 4),
                'turnover_rate': round(portfolio_result.turnover_rate, 4),
                'annual_return': round(portfolio_result.annual_return, 6),
                'sharpe': round(portfolio_result.sharpe, 4),
            })

        conn = sqlite3.connect(self._db_path)
        conn.execute(
            """INSERT OR REPLACE INTO backtest_results
               VALUES (:run_id, :run_date, :strategy_name, :config_json,
                       :total_trades, :win_rate, :avg_net_return, :sharpe,
                       :annual_return, :max_drawdown, :calmar, :capital_util,
                       :turnover_rate, :oos_sharpe, :notes)""",
            row,
        )
        conn.commit()
        conn.close()

        logger.info("回测结果已保存 | run_id={} | 策略={}", run_id, strategy_name)
        return run_id

    def compare(self, run_ids: list[str]) -> pd.DataFrame:
        """对比多次回测结果。"""
        conn = sqlite3.connect(self._db_path)
        placeholders = ','.join('?' * len(run_ids))
        df = pd.read_sql_query(
            f"SELECT * FROM backtest_results WHERE run_id IN ({placeholders})",
            conn, params=run_ids,
        )
        conn.close()
        return df

    def trend(self, strategy_name: str, limit: int = 50) -> pd.DataFrame:
        """查看某策略的历史回测趋势。"""
        conn = sqlite3.connect(self._db_path)
        df = pd.read_sql_query(
            "SELECT * FROM backtest_results WHERE strategy_name=? "
            "ORDER BY run_date DESC LIMIT ?",
            conn, params=[strategy_name, limit],
        )
        conn.close()
        return df

    def list_recent(self, limit: int = 20) -> pd.DataFrame:
        """列出最近的回测。"""
        conn = sqlite3.connect(self._db_path)
        df = pd.read_sql_query(
            "SELECT run_id, run_date, strategy_name, sharpe, annual_return, "
            "max_drawdown FROM backtest_results ORDER BY run_date DESC LIMIT ?",
            conn, params=[limit],
        )
        conn.close()
        return df

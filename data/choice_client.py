"""Choice API wrapper with a persistent, conservative quota guard.

The provider charges failed requests in some cases, so a request is reserved in
SQLite *before* it is sent.  This also makes concurrent jobs share one budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from data.database import Database


class ChoiceBudgetExceeded(RuntimeError):
    """Raised before a Choice request would exceed its configured budget."""


def week_start(day: date | None = None) -> str:
    current = day or date.today()
    return (current - timedelta(days=current.weekday())).isoformat()


def _actual_points(result: Any) -> int:
    data = getattr(result, "Data", None)
    if not isinstance(data, dict):
        return 0
    total = 0
    for values in data.values():
        if not isinstance(values, (list, tuple)):
            total += 1
            continue
        for value in values:
            if isinstance(value, (list, tuple)):
                total += len(value)
            else:
                total += 1
    return total


@dataclass(frozen=True)
class ChoiceBudgetConfig:
    ctr_weekly_limit: int = 10
    ctr_reserve: int = 1
    css_weekly_limit: int = 500_000
    css_reserve: int = 50_000
    csd_weekly_limit: int = 500_000
    csd_reserve: int = 50_000


class ChoiceBudget:
    def __init__(self, db: Database, config: ChoiceBudgetConfig | None = None):
        self.db = db
        self.config = config or ChoiceBudgetConfig()

    def usage(self, api_name: str) -> dict[str, int]:
        api = api_name.upper()
        row = self.db.query_one(
            """SELECT COUNT(*) AS calls,
                      COALESCE(SUM(estimated_points), 0) AS estimated_points,
                      COALESCE(SUM(actual_points), 0) AS actual_points
                 FROM choice_api_usage
                WHERE week_start=? AND api_name=?""",
            (week_start(), api),
        )
        return {
            "calls": int(row[0]),
            "estimated_points": int(row[1]),
            "actual_points": int(row[2]),
        }

    def _check(self, api_name: str, estimated_points: int) -> None:
        api = api_name.upper()
        used = self.usage(api)
        if api == "CTR":
            usable = max(0, self.config.ctr_weekly_limit - self.config.ctr_reserve)
            if used["calls"] + 1 > usable:
                raise ChoiceBudgetExceeded(
                    f"CTR 周预算不足：已记账 {used['calls']} 次，可用上限 {usable} 次"
                )
            return

        if api == "CSS":
            usable = max(0, self.config.css_weekly_limit - self.config.css_reserve)
        elif api == "CSD":
            usable = max(0, self.config.csd_weekly_limit - self.config.csd_reserve)
        else:
            return
        if used["estimated_points"] + estimated_points > usable:
            raise ChoiceBudgetExceeded(
                f"{api} 周预算不足：已预占 {used['estimated_points']} 点，"
                f"本次 {estimated_points} 点，可用上限 {usable} 点"
            )

    def reserve(self, api_name: str, dataset: str, estimated_points: int = 0) -> int:
        api = api_name.upper()
        conn = self.db.get_connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            self._check(api, estimated_points)
            cursor = conn.execute(
                """INSERT INTO choice_api_usage
                   (requested_at, week_start, api_name, dataset, estimated_points, success)
                   VALUES(?,?,?,?,?,0)""",
                (
                    datetime.now().isoformat(timespec="seconds"),
                    week_start(),
                    api,
                    dataset,
                    int(estimated_points),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)
        except Exception:
            conn.rollback()
            raise

    def finish(self, usage_id: int, result: Any) -> None:
        error_code = int(getattr(result, "ErrorCode", -1))
        self.db.execute(
            """UPDATE choice_api_usage
                  SET actual_points=?, success=?, error_code=?, error_message=?
                WHERE id=?""",
            (
                _actual_points(result),
                int(error_code == 0),
                error_code,
                str(getattr(result, "ErrorMsg", ""))[:500],
                usage_id,
            ),
        )
        self.db.commit()


class ChoiceClient:
    """Thin testable wrapper around an authenticated ``EmQuantAPI.c`` object."""

    def __init__(self, api: Any, budget: ChoiceBudget):
        self.api = api
        self.budget = budget

    def css(self, dataset: str, codes: list[str], indicators: list[str], options: str):
        points = len(codes) * len(indicators)
        usage_id = self.budget.reserve("CSS", dataset, points)
        try:
            result = self.api.css(",".join(codes), ",".join(indicators), options)
        except Exception as exc:
            self._finish_exception(usage_id, exc)
            raise
        self.budget.finish(usage_id, result)
        return result

    def csd(
        self, dataset: str, codes: list[str], indicators: list[str],
        start_date: str, end_date: str, estimated_dates: int, options: str = "",
    ):
        points = len(codes) * len(indicators) * max(1, estimated_dates)
        usage_id = self.budget.reserve("CSD", dataset, points)
        try:
            result = self.api.csd(
                ",".join(codes), ",".join(indicators), start_date, end_date, options
            )
        except Exception as exc:
            self._finish_exception(usage_id, exc)
            raise
        self.budget.finish(usage_id, result)
        return result

    def ctr(self, dataset: str, report_name: str, indicators: list[str], options: str):
        usage_id = self.budget.reserve("CTR", dataset, 0)
        try:
            result = self.api.ctr(report_name, ",".join(indicators), options)
        except Exception as exc:
            self._finish_exception(usage_id, exc)
            raise
        self.budget.finish(usage_id, result)
        return result

    def _finish_exception(self, usage_id: int, exc: Exception) -> None:
        self.db_update_exception(usage_id, exc)

    def db_update_exception(self, usage_id: int, exc: Exception) -> None:
        self.budget.db.execute(
            """UPDATE choice_api_usage
                  SET success=0, error_code=-1, error_message=? WHERE id=?""",
            (str(exc)[:500], usage_id),
        )
        self.budget.db.commit()

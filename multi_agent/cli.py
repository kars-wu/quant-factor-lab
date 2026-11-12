"""命令行入口。

    python -m multi_agent.cli demo                       # 合成数据端到端跑一遍
    python -m multi_agent.cli run --dataset sqlite --start 2018-01-01 --top-n 500
    python -m multi_agent.cli run --dataset sqlite --holdout-start 2025-01-01
    python -m multi_agent.cli validate --file examples/hypotheses.sample.jsonl
    python -m multi_agent.cli evolve --rounds 3
    python -m multi_agent.cli report --dir outputs/multi_agent/<run_id>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

from multi_agent import dsl
from multi_agent.contracts import Budget
from multi_agent.evolution import evolve, summarize
from multi_agent.orchestration import ResearchLoop, RunConfig, print_report
from multi_agent.policy import ALL_AUDIT_CHECKS, PolicyState


def _setup_logger(verbose: bool = True) -> None:
    logger.remove()
    if verbose:
        logger.add(sys.stdout, level='INFO', format='{time:HH:mm:ss} | {level:<5} | {message}')


def _policy(args: argparse.Namespace) -> PolicyState:
    checks = args.checks.split(',') if args.checks else list(ALL_AUDIT_CHECKS)
    return PolicyState(version=args.policy_version, audit_checks=checks,
                       proposal_quota=args.proposal_quota,
                       max_full_validations=args.max_validations)


def _config(args: argparse.Namespace) -> RunConfig:
    budget = Budget(
        max_proposals=args.max_proposals,
        max_variants=args.max_variants,
        max_train_evaluations=args.max_evaluations,
        max_full_validations=args.max_validations,
        max_llm_calls=args.max_llm_calls,
    )
    return RunConfig(
        objective=args.objective,
        dataset=args.dataset,
        db_path=args.db,
        start=args.start,
        end=args.end,
        universe=args.universe,
        top_n=args.top_n,
        selection_top=args.selection_top,
        hold_days=args.hold_days,
        dev_start=args.dev_start,
        dev_end=args.dev_end,
        holdout_start=args.holdout_start,
        budget=budget,
        policy=_policy(args),
        out_dir=args.out_dir,
        registry_path=args.registry,
        seed=args.seed,
        cost_stress_bps=args.cost_stress,
        design_strict=args.design_strict,
        synthetic_symbols=args.symbols,
        synthetic_dates=args.dates,
        synthetic_strength=args.synthetic_strength,
        run_id=args.run_id,
    )


# ─────────────────────────── 子命令 ───────────────────────────

def cmd_demo(args: argparse.Namespace) -> int:
    args.dataset = 'synthetic'
    args.db = None
    loop = ResearchLoop(_config(args))
    report = loop.run()
    return 0 if report['candidates'] else 0


def cmd_run(args: argparse.Namespace) -> int:
    loop = ResearchLoop(_config(args))
    loop.run()
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.file)
    if not path.exists():
        print(f'文件不存在: {path}')
        return 1
    rows: List[Dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f'[FAIL] 第 {i} 行不是合法 JSON: {exc}')

    errors = 0
    seen = set()
    for row in rows:
        hid = row.get('hypothesis_id') or row.get('local_ref') or '<missing>'
        if hid in seen:
            print(f'[FAIL] {hid}: 重复 id')
            errors += 1
        seen.add(hid)
        for field in ('family', 'mechanism', 'falsifiable_prediction', 'direction'):
            if not row.get(field):
                print(f'[FAIL] {hid}: 缺少字段 {field}')
                errors += 1
        if row.get('direction') not in ('high', 'low'):
            print(f'[FAIL] {hid}: direction 必须是 high/low')
            errors += 1
        spec = row.get('spec')
        if not isinstance(spec, dict):
            print(f'[FAIL] {hid}: 缺少 spec')
            errors += 1
            continue
        errs = dsl.validate(spec)
        for e in errs:
            print(f'[FAIL] {hid}: {e}')
            errors += 1
        if dsl.warmup(spec) > 250:
            print(f'[FAIL] {hid}: warmup {dsl.warmup(spec)} > 250')
            errors += 1
        for e in dsl.check_time_safety(spec):
            print(f'[WARN] {hid}: {e}')

    print(f'{len(rows)} 个假设，{errors} 个错误')
    return 1 if errors else 0


def cmd_evolve(args: argparse.Namespace) -> int:
    cfg = _config(args)
    loop = ResearchLoop(cfg)
    report = loop.run()
    view = loop.dev_view
    result = evolve(
        traces=loop.traces,
        view=view,
        policy=loop.policy,
        execution=loop.protocol.execution,
        evaluation=loop.protocol.evaluation,
        rounds=args.rounds,
        out_path=str(Path(cfg.out_dir) / report['run_id'] / 'evolution.json'),
    )
    print(summarize(result))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    path = Path(args.dir)
    if path.is_file():
        report_path = path
    else:
        report_path = path / 'report.json'
    if not report_path.exists():
        print(f'找不到 report.json: {report_path}')
        return 1
    report = json.loads(report_path.read_text(encoding='utf-8'))
    print_report(report)
    return 0


# ─────────────────────────── 参数 ───────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='python -m multi_agent.cli',
                                description='多 agent 因子研究系统')
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument('--dataset', default='synthetic', choices=['synthetic', 'sqlite'])
        sp.add_argument('--db', default=None, help='sqlite 模式下的数据库路径')
        sp.add_argument('--start', default='2018-01-01')
        sp.add_argument('--end', default=None)
        sp.add_argument('--universe', default='top_n_amount',
                        choices=['all', 'top_n_amount', 'none'])
        sp.add_argument('--top-n', dest='top_n', type=int, default=500,
                        help='股票池规模（universe top_n）')
        sp.add_argument('--selection-top', dest='selection_top', type=int, default=100,
                        help='每次选股数量（截面 top_k）')
        sp.add_argument('--hold-days', dest='hold_days', type=int, default=20,
                        help='每笔持仓交易日数；每日开 1/hold_days 资金的一笔')
        sp.add_argument('--dev-start', default=None)
        sp.add_argument('--dev-end', default=None)
        sp.add_argument('--holdout-start', default=None)
        sp.add_argument('--seed', type=int, default=7)
        sp.add_argument('--symbols', type=int, default=120)
        sp.add_argument('--dates', type=int, default=700)
        sp.add_argument('--synthetic-strength', dest='synthetic_strength', type=float,
                        default=0.35, help='合成数据里的信号强度（只影响 demo）')
        sp.add_argument('--cost-stress', type=float, default=50.0)
        sp.add_argument('--design-strict', dest='design_strict', action='store_true',
                        help='把"算子交互"设计要求变成硬门槛（≥3 类算子 + 交叉/相对算子 + 禁止线性叠加）')
        sp.add_argument('--objective', default='在给定研究池内寻找扣成本后有增量信息的短周期因子')
        sp.add_argument('--out-dir', default='outputs/multi_agent')
        sp.add_argument('--registry', default=None)
        sp.add_argument('--run-id', default=None)
        sp.add_argument('--policy-version', default='v0')
        sp.add_argument('--checks', default=','.join(ALL_AUDIT_CHECKS))
        sp.add_argument('--proposal-quota', type=int, default=4)
        sp.add_argument('--max-proposals', type=int, default=24)
        sp.add_argument('--max-variants', type=int, default=24)
        sp.add_argument('--max-evaluations', type=int, default=48)
        sp.add_argument('--max-validations', type=int, default=8)
        sp.add_argument('--max-llm-calls', type=int, default=400)
        sp.add_argument('--quiet', action='store_true')

    demo = sub.add_parser('demo', help='合成数据端到端演示')
    common(demo)

    run = sub.add_parser('run', help='跑一次完整研究循环')
    common(run)

    val = sub.add_parser('validate', help='校验假设文件（DSL/字段/窗口）')
    val.add_argument('--file', required=True)

    ev = sub.add_parser('evolve', help='跑一轮研究 + 外层策略进化')
    common(ev)
    ev.add_argument('--rounds', type=int, default=3)

    rep = sub.add_parser('report', help='打印已有 run 的报告')
    rep.add_argument('--dir', required=True)
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logger(not getattr(args, 'quiet', False))
    handlers = {
        'demo': cmd_demo,
        'run': cmd_run,
        'validate': cmd_validate,
        'evolve': cmd_evolve,
        'report': cmd_report,
    }
    return handlers[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())

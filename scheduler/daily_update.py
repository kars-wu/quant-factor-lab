"""
每日数据更新入口。

用法：
  # 全市场初始化（沪深两市主板+创业板+科创板+ETF，约 3-4 小时，支持断点续传）
  python scheduler/daily_update.py --init-all

  # 只初始化指定标的（快速测试）
  python scheduler/daily_update.py --init --symbols 510300 510500 000300 000001

  # 每日增量更新（接 cron: 0 18 * * 1-5）
  python scheduler/daily_update.py --incremental

  # 查看当前数据库状态
  python scheduler/daily_update.py --status
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click
from loguru import logger

from data.database import Database
from data.fetchers.akshare_fetcher import AkShareFetcher
from data.updater import DataUpdater
from scheduler.monitor import TaskMonitor


def _setup_logger(verbose: bool):
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stdout, level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <5}</level> | {message}"
    )
    if not verbose:
        # 同时写文件，方便长时间任务追踪
        logger.add(
            "logs/update_{time:YYYYMMDD}.log",
            level="INFO",
            rotation="1 day",
            retention="7 days",
            encoding="utf-8",
        )


@click.group()
def cli():
    pass


@cli.command("init-all")
@click.option("--start-date", default="2015-01-01", show_default=True,
              help="历史数据起始日期")
@click.option("--no-etf", is_flag=True, help="跳过 ETF")
@click.option("--no-resume", is_flag=True,
              help="不使用断点续传，强制重新拉取所有标的")
@click.option("--verbose", is_flag=True)
def init_all(start_date, no_etf, no_resume, verbose):
    """
    全市场初始化：沪深两市主板+创业板+科创板+ETF。

    约 6000+ 标的，50 RPM 限速下约 3-4 小时。
    支持断点续传（中途 Ctrl+C 后重新运行会跳过已完成的标的）。
    """
    _setup_logger(verbose)
    os.makedirs("logs", exist_ok=True)

    db = Database()
    fetcher = AkShareFetcher()
    updater = DataUpdater(db, fetcher)

    logger.info("全市场初始化 | 起始日期: {} | ETF: {} | 断点续传: {}",
                start_date, not no_etf, not no_resume)

    updater.init_all(
        symbols=None,
        start_date=start_date,
        include_etf=not no_etf,
        resume=not no_resume,
    )


@cli.command("init")
@click.option("--symbols", multiple=True, required=True,
              help="标的代码，可多个，如: --symbols 510300 --symbols 000001")
@click.option("--types", default="",
              help="标的类型映射，格式: symbol:type,... 如 000300:index,510300:etf。未指定的默认按代码自动判断")
@click.option("--start-date", default="2015-01-01", show_default=True)
@click.option("--no-resume", is_flag=True)
@click.option("--verbose", is_flag=True)
def init_specific(symbols, types, start_date, no_resume, verbose):
    """
    初始化指定标的，自动推断 stock/etf/index 类型。
    """
    _setup_logger(verbose)

    db = Database()
    fetcher = AkShareFetcher()
    updater = DataUpdater(db, fetcher)

    # 解析手动指定的类型
    manual_types = {}
    if types:
        for part in types.split(","):
            if ":" in part:
                s, t = part.split(":", 1)
                manual_types[s.strip()] = t.strip()

    # 自动推断未指定的类型
    asset_types = {}
    for s in symbols:
        if s in manual_types:
            asset_types[s] = manual_types[s]
        elif s.startswith(("51", "52", "15", "58")):
            asset_types[s] = "etf"
        elif s.startswith(("000", "399", "880")):
            asset_types[s] = "index"
        else:
            asset_types[s] = "stock"

    logger.info("初始化标的: {}", dict(asset_types))

    updater.init_symbols(
        symbols=list(symbols),
        asset_types=asset_types,
        start_date=start_date,
        resume=not no_resume,
    )


@cli.command("incremental")
@click.option("--verbose", is_flag=True)
def incremental(verbose):
    """每日盘后增量更新（建议 cron: 0 18 * * 1-5）"""
    _setup_logger(verbose)
    os.makedirs("logs", exist_ok=True)

    db = Database()
    fetcher = AkShareFetcher()
    updater = DataUpdater(db, fetcher)

    logger.info("开始增量更新...")
    monitor = TaskMonitor(name="每日增量更新")
    try:
        updater.update_incremental()
        monitor.record_success("增量更新完成")
    except Exception as e:
        monitor.record_failure("增量更新", str(e))
        logger.error("增量更新失败: {}", e)
        raise
    finally:
        report = monitor.finish()
        logger.info("\n{}", report.summary())


@cli.command("fetch-shares")
@click.option("--symbols", multiple=True, help="指定股票代码（不指定则更新全部缺失）")
@click.option("--verbose", is_flag=True)
def fetch_shares(symbols, verbose):
    """
    拉取并更新股票总股本/流通股本（存入 stock_shares 表）。

    用于支持换手率和市值过滤的策略评测。
    全市场约 5000 只个股，50 RPM 限速下约 2-3 小时。

    首次运行（更新全部缺失）:
      python scheduler/daily_update.py fetch-shares

    只更新指定股票:
      python scheduler/daily_update.py fetch-shares --symbols 600410 000001
    """
    _setup_logger(verbose)
    os.makedirs("logs", exist_ok=True)

    db = Database()
    fetcher = AkShareFetcher()
    updater = DataUpdater(db, fetcher)

    sym_list = list(symbols) if symbols else None
    updater.update_shares(symbols=sym_list)


@cli.command("fetch-valuation")
@click.option("--symbols", multiple=True, help="指定股票代码（不指定则取流动性最高的200只）")
@click.option("--limit", default=200, type=int, show_default=True,
              help="不指定 --symbols 时自动选取的高流动性股票数量")
@click.option("--verbose", is_flag=True)
def fetch_valuation(symbols, limit, verbose):
    """
    拉取总市值和PB数据（百度估值接口）。

    数据特点:
      - 已处理所有股本变动（送转/可转债转股/增发/回购），无未来函数风险
      - total_mv / pb / pe_ttm 存入 stock_valuation 表
      - 查询时按交易日 forward-fill 到最近交易日

    用法:
      # 拉取指定股票
      python scheduler/daily_update.py fetch-valuation --symbols 002460 --symbols 600519

      # 自动选取200只高流动性股票
      python scheduler/daily_update.py fetch-valuation

      # 拉取更多
      python scheduler/daily_update.py fetch-valuation --limit 500
    """
    _setup_logger(verbose)
    os.makedirs("logs", exist_ok=True)

    db = Database()
    fetcher = AkShareFetcher()
    updater = DataUpdater(db, fetcher)
    monitor = TaskMonitor(name="估值数据更新")

    if symbols:
        sym_list = list(symbols)
    else:
        # 自动选取高流动性股票（用数据库最新日期而非date('now')）
        rows = db.query(
            "SELECT sd.symbol FROM stock_daily sd "
            "JOIN stock_info si ON sd.symbol = si.symbol "
            "WHERE si.asset_type='stock' AND si.is_active=1 "
            "AND sd.trade_date >= (SELECT MAX(trade_date) FROM stock_daily) "
            "GROUP BY sd.symbol ORDER BY AVG(sd.amount) DESC LIMIT ?",
            (limit,)
        )
        sym_list = [r[0] for r in rows]

    logger.info("拉取 {} 只股票的估值数据", len(sym_list))
    try:
        updater.update_valuation(sym_list)
        monitor.record_success("估值更新完成")
    except Exception as e:
        monitor.record_failure("估值更新", str(e))
        logger.error("估值更新失败: {}", e)
        raise
    finally:
        report = monitor.finish()
        logger.info("\n{}", report.summary())
    _setup_logger(False)
    db = Database()

    print("\n=== 数据库状态 ===\n")

    # 各类标的数量
    rows = db.query(
        "SELECT asset_type, COUNT(*) as cnt FROM stock_info GROUP BY asset_type ORDER BY cnt DESC"
    )
    print("stock_info 标的登记：")
    for r in rows:
        print(f"  {r[0]:<8} {r[1]} 条")

    # 已完成数据拉取的标的数
    done = db.query_one("SELECT COUNT(*) FROM update_log WHERE last_price_date IS NOT NULL")
    total = db.query_one("SELECT COUNT(*) FROM stock_info")
    print(f"\n已完成数据拉取: {done[0]} / {total[0]}")

    # stock_daily 数据量
    daily_stats = db.query_one(
        "SELECT COUNT(*), MIN(trade_date), MAX(trade_date), COUNT(DISTINCT symbol) FROM stock_daily"
    )
    if daily_stats and daily_stats[0]:
        print(f"\nstock_daily: {daily_stats[0]:,} 条")
        print(f"  日期范围: {daily_stats[1]} ~ {daily_stats[2]}")
        print(f"  覆盖标的: {daily_stats[3]} 只")

    # adj_factor
    factor_stats = db.query_one(
        "SELECT COUNT(*), COUNT(DISTINCT symbol) FROM adj_factor"
    )
    if factor_stats and factor_stats[0]:
        print(f"\nadj_factor: {factor_stats[0]:,} 条，覆盖 {factor_stats[1]} 只")

    # index_daily
    idx_stats = db.query_one(
        "SELECT COUNT(*), COUNT(DISTINCT symbol) FROM index_daily"
    )
    if idx_stats and idx_stats[0]:
        print(f"\nindex_daily: {idx_stats[0]:,} 条，覆盖 {idx_stats[1]} 只")

    # 最近更新
    recent = db.query(
        "SELECT symbol, last_price_date, updated_at FROM update_log "
        "ORDER BY updated_at DESC LIMIT 5"
    )
    if recent:
        print("\n最近更新的标的：")
        for r in recent:
            print(f"  {r[0]}  last={r[1]}  at={r[2]}")

    print()


@cli.command("fetch-financial")
@click.option("--symbols", multiple=True, help="指定股票代码（不指定则全市场）")
@click.option("--force", is_flag=True, help="强制重拉所有股票（默认跳过已有数据的）")
@click.option("--verbose", is_flag=True)
def fetch_financial(symbols, force, verbose):
    """
    拉取三大报表（利润表 + 资产负债表 + 现金流量表），存入 financial_report 表。

    数据源：新浪财经 stock_financial_report_sina 接口
    每只股票 3 次 API 调用，50RPM 限速下约 3.6 秒/只。
    全市场约 5200 只，约 5 小时。

    数据特点：
      - 全历史季报覆盖（上市以来全部报告期）
      - announce_date 字段已存入，回测时按 announce_date <= T 切片避免未来函数
      - UPSERT 写入，新数据覆盖旧版本

    用法：
      # 首次初始化（跳过已有，断点续传）
      python scheduler/daily_update.py fetch-financial

      # 指定股票
      python scheduler/daily_update.py fetch-financial --symbols 600519 --symbols 002460

      # 强制重拉所有股票（增量更新用，UPSERT 覆盖旧数据）
      python scheduler/daily_update.py fetch-financial --force
    """
    _setup_logger(verbose)
    os.makedirs("logs", exist_ok=True)

    db = Database()
    fetcher = AkShareFetcher()
    updater = DataUpdater(db, fetcher)
    monitor = TaskMonitor(name="财报数据更新")

    sym_list = list(symbols) if symbols else None
    try:
        updater.update_financial_reports(symbols=sym_list, skip_existing=not force)
        monitor.record_success("财报更新完成")
    except Exception as e:
        monitor.record_failure("财报更新", str(e))
        logger.error("财报更新失败: {}", e)
        raise
    finally:
        report = monitor.finish()
        logger.info("\n{}", report.summary())


@cli.command("audit")
@click.option("--verbose", is_flag=True)
def audit(verbose):
    """输出数据库健康检查摘要。"""
    _setup_logger(verbose)
    db = Database()
    updater = DataUpdater(db, AkShareFetcher())

    print("\n=== 数据健康审计 ===\n")

    rows = db.query(
        "SELECT asset_type, COUNT(*) cnt FROM stock_info GROUP BY asset_type ORDER BY cnt DESC"
    )
    print("stock_info 资产分布：")
    for r in rows:
        print(f"  {r['asset_type']:<8} {r['cnt']} 条")

    summary = db.query_one(
        "SELECT COUNT(*) total, "
        "SUM(last_price_date IS NOT NULL) loaded, "
        "SUM(load_status='pending') pending, "
        "SUM(load_status='failed') failed "
        "FROM update_log"
    )
    print("\nupdate_log 状态：")
    print(f"  total={summary['total']} loaded={summary['loaded']} pending={summary['pending']} failed={summary['failed']}")

    bad_stock = db.query_one(
        "SELECT COUNT(*) cnt FROM stock_daily "
        "WHERE open <= 0 OR high <= 0 OR low <= 0 OR close <= 0 "
        "OR high < low OR high < open OR high < close OR low > open OR low > close"
    )
    bad_index = db.query_one(
        "SELECT COUNT(*) cnt FROM index_daily "
        "WHERE open <= 0 OR high <= 0 OR low <= 0 OR close <= 0 "
        "OR high < low OR high < open OR high < close OR low > open OR low > close"
    )
    print("\n坏行情行数：")
    print(f"  stock_daily={bad_stock['cnt']} index_daily={bad_index['cnt']}")

    mismatches = db.query(
        "SELECT symbol, market, asset_type FROM stock_info ORDER BY symbol, market"
    )
    mismatch_count = 0
    for row in mismatches:
        expected = updater._infer_market(row['symbol'], row['asset_type'])
        if row['market'] != expected:
            mismatch_count += 1
    print("\n主数据 market 异常：")
    print(f"  mismatch={mismatch_count}")

    pending_by_type = db.query(
        "SELECT si.asset_type, COUNT(*) cnt "
        "FROM update_log ul JOIN stock_info si ON ul.symbol = si.symbol "
        "WHERE ul.load_status='pending' AND ul.last_price_date IS NULL "
        "GROUP BY si.asset_type ORDER BY cnt DESC"
    )
    if pending_by_type:
        print("\nPending 资产分布：")
        for r in pending_by_type:
            print(f"  {r['asset_type']:<8} {r['cnt']}")

        samples = db.query(
            "SELECT ul.symbol, si.asset_type, si.market, si.name "
            "FROM update_log ul JOIN stock_info si ON ul.symbol = si.symbol "
            "WHERE ul.load_status='pending' AND ul.last_price_date IS NULL "
            "ORDER BY si.asset_type, ul.symbol LIMIT 10"
        )
        if samples:
            print("  样本:")
            for r in samples:
                print(f"    {r['symbol']} {r['asset_type']} {r['market']} {r['name']}")

    quality = db.query(
        "SELECT table_name, issue_type, COUNT(*) cnt FROM data_quality_log "
        "GROUP BY table_name, issue_type ORDER BY cnt DESC LIMIT 10"
    )
    if quality:
        print("\n最近审计问题汇总：")
        for r in quality:
            print(f"  {r['table_name']:<12} {r['issue_type']:<24} {r['cnt']}")

    print()


@cli.command("audit-fields")
@click.option("--verbose", is_flag=True)
def audit_fields(verbose):
    """输出关键表字段覆盖率。"""
    _setup_logger(verbose)
    db = Database()

    print("\n=== 字段覆盖率审计 ===\n")

    audits = [
        (
            "stock_daily",
            ["open", "high", "low", "close", "volume", "amount", "turnover_rate", "outstanding_share", "pe_ttm", "pb", "total_mv", "circ_mv"],
            "symbol",
        ),
        (
            "stock_valuation",
            ["total_mv", "pb", "pe_ttm"],
            "symbol",
        ),
        (
            "stock_shares",
            ["outstanding_share", "circ_share"],
            "symbol",
        ),
        (
            "adj_factor",
            ["hfq_factor"],
            "symbol",
        ),
    ]

    for table, cols, symbol_col in audits:
        metrics = [
            "COUNT(*) AS total_rows",
            f"COUNT(DISTINCT {symbol_col}) AS total_symbols",
        ]
        for col in cols:
            metrics.append(f"SUM({col} IS NOT NULL) AS {col}_nonnull")
            metrics.append(
                f"COUNT(DISTINCT CASE WHEN {col} IS NOT NULL THEN {symbol_col} END) AS {col}_sym_cov"
            )

        total_row = db.query_one(f"SELECT {', '.join(metrics)} FROM {table}")
        total = total_row['total_rows'] if total_row else 0
        sym_total = total_row['total_symbols'] if total_row else 0
        print(f"[{table}] rows={total} symbols={sym_total}")
        if total == 0:
            print("  (empty)\n")
            continue

        for col in cols:
            nonnull = total_row[f'{col}_nonnull'] or 0
            sym_cov = total_row[f'{col}_sym_cov'] or 0
            row_ratio = nonnull / total if total else 0
            sym_ratio = sym_cov / sym_total if sym_total else 0
            print(f"  {col:<18} row_nonnull={nonnull:<10} row_ratio={row_ratio:>6.2%}  sym_cov={sym_cov:<8} sym_ratio={sym_ratio:>6.2%}")
        print()


@cli.command("retry-pending")
@click.option("--asset-type", type=click.Choice(["stock", "etf", "index"]), default=None,
              help="只重试指定资产类型")
@click.option("--limit", type=int, default=10, show_default=True,
              help="最多重试多少个 pending 标的")
@click.option("--verbose", is_flag=True)
def retry_pending(asset_type, limit, verbose):
    """对 pending 且无价格数据的标的做小批量重试。"""
    _setup_logger(verbose)
    os.makedirs("logs", exist_ok=True)

    db = Database()
    fetcher = AkShareFetcher()
    updater = DataUpdater(db, fetcher)

    result = updater.retry_pending_symbols(asset_type=asset_type, limit=limit)

    print("\n=== Pending 重试 ===\n")
    print(f"selected={result['selected']} success={result['success']} failed={result['failed']}")
    if result['symbols']:
        print("symbols:")
        for s in result['symbols']:
            print(f"  {s}")
    print()


@cli.command("backfill-missing-fields")
@click.option("--field-group", type=click.Choice(["shares", "valuation", "daily-extras"]), required=True,
              help="要补拉的缺失字段组")
@click.option("--limit", type=int, default=20, show_default=True,
              help="最多补拉多少个标的")
@click.option("--verbose", is_flag=True)
def backfill_missing_fields(field_group, limit, verbose):
    """从接口重拉缺失字段。"""
    _setup_logger(verbose)
    os.makedirs("logs", exist_ok=True)

    db = Database()
    fetcher = AkShareFetcher()
    updater = DataUpdater(db, fetcher)

    print("\n=== 缺失字段补拉 ===\n")

    if field_group == "shares":
        symbols = updater.get_symbols_missing_shares(limit=limit)
        updater.update_shares(symbols=symbols)
        print(f"shares: selected={len(symbols)}")
        for s in symbols:
            print(f"  {s}")

    elif field_group == "valuation":
        symbols = updater.get_symbols_missing_valuation(limit=limit)
        updater.update_valuation(symbols)
        print(f"valuation: selected={len(symbols)}")
        for s in symbols:
            print(f"  {s}")

    else:
        result = updater.backfill_daily_extras(limit=limit)
        print(f"daily-extras: selected={result['selected']} success={result['success']} failed={result['failed']}")
        for s in result['symbols']:
            print(f"  {s}")

    print()


@cli.command("repair")
@click.option("--fix-market", is_flag=True, help="修正 stock_info 中 market 异常")
@click.option("--backfill-update-log", is_flag=True, help="按现有数据回填 update_log")
@click.option("--scan-bad-rows", is_flag=True, help="扫描历史坏 OHLC 行并写入 data_quality_log")
@click.option("--delete-invalid", is_flag=True, help="配合 --scan-bad-rows 删除坏 OHLC 行")
@click.option("--verbose", is_flag=True)
def repair(fix_market, backfill_update_log, scan_bad_rows, delete_invalid, verbose):
    """执行数据库修复任务。"""
    _setup_logger(verbose)
    db = Database()
    fetcher = AkShareFetcher()
    updater = DataUpdater(db, fetcher)

    if not any([fix_market, backfill_update_log, scan_bad_rows]):
        fix_market = backfill_update_log = scan_bad_rows = True

    print("\n=== 数据修复 ===\n")

    if fix_market:
        result = updater.repair_stock_info_markets()
        print(f"market 修复: fixed={result['fixed']} merged={result['merged']}")

    if backfill_update_log:
        result = updater.backfill_update_log()
        print(
            "update_log 回填: "
            f"updated={result['updated']} pending={result['pending']} "
            f"failed_kept={result['failed_kept']} total={result['total']}"
        )

    if scan_bad_rows:
        for table_name in ("stock_daily", "index_daily"):
            result = updater.scan_historical_bad_rows(table_name=table_name, delete_invalid=delete_invalid)
            print(
                f"{table_name} 坏行扫描: invalid={result['invalid_rows']} deleted={result['deleted']}"
            )

    print()


# 兼容旧写法（直接 python daily_update.py --init）
@cli.command("legacy-init")
@click.option("--init", "mode", flag_value="init")
@click.option("--incremental", "mode", flag_value="incremental", default=True)
@click.option("--symbols", multiple=True)
@click.option("--start-date", default="2015-01-01")
@click.option("--include-etf/--no-etf", default=True)
@click.option("--verbose", is_flag=True)
@click.pass_context
def legacy(ctx, mode, symbols, start_date, include_etf, verbose):
    """旧版兼容入口（不推荐）"""
    if mode == "init" and symbols:
        ctx.invoke(init_specific, symbols=symbols, start_date=start_date, verbose=verbose)
    elif mode == "init":
        ctx.invoke(init_all, start_date=start_date, no_etf=not include_etf, verbose=verbose)
    else:
        ctx.invoke(incremental, verbose=verbose)


if __name__ == "__main__":
    cli()

-- ============================================================
-- 股票/ETF/指数 元信息表
-- ============================================================
CREATE TABLE IF NOT EXISTS stock_info (
    symbol       TEXT NOT NULL,
    name         TEXT NOT NULL,
    market       TEXT NOT NULL,          -- 'SH' / 'SZ'
    asset_type   TEXT NOT NULL,          -- 'stock' / 'etf' / 'index'
    sector       TEXT,                   -- '主板' / '创业板' / '科创板'
    industry     TEXT,
    list_date    TEXT,
    delist_date  TEXT,
    is_active    INTEGER NOT NULL DEFAULT 1,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (symbol, market)
);

CREATE INDEX IF NOT EXISTS idx_stock_info_type ON stock_info (asset_type, is_active);

-- ============================================================
-- 日线行情主表（不复权原始数据，永不修改历史行）
-- ============================================================
CREATE TABLE IF NOT EXISTS stock_daily (
    symbol       TEXT NOT NULL,
    trade_date   TEXT NOT NULL,          -- 'YYYY-MM-DD'
    open         REAL NOT NULL,
    high         REAL NOT NULL,
    low          REAL NOT NULL,
    close        REAL NOT NULL,
    volume       REAL NOT NULL,          -- 成交量（手）
    amount       REAL,                   -- 成交额（元）
    turnover_rate REAL,
    outstanding_share REAL,             -- 总股本（股，时间序列，防未来函数）
    pe_ttm       REAL,
    pb           REAL,
    total_mv     REAL,
    circ_mv      REAL,
    PRIMARY KEY (symbol, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_stock_daily_date ON stock_daily (trade_date);

-- ============================================================
-- 后复权因子表（每日全量 INSERT OR REPLACE）
--
-- 定义：hfq_price(t) = raw_price(t) × hfq_factor(t)
--       上市首日 hfq_factor = 1.0（最小值，随时间单调不减）
--
-- 前复权推导公式（回测使用）：
--       qfq_price(t) = raw_price(t) × hfq_factor(t) / hfq_factor(end_date)
--
-- 注意：volume 不做复权处理
-- ============================================================
CREATE TABLE IF NOT EXISTS adj_factor (
    symbol       TEXT NOT NULL,
    trade_date   TEXT NOT NULL,          -- 'YYYY-MM-DD'
    hfq_factor   REAL NOT NULL,
    PRIMARY KEY (symbol, trade_date)
);

-- ============================================================
-- 指数/ETF 日线表（结构同 stock_daily，无财务字段）
-- ============================================================
CREATE TABLE IF NOT EXISTS index_daily (
    symbol       TEXT NOT NULL,
    trade_date   TEXT NOT NULL,
    open         REAL NOT NULL,
    high         REAL NOT NULL,
    low          REAL NOT NULL,
    close        REAL NOT NULL,
    volume       REAL,
    amount       REAL,
    PRIMARY KEY (symbol, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_index_daily_date ON index_daily (trade_date);

-- ============================================================
-- 交易日历
-- ============================================================
CREATE TABLE IF NOT EXISTS trade_calendar (
    trade_date     TEXT PRIMARY KEY,
    is_open        INTEGER NOT NULL,     -- 1=交易日, 0=非交易日
    pretrade_date  TEXT,
    nexttrade_date TEXT
);

-- ============================================================
-- 股票股本信息表（用于计算换手率和市值）
-- outstanding_share: 总股本（股）
-- circ_share: 流通股本（股）
-- 注：AkShare 的 stock_zh_a_daily 返回总股本，暂用总股本近似流通股本。
--       换手率计算可能因此有偏差，后续应接入独立流通股本数据源。
-- ============================================================
CREATE TABLE IF NOT EXISTS stock_shares (
    symbol           TEXT PRIMARY KEY,
    outstanding_share REAL,             -- 总股本（股）
    circ_share        REAL,             -- 流通股本（股，暂与总股本相同）
    updated_at        TEXT NOT NULL
);

-- ============================================================
-- 个股估值数据（百度接口，总市值 + PB）
-- 注意: 日期为日历日(含周末)，非交易日。查询时需 forward-fill 到最近交易日。
-- ============================================================
CREATE TABLE IF NOT EXISTS stock_valuation (
    symbol       TEXT NOT NULL,
    trade_date   TEXT NOT NULL,          -- 'YYYY-MM-DD'（日历日）
    total_mv     REAL,                   -- 总市值（亿元）
    pb           REAL,                   -- 市净率
    pe_ttm       REAL,                   -- 市盈率(TTM)
    PRIMARY KEY (symbol, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_valuation_symbol ON stock_valuation (symbol, trade_date);

-- ============================================================
-- 财务报表（三大报表合一，按报告期 + 公告日）
--
-- 关键防未来函数设计：
--   1. announce_date 必填，回测时按 WHERE announce_date <= T 切片
--   2. 同一 report_date 可有多条记录（预告/快报/正式），用 (data_source, is_preliminary) 区分
--   3. 同一 report_date 同一数据源只保留最新版本（INSERT OR REPLACE）
--
-- 数据说明：
--   - 利润表/现金流量表为"本期累计"（Q1=1-3月累计，半年报=1-6月累计），TTM 需计算
--   - 资产负债表为"期末值"（时点数据）
--   - 金额单位：元（与 AkShare 返回一致，未做单位换算）
-- ============================================================
CREATE TABLE IF NOT EXISTS financial_report (
    symbol            TEXT NOT NULL,
    report_date       TEXT NOT NULL,          -- 报告期末日 'YYYY-MM-DD'（如 2024-03-31）
    report_type       TEXT NOT NULL,          -- 'annual' / 'q1' / 'interim' / 'q3'
    announce_date     TEXT NOT NULL,          -- 公告日（避免未来函数的关键）
    -- 利润表
    revenue              REAL,                -- 营业收入
    operating_income     REAL,                -- 营业总收入
    operating_cost       REAL,                -- 营业成本
    sell_expense         REAL,                -- 销售费用
    admin_expense        REAL,                -- 管理费用
    rd_expense           REAL,                -- 研发费用
    finance_expense      REAL,                -- 财务费用
    net_profit           REAL,                -- 净利润（含少数股东）
    net_profit_attr      REAL,                -- 归母净利润
    deducted_net_profit   REAL,                -- 扣非净利润（新浪接口不直接返回，预留）
    eps                   REAL,                -- 基本每股收益
    -- 资产负债表
    total_assets          REAL,
    total_liabilities     REAL,
    equity_attr           REAL,                -- 归母股东权益
    equity                REAL,                -- 全部股东权益（含少数股东）
    cash                  REAL,                -- 货币资金
    accounts_receivable   REAL,                -- 应收账款
    inventory             REAL,
    goodwill              REAL,
    contract_liability    REAL,                -- 合同负债
    interest_bearing_debt REAL,                -- 有息负债（短期借款+长期借款+应付债券+一年内到期）
    bvps                  REAL,                -- 每股净资产（= 归母权益 / 实收资本）
    -- 现金流量表
    operating_cf          REAL,                -- 经营现金流净额
    investing_cf          REAL,                -- 投资现金流净额
    financing_cf          REAL,                -- 筹资现金流净额
    capex                 REAL,                -- 资本性支出（购建固定/无形/其他长期资产支付的现金）
    free_cf              REAL,                 -- 自由现金流（= 经营现金流 - 资本开支）
    -- 元信息
    is_preliminary    INTEGER DEFAULT 0,        -- 1=业绩预告/快报, 0=正式财报
    data_source       TEXT NOT NULL DEFAULT 'sina',  -- 'sina' / 'ths' / 'em'
    is_audited        TEXT,                    -- '是' / '未审计'（仅年报有审计意见）
    updated_at        TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    PRIMARY KEY (symbol, report_date, data_source, is_preliminary)
);

CREATE INDEX IF NOT EXISTS idx_fin_announce ON financial_report (symbol, announce_date);
CREATE INDEX IF NOT EXISTS idx_fin_report   ON financial_report (report_date);

-- ============================================================
-- 增量更新断点记录（驱动每日增量更新）
-- ============================================================
CREATE TABLE IF NOT EXISTS update_log (
    symbol          TEXT PRIMARY KEY,
    last_price_date TEXT,                -- stock_daily 最新入库日期
    last_factor_date TEXT,               -- adj_factor 最新全量覆盖日期
    load_status     TEXT NOT NULL DEFAULT 'pending',  -- pending/success/failed/partial
    last_error      TEXT,
    updated_at      TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);

-- ============================================================
-- 数据质量审计日志
-- ============================================================
CREATE TABLE IF NOT EXISTS data_quality_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT NOT NULL,
    trade_date    TEXT,
    table_name    TEXT NOT NULL,
    issue_type    TEXT NOT NULL,
    detail        TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);

CREATE INDEX IF NOT EXISTS idx_quality_symbol_date
ON data_quality_log (symbol, trade_date);

-- ============================================================
-- Choice 扩展因子数据
--
-- 设计原则：
--   1. 事件数据保留公告日和生效日，不扩成每日宽表；
--   2. 截面数据保留查询时点/可用时点，回测只能使用 available_date <= T；
--   3. 所有 Choice 调用均写入本地账本，避免超过周度额度。
-- ============================================================
CREATE TABLE IF NOT EXISTS choice_api_usage (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    requested_at      TEXT NOT NULL,
    week_start        TEXT NOT NULL,
    api_name          TEXT NOT NULL,
    dataset           TEXT NOT NULL,
    estimated_points  INTEGER NOT NULL DEFAULT 0,
    actual_points     INTEGER NOT NULL DEFAULT 0,
    success           INTEGER NOT NULL DEFAULT 0,
    error_code        INTEGER,
    error_message     TEXT
);

CREATE INDEX IF NOT EXISTS idx_choice_api_usage_week
ON choice_api_usage (week_start, api_name);

CREATE TABLE IF NOT EXISTS choice_sync_state (
    dataset       TEXT NOT NULL,
    partition_key TEXT NOT NULL,
    api_name      TEXT NOT NULL,
    status        TEXT NOT NULL,
    row_count     INTEGER NOT NULL DEFAULT 0,
    error_code    INTEGER,
    error_message TEXT,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (dataset, partition_key)
);

CREATE TABLE IF NOT EXISTS choice_corporate_event (
    dataset       TEXT NOT NULL,
    event_key     TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    announce_date TEXT,
    effective_date TEXT,
    report_date   TEXT,
    direction     TEXT,
    quantity      REAL,
    amount        REAL,
    ratio         REAL,
    price         REAL,
    status        TEXT,
    payload_json  TEXT NOT NULL,
    source_api    TEXT NOT NULL,
    fetched_at    TEXT NOT NULL,
    PRIMARY KEY (dataset, event_key)
);

CREATE INDEX IF NOT EXISTS idx_choice_event_symbol_announce
ON choice_corporate_event (symbol, announce_date);
CREATE INDEX IF NOT EXISTS idx_choice_event_dataset_date
ON choice_corporate_event (dataset, announce_date);

CREATE TABLE IF NOT EXISTS choice_factor_snapshot (
    domain         TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    as_of_date     TEXT NOT NULL,
    available_date TEXT NOT NULL,
    report_date    TEXT NOT NULL DEFAULT '',
    forecast_year  INTEGER NOT NULL DEFAULT 0,
    payload_json   TEXT NOT NULL,
    source_api     TEXT NOT NULL,
    fetched_at     TEXT NOT NULL,
    PRIMARY KEY (domain, symbol, as_of_date, report_date, forecast_year)
);

CREATE INDEX IF NOT EXISTS idx_choice_snapshot_available
ON choice_factor_snapshot (domain, available_date, symbol);

CREATE TABLE IF NOT EXISTS choice_factor_value (
    domain         TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    as_of_date     TEXT NOT NULL,
    available_date TEXT NOT NULL,
    report_date    TEXT NOT NULL DEFAULT '',
    forecast_year  INTEGER NOT NULL DEFAULT 0,
    field          TEXT NOT NULL,
    value_num      REAL,
    value_text     TEXT,
    source_api     TEXT NOT NULL,
    fetched_at     TEXT NOT NULL,
    PRIMARY KEY (domain, symbol, as_of_date, report_date, forecast_year, field)
);

CREATE INDEX IF NOT EXISTS idx_choice_value_factor_date
ON choice_factor_value (domain, field, available_date, symbol);

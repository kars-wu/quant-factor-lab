import sqlite3
import threading
from pathlib import Path
from loguru import logger

from config.settings import DB_PATH


class Database:
    """
    SQLite 连接管理。
    - WAL 模式：读写并发，回测读取不阻塞数据更新写入
    - 线程本地连接：每个线程独立连接，避免多线程共享问题
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.initialize_schema()

    def get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")   # 64MB 缓存
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return self._local.conn

    def initialize_schema(self):
        schema_path = Path(__file__).parent / "schema.sql"
        schema_sql = schema_path.read_text(encoding="utf-8")
        conn = self.get_connection()
        conn.executescript(schema_sql)
        self._run_migrations(conn)
        conn.commit()
        logger.debug("数据库 Schema 初始化完成: {}", self.db_path)

    def _run_migrations(self, conn: sqlite3.Connection):
        """对已存在数据库执行轻量级幂等迁移。"""
        self._ensure_column(conn, "stock_valuation", "pe_ttm", "REAL")
        self._ensure_column(conn, "update_log", "load_status", "TEXT NOT NULL DEFAULT 'pending'")
        self._ensure_column(conn, "update_log", "last_error", "TEXT")

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str):
        cols = {
            row[1]
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column in cols:
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
        logger.debug("数据库迁移: {}.{} 已补齐", table, column)

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        conn = self.get_connection()
        return conn.execute(sql, params)

    def execute_many(self, sql: str, params_list: list) -> int:
        if not params_list:
            return 0
        conn = self.get_connection()
        cursor = conn.executemany(sql, params_list)
        conn.commit()
        return cursor.rowcount

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: tuple = ()):
        return self.execute(sql, params).fetchone()

    def commit(self):
        self.get_connection().commit()

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

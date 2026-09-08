"""
会话历史持久化（P1 补债：简历「MySQL 存对话」的实现）。

语法：SQLAlchemy 2.0 Engine + text() 参数化 SQL；MySQL 走 pymysql，缺省落 SQLite。
原理：
  - 写路径：每轮问答 append 一行 chat_turns（session_id 维度），替代原先进程内 _HISTORY dict
    —— dict 重启即失，持久化后 /api/history 语义不变但跨重启可用
  - 读路径：按 session_id 取全部轮次，拼回 app 层的历史结构（q/a/elapsed_ms/ts）
  - 双引擎与 P2 同一工程模式（URL 抽象换库）：MYSQL_URL 配了连 MySQL（生产形态），
    没配自动落 SQLite（data/db/chats.db，开发零配置）——两项目互相印证「库无关设计」
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone

from sqlalchemy import create_engine, text

from src.core.config import settings

_DDL = """
CREATE TABLE IF NOT EXISTS chat_turns (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    question TEXT NOT NULL,
    answer MEDIUMTEXT,
    elapsed_ms INT DEFAULT 0,
    cached TINYINT DEFAULT 0,
    created_at DATETIME NOT NULL,
    INDEX idx_session (session_id, id)
)
"""

# SQLite 方言没有 AUTO_INCREMENT/MEDIUMTEXT/TINYINT，用等价建表语句
_DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS chat_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,
    elapsed_ms INTEGER DEFAULT 0,
    cached INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
)
"""

_lock = threading.Lock()
_engine = None
_backend = "unknown"


def _get_engine():
    """惰性单例：MYSQL_URL 优先（生产），否则 SQLite（开发零配置）。失败快速抛错。"""
    global _engine, _backend
    if _engine is not None:
        return _engine
    if settings.mysql_url:
        _engine = create_engine(settings.mysql_url, pool_pre_ping=True, future=True)
        _backend = "mysql"
        ddl = _DDL
    else:
        db_path = settings.data_dir / "db" / "chats.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            future=True,
        )
        _backend = "sqlite"
        ddl = _DDL_SQLITE
    with _engine.begin() as con:
        con.execute(text(ddl))
    return _engine


def ping() -> dict:
    """启动探活：返回实际使用的后端与连通性（写进日志，面试可演示「MySQL 真连」）。"""
    eng = _get_engine()
    with eng.connect() as con:
        con.execute(text("SELECT 1"))
    return {"backend": _backend, "ok": True}


def append_turn(session_id: str, question: str, answer: str,
                elapsed_ms: int = 0, cached: bool = False) -> None:
    """追加一轮问答。任何写库异常都向上抛——由调用方决定是否影响主流程。"""
    eng = _get_engine()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with _lock, eng.begin() as con:
        con.execute(
            text("INSERT INTO chat_turns (session_id, question, answer, elapsed_ms,"
                 " cached, created_at) VALUES (:s, :q, :a, :e, :c, :t)"),
            {"s": session_id, "q": question, "a": answer,
             "e": int(elapsed_ms), "c": int(cached), "t": now},
        )


def load_history(session_id: str) -> list[dict]:
    """读取整段会话历史，返回 app 层结构 [{q, a, elapsed_ms, ts}, ...]。"""
    eng = _get_engine()
    with eng.connect() as con:
        rows = con.execute(
            text("SELECT question, answer, elapsed_ms, created_at FROM chat_turns"
                 " WHERE session_id = :s ORDER BY id"),
            {"s": session_id},
        ).fetchall()
    out = []
    for q, a, e, t in rows:
        # created_at 在 MySQL 是 datetime、SQLite 是字符串，统一转 epoch 秒
        ts = t.timestamp() if hasattr(t, "timestamp") else _parse_dt(t)
        out.append({"q": q, "a": a or "", "elapsed_ms": int(e or 0), "ts": ts})
    return out


def _parse_dt(s: str) -> float:
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    return 0.0


def stats() -> dict:
    """运维视图：总轮次/会话数/最近写入时间（展示「存储层真实在用」）。"""
    eng = _get_engine()
    with eng.connect() as con:
        row = con.execute(text(
            "SELECT COUNT(*), COUNT(DISTINCT session_id), MAX(created_at)"
            " FROM chat_turns")).fetchone()
    n, sessions, last = row
    return {"backend": _backend, "turns": int(n), "sessions": int(sessions),
            "last_turn_at": str(last) if last else None}

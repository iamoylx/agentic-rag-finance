"""
存储层单测：SQLite 回退路径（CI / 无 MySQL 环境下的零依赖验证）。
MySQL 真连路径不进单测（依赖外部服务），由 scripts/verify_store.py 手动验收。
"""
from __future__ import annotations

import os

# 必须在 import store 前清掉 MYSQL_URL，强制走 SQLite 回退
os.environ["MYSQL_URL"] = ""

from src.api import store  # noqa: E402


def test_sqlite_roundtrip(monkeypatch, tmp_path):
    # 强制 engine 指向临时库，避免污染开发库
    monkeypatch.setattr(store, "_engine", None)
    import sqlalchemy
    eng = sqlalchemy.create_engine(
        f"sqlite:///{tmp_path/'chats.db'}",
        connect_args={"check_same_thread": False}, future=True)
    monkeypatch.setattr(store, "_engine", eng)
    monkeypatch.setattr(store, "_backend", "sqlite")
    with eng.begin() as con:
        con.execute(sqlalchemy.text(store._DDL_SQLITE))

    store.append_turn("t1", "问题A", "答案A", 100, False)
    store.append_turn("t1", "问题B", "答案B", 200, True)
    store.append_turn("t2", "别的会话", "x", 0, False)

    h1 = store.load_history("t1")
    assert [r["q"] for r in h1] == ["问题A", "问题B"]
    assert h1[1]["elapsed_ms"] == 200
    assert store.load_history("t2")[0]["q"] == "别的会话"
    assert store.load_history("nope") == []

    s = store.stats()
    assert s["turns"] == 3 and s["sessions"] == 2

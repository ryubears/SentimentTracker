"""
SQLite persistence. Weight snapshots are stored per period to avoid look-ahead in evaluation.
"""

from __future__ import annotations
from pathlib import Path
import json
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
  post_id TEXT PRIMARY KEY, account TEXT, created_at TEXT, text TEXT, likes INT, score REAL);
CREATE TABLE IF NOT EXISTS periods (
  period_ts TEXT PRIMARY KEY, horizon TEXT, agg_score REAL, agg_uniform REAL,
  price_now REAL, price_later REAL, realized_return REAL, resolved INT DEFAULT 0);
CREATE TABLE IF NOT EXISTS account_signals (
  period_ts TEXT, account TEXT, signal REAL, n_posts INT, PRIMARY KEY (period_ts, account));
CREATE TABLE IF NOT EXISTS weight_snapshots (
  period_ts TEXT PRIMARY KEY, weights_json TEXT, state_json TEXT);
"""


def connect(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con

def save_posts(con, posts: list[dict]) -> None:
    con.executemany(
        "INSERT OR IGNORE INTO posts VALUES (:post_id,:account,:created_at,:text,:likes,:score)", posts)
    con.commit()

def save_period(con, period_ts: str, horizon: str, agg: float, agg_uniform: float,
                price_now: float, signals: dict[str, float], counts: dict[str, int],
                weights: dict[str, float], state_json: str) -> None:
    con.execute("INSERT OR REPLACE INTO periods (period_ts,horizon,agg_score,agg_uniform,price_now) "
                "VALUES (?,?,?,?,?)", (period_ts, horizon, agg, agg_uniform, price_now))
    con.executemany("INSERT OR REPLACE INTO account_signals VALUES (?,?,?,?)",
                    [(period_ts, a, s, counts[a]) for a, s in signals.items()])
    con.execute("INSERT OR REPLACE INTO weight_snapshots VALUES (?,?,?)",
                (period_ts, json.dumps(weights), state_json))
    con.commit()

def latest_state(con) -> str | None:
    row = con.execute("SELECT state_json FROM weight_snapshots ORDER BY period_ts DESC LIMIT 1").fetchone()
    return row[0] if row else None

def unresolved_periods(con) -> list[tuple[str, float]]:
    return con.execute("SELECT period_ts, price_now FROM periods WHERE resolved=0").fetchall()

def resolve_period(con, period_ts: str, price_later: float, ret: float) -> None:
    con.execute("UPDATE periods SET price_later=?, realized_return=?, resolved=1 WHERE period_ts=?",
                (price_later, ret, period_ts))
    con.commit()

def signals_for(con, period_ts: str) -> dict[str, float]:
    return dict(con.execute("SELECT account, signal FROM account_signals WHERE period_ts=?", (period_ts,)))

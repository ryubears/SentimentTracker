"""
SQLite persistence. Weight snapshots are stored per period to avoid look-ahead in evaluation.
"""

from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
  post_id TEXT PRIMARY KEY, account TEXT, created_at TEXT, text TEXT, likes INT, score REAL,
  engagement REAL, engagement_at TEXT);
CREATE TABLE IF NOT EXISTS periods (
  period_ts TEXT PRIMARY KEY, horizon TEXT, agg_score REAL, agg_uniform REAL,
  price_now REAL, price_later REAL, realized_return REAL, resolved INT DEFAULT 0);
CREATE TABLE IF NOT EXISTS account_signals (
  period_ts TEXT, account TEXT, signal REAL, n_posts INT, PRIMARY KEY (period_ts, account));
CREATE TABLE IF NOT EXISTS weight_snapshots (
  period_ts TEXT PRIMARY KEY, weights_json TEXT, state_json TEXT);
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS klines (
  symbol TEXT, ts TEXT, close REAL, PRIMARY KEY (symbol, ts));
"""


def connect(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    cols = [r[1] for r in con.execute("PRAGMA table_info(posts)")]
    if "engagement" not in cols:
        # Pre-engagement db: add the column; likes are the best proxy for old rows.
        con.execute("ALTER TABLE posts ADD COLUMN engagement REAL")
        con.execute("UPDATE posts SET engagement = likes")
    if "engagement_at" not in cols:
        # NULL means never measured at maturity, so the refresh pass picks these up.
        con.execute("ALTER TABLE posts ADD COLUMN engagement_at TEXT")
    con.commit()
    return con

def save_posts(con, posts: list[dict]) -> None:
    con.executemany(
        "INSERT OR IGNORE INTO posts (post_id,account,created_at,text,likes,score,engagement,engagement_at) "
        "VALUES (:post_id,:account,:created_at,:text,:likes,:score,:engagement,:engagement_at)", posts)
    con.commit()

def stale_engagement_posts(con, matured_before_iso: str, limit: int) -> list[str]:
    """
    Posts old enough that their engagement has settled (created before the
    cutoff) but whose engagement was last measured — if ever — less than a day
    after posting. These are due for one refresh via fetch_engagement.
    """
    out = []
    for pid, created, measured in con.execute(
            "SELECT post_id, created_at, engagement_at FROM posts WHERE created_at<=? "
            "ORDER BY created_at", (matured_before_iso,)):
        if measured is None or (datetime.fromisoformat(measured)
                                < datetime.fromisoformat(created) + timedelta(days=1)):
            out.append(pid)
            if len(out) == limit:
                break
    return out

def update_engagement(con, post_ids: list[str], fresh: dict[str, float], at_iso: str) -> None:
    """Record refreshed engagement. Posts absent from `fresh` (deleted or
    protected) keep their last value but are stamped so they aren't retried."""
    con.executemany(
        "UPDATE posts SET engagement=COALESCE(?, engagement), engagement_at=? WHERE post_id=?",
        [(fresh.get(pid), at_iso, pid) for pid in post_ids])
    con.commit()

def scored_posts_in_range(con, accounts: list[str], start_iso: str, end_iso: str) -> list[dict]:
    """Every scored post cached for these accounts in (start, end]. Backfill buckets
    from here rather than from one fetch's results, so a transient shortfall from the
    X API can't silently drop an account's history out of the recomputed periods."""
    marks = ",".join("?" * len(accounts))
    rows = con.execute(
        f"SELECT account, created_at, score, engagement FROM posts "
        f"WHERE account IN ({marks}) AND created_at>? AND created_at<=? AND score IS NOT NULL",
        (*accounts, start_iso, end_iso)).fetchall()
    return [{"account": a, "created_at": c, "score": s, "engagement": e} for a, c, s, e in rows]

def engagement_history(con, account: str, before_iso: str) -> list[float]:
    """Engagement of the account's cached posts strictly before a cutoff."""
    return [r[0] for r in con.execute(
        "SELECT engagement FROM posts WHERE account=? AND created_at<? AND engagement IS NOT NULL",
        (account, before_iso))]

def save_period(con, period_ts: str, horizon: str, agg: float, agg_uniform: float,
                price_now: float, signals: dict[str, float], counts: dict[str, int],
                weights: dict[str, float], state_json: str) -> None:
    con.execute("INSERT OR REPLACE INTO periods (period_ts,horizon,agg_score,agg_uniform,price_now) "
                "VALUES (?,?,?,?,?)", (period_ts, horizon, agg, agg_uniform, price_now))
    # Replace the period's signal set wholesale: an account that contributed on an
    # earlier run but not this one must disappear, or its stale row keeps claiming a
    # contribution the saved agg_score no longer includes.
    con.execute("DELETE FROM account_signals WHERE period_ts=?", (period_ts,))
    con.executemany("INSERT INTO account_signals VALUES (?,?,?,?)",
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

def latest_period_ts(con, horizon: str) -> str | None:
    return con.execute("SELECT MAX(period_ts) FROM periods WHERE horizon=?", (horizon,)).fetchone()[0]

def cached_klines(con, symbol: str, start_iso: str, end_iso: str) -> list[tuple[str, float]]:
    return con.execute("SELECT ts, close FROM klines WHERE symbol=? AND ts BETWEEN ? AND ? "
                       "ORDER BY ts", (symbol, start_iso, end_iso)).fetchall()

def save_klines(con, symbol: str, rows: list[tuple[str, float]]) -> None:
    con.executemany("INSERT OR IGNORE INTO klines VALUES (?,?,?)",
                    [(symbol, ts, close) for ts, close in rows])
    con.commit()

def get_meta(con, key: str) -> str | None:
    row = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None

def set_meta(con, key: str, value: str) -> None:
    con.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))
    con.commit()

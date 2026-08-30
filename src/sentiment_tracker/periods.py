"""
Period boundary alignment shared by run_period.py and backfill.py.

"1d" periods must anchor to a fixed hour of day: flooring wall-clock time to
the hour would let a 05:43 run and a 09:15 re-run each start a parallel
period series (05:00-anchored vs 09:00-anchored) in the same db. The anchor
is pinned in the db's meta table the first time it is needed — inferred from
the earliest existing period for a pre-existing db, midnight UTC for a fresh
one — so re-runs extend and replace the same series forever after, even if
old rows get pruned.
"""

from __future__ import annotations
from datetime import datetime, timedelta

from . import db

HORIZON = {"1d": timedelta(days=1), "1h": timedelta(hours=1)}

def anchor_hour(con, horizon: str) -> int:
    """Hour-of-day this db's series is anchored to. Pinned in meta on first call."""
    key = f"anchor_hour_{horizon}"
    stored = db.get_meta(con, key)
    if stored is not None:
        return int(stored)
    row = con.execute("SELECT MIN(period_ts) FROM periods WHERE horizon=?", (horizon,)).fetchone()
    hour = datetime.fromisoformat(row[0]).hour if row and row[0] else 0
    db.set_meta(con, key, str(hour))
    return hour

def current_boundary(now: datetime, step: timedelta, anchor: int = 0) -> datetime:
    """Most recent period boundary at or before now: the floored hour for the
    hourly step, the last occurrence of hour `anchor` for the daily step."""
    now = now.replace(minute=0, second=0, microsecond=0)
    if step < timedelta(days=1):
        return now
    boundary = now.replace(hour=anchor)
    if boundary > now:
        boundary -= timedelta(days=1)
    return boundary

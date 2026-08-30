"""
Period boundary alignment shared by run_period.py and backfill.py.

"1d" periods must anchor to a fixed hour of day: flooring wall-clock time to
the hour would let a 05:43 run and a 09:15 re-run each start a parallel
period series (05:00-anchored vs 09:00-anchored) in the same db. The anchor
is the hour of the earliest period already in the db, so re-runs extend and
replace the existing series; a fresh db anchors at midnight UTC.
"""

from __future__ import annotations
from datetime import datetime, timedelta

HORIZON = {"1d": timedelta(days=1), "1h": timedelta(hours=1)}

def anchor_hour(con, horizon: str) -> int:
    """Hour-of-day the db's existing series is anchored to, or 0 for a fresh db."""
    row = con.execute("SELECT MIN(period_ts) FROM periods WHERE horizon=?", (horizon,)).fetchone()
    return datetime.fromisoformat(row[0]).hour if row and row[0] else 0

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

"""
Period boundary alignment shared by run_period.py and backfill.py.

"1d" periods must anchor to a fixed hour of day: flooring wall-clock time to
the hour would let a 05:43 run and a 09:15 re-run each start a parallel
period series (05:00-anchored vs 09:00-anchored) in the same db. The anchor
is midnight UTC, pinned in the db's meta table the first time it is needed
so it can never drift; a different hour can be adopted deliberately by
setting the meta key before the first run.
"""

from __future__ import annotations
from datetime import datetime, timedelta

from . import db

HORIZON = {"1d": timedelta(days=1), "1h": timedelta(hours=1)}

def anchor_hour(con, horizon: str) -> int:
    """Hour-of-day this db's series is anchored to: midnight UTC unless a
    different anchor was pinned in the meta table. Pinned on first call."""
    key = f"anchor_hour_{horizon}"
    stored = db.get_meta(con, key)
    if stored is not None:
        return int(stored)
    db.set_meta(con, key, "0")
    return 0

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

"""
Bootstrap N days of history before going live.

Paginates historical X posts and BTC price history, then replays the exact
Phase A / Phase B loop from run_period.py one period at a time, so the backfilled
series is built the same way the live one is.

Because we have the whole price history up front, "waiting for the horizon to
elapse" just means "the next period boundary in the walk", instead of waiting
on a real clock the way run_period.py does when run on a schedule.

Usage:
  python backfill.py --days 60
"""

from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import argparse
import bisect
import sys
import pandas as pd
import yaml

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from sentiment_tracker import db, engine, prices, runtime, sentiment
from sentiment_tracker.fetch_x import fetch_historical_posts
from sentiment_tracker.periods import HORIZON, anchor_hour, current_boundary

def period_boundaries(start: datetime, end: datetime, step: timedelta) -> list[pd.Timestamp]:
    ts, out = start, []
    while ts <= end:
        out.append(pd.Timestamp(ts))
        ts += step
    return out

def bucket_by_period(posts: list[dict], periods: list[pd.Timestamp],
                     step: timedelta) -> dict[pd.Timestamp, dict[str, list[float]]]:
    """Assign each post's score to the period boundary t such that
    t - step < created_at <= t — the same (now - step, now] window run_period.py's
    fetch_posts(since=now-step, until=now) covers."""
    by_period: dict[pd.Timestamp, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for p in posts:
        ts = pd.Timestamp(p["created_at"])
        i = bisect.bisect_left(periods, ts)
        if i < len(periods) and periods[i] - step < ts <= periods[i]:
            by_period[periods[i]][p["account"]].append(p["score"])
    return by_period

def main(cfg_path: str | None = None, days: int = 60,
         workers: int = sentiment.DEFAULT_WORKERS) -> None:
    runtime.load_secrets()
    cfg = runtime.load_config(cfg_path)
    handles = [a["handle"] for a in cfg["accounts"]]
    step = HORIZON[cfg["horizon"]]
    con = db.connect(cfg["db_path"])
    now = current_boundary(datetime.now(timezone.utc), step, anchor_hour(con, cfg["horizon"]))
    start = now - timedelta(days=days)

    print(f"loading {days}d of {cfg['symbol']} price history (cached hours are not refetched)...")
    klines = prices.get_klines(con, cfg["symbol"], pd.Timestamp(start) - step, pd.Timestamp(now))

    print(f"fetching historical posts for {len(handles)} accounts since {start.date()}...")
    posts = fetch_historical_posts(handles, since=start, until=now)
    known = dict(con.execute("SELECT post_id, score FROM posts WHERE score IS NOT NULL"))
    fresh = [p for p in posts if p["post_id"] not in known]
    print(f"scoring {len(fresh)} posts ({len(posts) - len(fresh)} already scored in the db) "
         f"with backend={cfg['sentiment']['backend']} across {workers} workers...", flush=True)
    scored = sentiment.score_many([p["text"] for p in fresh],
                                  {**cfg["sentiment"], "horizon": cfg["horizon"]}, workers=workers)
    for p, s in zip(fresh, scored):
        p["score"] = s
    for p in posts:
        if p["post_id"] in known:
            p["score"] = known[p["post_id"]]
    # Posts that failed to score are left out entirely, so a later run retries them
    # rather than the cache serving a placeholder score forever.
    db.save_posts(con, [p for p in posts if p["score"] is not None])

    periods = period_boundaries(start, now, step)
    # Bucket from the db, not from `posts`: the cache is the union of every fetch, so a
    # short fetch (a rate limit, an account temporarily returning less) can no longer
    # drop that account's history out of the recomputed periods. Also makes re-runs
    # idempotent — the same window always rebuilds from the same set.
    cached = db.scored_posts_in_range(con, handles, (start - step).isoformat(), now.isoformat())
    print(f"bucketing {len(cached)} cached posts ({len(posts)} returned by this fetch)...")
    by_period = bucket_by_period(cached, periods, step)

    resolved_n = 0
    for t in periods:
        # Phase A: resolve anything whose horizon has elapsed by this boundary.
        resolved_n += len(engine.resolve_matured(con, klines, t, step))
        # Phase B: score this period.
        engine.score_period(con, t, by_period.get(t, {}), prices.price_at(klines, t),
                            cfg["horizon"],
                            relevance_min=cfg.get("signal", {}).get("relevance_min", 0.0))

    print(f"backfilled {len(periods)} periods ({resolved_n} resolved) "
         f"from {start.date()} to {now.date()}")
    print("the most recent period is left unresolved, same as a normal run_period.py call — "
         "the next scheduled run (or another backfill) will resolve it.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--workers", type=int, default=sentiment.DEFAULT_WORKERS,
                    help="concurrent LLM scoring requests (default %(default)s)")
    args = ap.parse_args()
    main(args.config, args.days, args.workers)

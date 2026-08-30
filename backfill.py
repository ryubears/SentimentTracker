"""
Bootstrap N days of history before going live.

Paginates historical X posts and BTC price history, then replays the exact
Phase A / Phase B loop from run_period.py one period at a time — so weights
warm up under the same no-look-ahead rule used in production: the score
saved for period t only ever uses weights snapshotted from periods before t,
and a period's outcome is only folded into the weights once its horizon has
actually elapsed.

Because we have the whole price history up front, "waiting for the horizon to
elapse" just means "the next period boundary in the walk", instead of waiting
on a real clock the way run_period.py does when run on a schedule.

Usage:
  python backfill.py --days 60
  python backfill.py --days 14 --resume # Continue from the current weight state.
"""

from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import argparse
import bisect
import sys
import pandas as pd
import yaml

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, "src")
from sentiment_tracker import db, engine, prices, sentiment
from sentiment_tracker.fetch_x import fetch_historical_posts
from sentiment_tracker.periods import HORIZON, anchor_hour, current_boundary

def period_boundaries(start: datetime, end: datetime, step: timedelta) -> list[pd.Timestamp]:
    ts, out = start, []
    while ts <= end:
        out.append(pd.Timestamp(ts))
        ts += step
    return out

def bucket_by_period(posts: list[dict], periods: list[pd.Timestamp],
                     step: timedelta) -> dict[pd.Timestamp, dict[str, list[tuple[float, float]]]]:
    """Assign each post's (score, engagement) to the period boundary t such that
    t - step < created_at <= t — the same (now - step, now] window run_period.py's
    fetch_posts(since=now-step, until=now) covers."""
    by_period: dict[pd.Timestamp, dict[str, list[tuple[float, float]]]] = \
        defaultdict(lambda: defaultdict(list))
    for p in posts:
        ts = pd.Timestamp(p["created_at"])
        i = bisect.bisect_left(periods, ts)
        if i < len(periods) and periods[i] - step < ts <= periods[i]:
            by_period[periods[i]][p["account"]].append((p["score"], p["engagement"]))
    return by_period

def main(cfg_path: str = "config.yaml", days: int = 60, resume: bool = False) -> None:
    cfg = yaml.safe_load(open(cfg_path))
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
         f"with backend={cfg['sentiment']['backend']}"
         + (" (this calls the LLM once per unscored post — switch to vader in config.yaml to backfill for free)"
            if cfg["sentiment"]["backend"] == "llm" and fresh else "") + "...")
    for p in posts:
        p["score"] = known[p["post_id"]] if p["post_id"] in known else \
            sentiment.score_post(p["text"], {**cfg["sentiment"], "horizon": cfg["horizon"]})
    db.save_posts(con, posts)

    periods = period_boundaries(start, now, step)
    # Bucket from the db, not from `posts`: the cache is the union of every fetch, so a
    # short fetch (a rate limit, an account temporarily returning less) can no longer
    # drop that account's history out of the recomputed periods. Also makes re-runs
    # idempotent — the same window always rebuilds from the same set.
    cached = db.scored_posts_in_range(con, handles, (start - step).isoformat(), now.isoformat())
    print(f"bucketing {len(cached)} cached posts ({len(posts)} returned by this fetch)...")
    by_period = bucket_by_period(cached, periods, step)

    aw = engine.load_weights(con, handles, cfg["weights"], resume=resume)

    resolved_n = 0
    for t in periods:
        # Phase A: resolve anything whose horizon has elapsed by this boundary.
        resolved_n += len(engine.resolve_matured(con, aw, klines, t, step))
        # Phase B: score this period with weights as of before its own outcome.
        engine.score_period(con, aw, t, by_period.get(t, {}),
                            prices.price_at(klines, t), cfg["horizon"])

    print(f"backfilled {len(periods)} periods ({resolved_n} resolved) "
         f"from {start.date()} to {now.date()}")
    print("the most recent period is left unresolved, same as a normal run_period.py call — "
         "the next scheduled run (or another backfill) will resolve it.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--resume", action="store_true",
                    help="continue from the current weight_snapshots state instead of a cold start")
    args = ap.parse_args()
    main(args.config, args.days, args.resume)

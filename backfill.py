"""Bootstrap N days of history before going "live" (see README roadmap).

Paginates historical X posts and BTC price history, then replays the exact
Phase A / Phase B loop from run_period.py one period at a time — so weights
warm up under the *same* no-look-ahead rule used in production: the score
saved for period t only ever uses weights snapshotted from periods before t,
and a period's outcome is only folded into the weights once its horizon has
actually elapsed.

Because we have the whole price history up front, "waiting for the horizon to
elapse" just means "the next period boundary in the walk", instead of waiting
on a real clock the way run_period.py does when run on a schedule.

Usage:
  python backfill.py --days 60
  python backfill.py --days 14 --resume     # continue from the current weight state
"""
from __future__ import annotations

import argparse
import bisect
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import pandas as pd
import yaml

sys.path.insert(0, "src")
from sentiment_tracker import db, prices, sentiment  # noqa: E402
from sentiment_tracker.fetch_x import fetch_historical_posts  # noqa: E402
from sentiment_tracker.weights import AccountWeights, aggregate  # noqa: E402

HORIZON = {"1d": timedelta(days=1), "1h": timedelta(hours=1)}


def period_boundaries(start: datetime, end: datetime, step: timedelta) -> list[pd.Timestamp]:
    ts, out = start, []
    while ts <= end:
        out.append(pd.Timestamp(ts))
        ts += step
    return out


def bucket_by_period(posts: list[dict], periods: list[pd.Timestamp],
                     step: timedelta) -> dict[pd.Timestamp, dict[str, list[float]]]:
    """Assign each post to the period boundary t such that t - step < created_at <= t —
    the same (now - step, now] window run_period.py's fetch_posts(since=now-step) covers."""
    by_period: dict[pd.Timestamp, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for p in posts:
        ts = pd.Timestamp(p["created_at"])
        i = bisect.bisect_left(periods, ts)
        if i < len(periods) and periods[i] - step < ts <= periods[i]:
            by_period[periods[i]][p["account"]].append(p["score"])
    return by_period


def main(cfg_path: str = "config.yaml", days: int = 60, resume: bool = False) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    handles = [a["handle"] for a in cfg["accounts"]]
    step = HORIZON[cfg["horizon"]]
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(days=days)
    con = db.connect(cfg["db_path"])

    print(f"fetching {days}d of {cfg['symbol']} price history...")
    klines = prices.fetch_klines_range(cfg["symbol"], "1h", pd.Timestamp(start) - step, pd.Timestamp(now))

    print(f"fetching historical posts for {len(handles)} accounts since {start.date()}...")
    posts = fetch_historical_posts(handles, since=start, until=now)
    print(f"scoring {len(posts)} posts with backend={cfg['sentiment']['backend']}"
         + (" (this calls the LLM once per post — switch to vader in config.yaml to backfill for free)"
            if cfg["sentiment"]["backend"] == "llm" else "") + "...")
    for p in posts:
        p["score"] = sentiment.score_post(p["text"], {**cfg["sentiment"], "horizon": cfg["horizon"]})
    db.save_posts(con, posts)

    periods = period_boundaries(start, now, step)
    by_period = bucket_by_period(posts, periods, step)

    state = db.latest_state(con) if resume else None
    aw = AccountWeights.from_json(state) if state else AccountWeights(handles, **cfg["weights"])
    for h in handles:
        aw.add_account(h)

    pending = None  # (period_ts, signals, price_now) awaiting resolution
    resolved_n = 0
    for t in periods:
        acct_scores = by_period.get(t, {})
        signals = {a: sum(v) / len(v) for a, v in acct_scores.items()}
        counts = {a: len(v) for a, v in acct_scores.items()}
        price_now = prices.price_at(klines, t)

        # ---- Phase A: the previous period's horizon has now elapsed
        if pending is not None:
            prev_t, prev_signals, prev_price = pending
            ret = price_now / prev_price - 1.0
            aw.update(prev_signals, ret)
            db.resolve_period(con, prev_t.isoformat(), price_now, ret)
            resolved_n += 1

        # ---- Phase B: score this period with weights as of *before* its own outcome
        w = aw.weights()
        agg = aggregate(signals, w)
        agg_uniform = aggregate(signals, {a: 1.0 for a in signals})
        db.save_period(con, t.isoformat(), cfg["horizon"], agg, agg_uniform, price_now,
                       signals, counts, w, aw.to_json())
        pending = (t, signals, price_now)

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

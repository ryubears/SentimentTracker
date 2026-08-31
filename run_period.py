"""
One scheduled run. Two phases:
  A. Resolve past periods whose horizon has elapsed, recording realized returns.
  B. Score the current period from this window's posts.
Run daily (horizon "1d") or hourly (horizon "1h"), change in config.
"""

from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
import sys
import pandas as pd
import yaml

# Load all environment variables from .env file.
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, "src")
from sentiment_tracker import db, engine, prices, sentiment, trader
from sentiment_tracker.fetch_x import fetch_posts
from sentiment_tracker.periods import HORIZON, anchor_hour, current_boundary


def main(config_path: str = "config.yaml") -> None:
    config = yaml.safe_load(open(config_path))
    handles = [a["handle"] for a in config["accounts"]]
    horizon = config["horizon"]
    step = HORIZON[horizon]
    con = db.connect(config["db_path"])
    real_now = datetime.now(timezone.utc)
    now = current_boundary(real_now, step, anchor_hour(con, horizon))

    if real_now - now > timedelta(minutes=30):
        print(f"warning: running {real_now - now} after the {now.isoformat()} boundary; "
              "posts after the boundary belong to the next period and are excluded")
    last = db.latest_period_ts(con, horizon)
    if last is not None:
        missed = int((now - datetime.fromisoformat(last)) / step) - 1
        if missed > 0:
            print(f"warning: {missed} period(s) missing between {last} and {now.isoformat()} — "
                  "they stay empty until filled with backfill.py")

    # Price history spanning everything this run prices: each unresolved period's
    # maturity and the current boundary (minus 1h so a bar at or before each exists).
    maturities = [pd.Timestamp(ts) + step for ts, _ in db.unresolved_periods(con)]
    kstart = min(maturities + [pd.Timestamp(now)]) - timedelta(hours=1)
    klines = prices.get_klines(con, config["symbol"], kstart, pd.Timestamp(now))

    # Phase A: resolve matured periods.
    for period_ts, ret in engine.resolve_matured(con, klines, now, step):
        print(f"resolved {period_ts}: return {ret:+.3%}")

    # Phase B: score the current period from posts in (now - step, now].
    posts = fetch_posts(handles, since=now - step, until=now)
    scored = sentiment.score_many([p["text"] for p in posts],
                                  {**config["sentiment"], "horizon": horizon})
    for p, s in zip(posts, scored):
        p["score"] = s
    posts = [p for p in posts if p["score"] is not None] # Unscored posts retry next run.
    db.save_posts(con, posts)

    by_acct = defaultdict(list)
    for p in posts:
        by_acct[p["account"]].append(p["score"])

    price_now = prices.price_at(klines, pd.Timestamp(now))
    agg, signals = engine.score_period(
        con, now, by_acct, price_now, horizon,
        relevance_min=config.get("signal", {}).get("relevance_min", 0.0))

    decision = trader.act(con, now.isoformat(), agg, config)
    print(f"trade: {decision.action} - {decision.reason}")

    print(json.dumps({"period": now.isoformat(), "posts": len(posts), "score": round(agg, 4),
                      "btc": price_now, "accounts_contributing": len(signals),
                      "most_bullish": sorted(signals.items(), key=lambda kv: -kv[1])[:3],
                      "most_bearish": sorted(signals.items(), key=lambda kv: kv[1])[:3]}, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])

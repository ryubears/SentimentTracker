"""
One scheduled run. Two phases:
  A. Resolve past periods whose horizon has elapsed and update account weights.
  B. Score the current period with the updated weights and snapshot them.
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
from sentiment_tracker import db, engine, prices, sentiment
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
    aw = engine.load_weights(con, handles, config["weights"])

    # Phase A: resolve matured periods, update weights.
    for period_ts, ret in engine.resolve_matured(con, aw, klines, now, step):
        print(f"resolved {period_ts}: return {ret:+.3%}")

    # Phase B: score the current period from posts in (now - step, now].
    posts = fetch_posts(handles, since=now - step, until=now)
    for p in posts:
        p["score"] = sentiment.score_post(p["text"], {**config["sentiment"], "horizon": horizon})
    db.save_posts(con, posts)

    by_acct = defaultdict(list)
    for p in posts:
        by_acct[p["account"]].append((p["score"], p["engagement"]))

    price_now = prices.price_at(klines, pd.Timestamp(now))
    agg, agg_uniform, w = engine.score_period(con, aw, now, by_acct, price_now, horizon)

    print(json.dumps({"period": now.isoformat(), "posts": len(posts), "score": round(agg, 4),
                      "uniform": round(agg_uniform, 4), "btc": price_now,
                      "top_weights": sorted(w.items(), key=lambda kv: -kv[1])[:5]}, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])

"""
One scheduled run. Two phases:
  A. Resolve past periods whose horizon has elapsed and update account weights.
  B. Score the current period with the updated weights and snapshot them.
Run daily (horizon "1d") or hourly (horizon "1h"), change in config.
"""

from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import json
import sys
import pandas as pd
import yaml

# Load all environment variables from .env file.
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, "src")
from sentiment_tracker import db, prices, sentiment
from sentiment_tracker.fetch_x import fetch_posts
from sentiment_tracker.periods import HORIZON, anchor_hour, current_boundary
from sentiment_tracker.weights import AccountWeights, aggregate

def main(config_path: str = "config.yaml") -> None:
    config = yaml.safe_load(open(config_path))
    handles = [a["handle"] for a in config["accounts"]]
    step = HORIZON[config["horizon"]]
    con = db.connect(config["db_path"])
    now = current_boundary(datetime.now(timezone.utc), step, anchor_hour(con, config["horizon"]))
    klines = prices.fetch_klines(config["symbol"], interval="1h")

    state = db.latest_state(con)
    aw = AccountWeights.from_json(state) if state else AccountWeights(handles, **config["weights"])
    for h in handles:
        aw.add_account(h)

    # Phase A: resolve matured periods, update weights.
    for period_ts, price_then in db.unresolved_periods(con):
        t0 = pd.Timestamp(period_ts)
        if now < t0 + step:
            continue
        p1 = prices.price_at(klines, t0 + step)
        ret = p1 / price_then - 1.0
        aw.update(db.signals_for(con, period_ts), ret)
        db.resolve_period(con, period_ts, p1, ret)
        print(f"resolved {period_ts}: return {ret:+.3%}")

    # Phase B: score current period. An off-schedule run can execute well after the
    # boundary, so drop posts newer than it — they belong to the next period.
    posts = [p for p in fetch_posts(handles, since=now - step)
             if pd.Timestamp(p["created_at"]) <= pd.Timestamp(now)]
    for p in posts:
        p["score"] = sentiment.score_post(p["text"], {**config["sentiment"], "horizon": config["horizon"]})
    db.save_posts(con, posts)

    by_acct, counts = defaultdict(list), defaultdict(int)
    for p in posts:
        by_acct[p["account"]].append(p["score"])
        counts[p["account"]] += 1
    signals = {a: sum(v) / len(v) for a, v in by_acct.items()}

    w = aw.weights()
    agg = aggregate(signals, w)
    agg_uniform = aggregate(signals, {a: 1.0 for a in signals})
    price_now = prices.price_at(klines, pd.Timestamp(now))
    db.save_period(con, now.isoformat(), config["horizon"], agg, agg_uniform, price_now,
                   signals, counts, w, aw.to_json())

    print(json.dumps({"period": now.isoformat(), "posts": len(posts), "score": round(agg, 4),
                      "uniform": round(agg_uniform, 4), "btc": price_now,
                      "top_weights": sorted(w.items(), key=lambda kv: -kv[1])[:5]}, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])

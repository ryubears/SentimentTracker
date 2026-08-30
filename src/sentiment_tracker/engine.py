"""
The Phase A / Phase B core shared by run_period.py (live, one period per
scheduled run) and backfill.py (replaying the same loop over history), so the
production loop and the backfill loop cannot drift apart.
"""

from __future__ import annotations
from datetime import datetime, timedelta
from statistics import median
import numpy as np
import pandas as pd

from . import db, prices
from .periods import HORIZON
from .weights import AccountWeights, aggregate

def load_weights(con, handles: list[str], weights_cfg: dict, resume: bool = True) -> AccountWeights:
    """Latest persisted weight state (or a cold start), with any new accounts added."""
    state = db.latest_state(con) if resume else None
    aw = AccountWeights.from_json(state) if state else AccountWeights(handles, **weights_cfg)
    for h in handles:
        aw.add_account(h)
    return aw

def resolve_matured(con, aw: AccountWeights, klines: pd.DataFrame,
                    now: datetime | pd.Timestamp, step: timedelta) -> list[tuple[str, float]]:
    """
    Phase A: resolve stored periods whose horizon has elapsed by `now`, folding
    each realized return into the weights. Periods maturing before the start of
    the price history are left unresolved rather than priced with the wrong bar.
    Returns the (period_ts, realized_return) pairs resolved.
    """
    out = []
    for period_ts, price_then in db.unresolved_periods(con):
        t1 = pd.Timestamp(period_ts) + step
        if pd.Timestamp(now) < t1 or klines.empty or t1 < klines["ts"].iloc[0]:
            continue
        p1 = prices.price_at(klines, t1)
        ret = p1 / price_then - 1.0
        aw.update(db.signals_for(con, period_ts), ret)
        db.resolve_period(con, period_ts, p1, ret)
        out.append((period_ts, ret))
    return out

def engagement_weight(e: float, baseline: float | None) -> float:
    """
    How unusual this post's engagement is *for its own account*, as a
    multiplicative post weight: the ratio to the account's typical (median)
    engagement, sqrt-damped and clipped to [0.5, 3] so a single viral post
    can't drown the rest of the period. Ratios are per-account, so a small
    account's overperforming post counts exactly like a big account's — only
    "better than usual for you" matters, never absolute reach. With no history
    yet (baseline None) every post weighs 1.
    """
    if baseline is None:
        return 1.0
    return float(np.clip(np.sqrt((e + 1.0) / (baseline + 1.0)), 0.5, 3.0))

def score_period(con, aw: AccountWeights, t: datetime | pd.Timestamp,
                 acct_posts: dict[str, list[tuple[float, float]]], price_now: float,
                 horizon: str) -> tuple[float, float, dict[str, float]]:
    """
    Phase B: aggregate per-account post (score, engagement) pairs with the
    current weights and save the period plus a weight snapshot. Each account's
    signal is the engagement-weighted mean of its post scores, with weights
    normalized against that account's own history before this period's window —
    cached posts only, and strictly pre-window so backfill (which saves all
    posts up front) can't look ahead. Returns (agg, agg_uniform, weights).
    """
    cutoff = (t - HORIZON[horizon]).isoformat()
    signals, counts = {}, {}
    for a, posts in acct_posts.items():
        hist = db.engagement_history(con, a, cutoff)
        baseline = median(hist) if hist else None
        pw = [(engagement_weight(e, baseline), s) for s, e in posts]
        signals[a] = sum(w * s for w, s in pw) / sum(w for w, _ in pw)
        counts[a] = len(posts)
    w = aw.weights()
    agg = aggregate(signals, w)
    agg_uniform = aggregate(signals, {a: 1.0 for a in signals})
    db.save_period(con, t.isoformat(), horizon, agg, agg_uniform, price_now,
                   signals, counts, w, aw.to_json())
    return agg, agg_uniform, w

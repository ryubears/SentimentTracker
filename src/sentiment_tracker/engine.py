"""
The Phase A / Phase B core shared by run_period.py (live, one period per
scheduled run) and backfill.py (replaying the same loop over history), so the
production loop and the backfill loop cannot drift apart.
"""

from __future__ import annotations
from datetime import datetime, timedelta
import pandas as pd

from . import db, prices
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

def score_period(con, aw: AccountWeights, t: datetime | pd.Timestamp,
                 acct_scores: dict[str, list[float]], price_now: float,
                 horizon: str) -> tuple[float, float, dict[str, float]]:
    """
    Phase B: aggregate per-account post scores with the current weights and save
    the period plus a weight snapshot. Returns (agg, agg_uniform, weights).
    """
    signals = {a: sum(v) / len(v) for a, v in acct_scores.items()}
    counts = {a: len(v) for a, v in acct_scores.items()}
    w = aw.weights()
    agg = aggregate(signals, w)
    agg_uniform = aggregate(signals, {a: 1.0 for a in signals})
    db.save_period(con, t.isoformat(), horizon, agg, agg_uniform, price_now,
                   signals, counts, w, aw.to_json())
    return agg, agg_uniform, w

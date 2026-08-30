"""
The Phase A / Phase B core shared by run_period.py (live, one period per
scheduled run) and backfill.py (replaying the same loop over history), so the
production loop and the backfill loop cannot drift apart.
"""

from __future__ import annotations
from datetime import datetime, timedelta
import pandas as pd

from . import db, prices

def resolve_matured(con, klines: pd.DataFrame, now: datetime | pd.Timestamp,
                    step: timedelta) -> list[tuple[str, float]]:
    """
    Phase A: resolve stored periods whose horizon has elapsed by `now`, recording
    the realized return over that horizon. Periods maturing before the start of
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
        db.resolve_period(con, period_ts, p1, ret)
        out.append((period_ts, ret))
    return out

def score_period(con, t: datetime | pd.Timestamp, acct_scores: dict[str, list[float]],
                 price_now: float, horizon: str, relevance_min: float = 0.0
                 ) -> tuple[float, dict[str, float]]:
    """
    Phase B: aggregate this period's post scores into one number and save it.

    An account's signal is the mean of its *relevant* posts only — those with
    |score| > relevance_min. The scoring prompt gives an off-topic post score 0,
    and averaging those in treats "said nothing about BTC" as "said neutral",
    which drags every real view toward zero. An account with nothing relevant to
    say contributes no signal at all, exactly like one that did not post.

    The period aggregate is the plain mean across contributing accounts: every
    account gets an equal vote. Returns (aggregate, per-account signals).
    """
    relevant = {a: [s for s in v if abs(s) > relevance_min] for a, v in acct_scores.items()}
    signals = {a: sum(v) / len(v) for a, v in relevant.items() if v}
    counts = {a: len(v) for a, v in relevant.items() if v} # n_posts counts what formed the signal
    agg = sum(signals.values()) / len(signals) if signals else 0.0
    db.save_period(con, t.isoformat(), horizon, agg, price_now, signals, counts)
    return agg, signals

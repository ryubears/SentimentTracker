import sys; sys.path.insert(0, "src")
from datetime import datetime, timedelta, timezone
import pandas as pd
from sentiment_tracker import db, engine

UTC = timezone.utc
STEP = timedelta(days=1)
CFG = {"eta": 2.0, "decay": 0.9, "floor": 0.2}
T0 = datetime(2026, 8, 1, 5, tzinfo=UTC)

def make_klines(start: datetime, hours: int) -> pd.DataFrame:
    """Hourly bars where price rises 1.0 per hour, starting at 100."""
    return pd.DataFrame({"ts": [pd.Timestamp(start) + pd.Timedelta(hours=h) for h in range(hours)],
                         "close": [100.0 + h for h in range(hours)]})

def walk(con, aw, klines, boundaries) -> int:
    resolved = 0
    for t in boundaries:
        resolved += len(engine.resolve_matured(con, aw, klines, t, STEP))
        engine.score_period(con, aw, t, {"bull": [(0.8, 0.0)], "bear": [(-0.8, 0.0)]},
                            float(klines.loc[klines["ts"] <= t, "close"].iloc[-1]), "1d")
    return resolved

def test_walk_resolves_matured_periods_and_rewards_the_right_account():
    con = db.connect(":memory:")
    klines = make_klines(T0, 72)
    aw = engine.load_weights(con, ["bull", "bear"], CFG)
    boundaries = [pd.Timestamp(T0) + i * STEP for i in range(3)]
    assert walk(con, aw, klines, boundaries) == 2 # all but the last period mature.
    rows = con.execute("SELECT resolved, realized_return FROM periods ORDER BY period_ts").fetchall()
    assert [r[0] for r in rows] == [1, 1, 0]
    assert abs(rows[0][1] - 24 / 100) < 1e-9 # +24 price move over the first day.
    w = aw.weights()
    assert w["bull"] > w["bear"] # bullish account was right in a rising market.

def test_rerun_over_the_same_boundaries_replaces_rows_not_duplicates():
    con = db.connect(":memory:")
    klines = make_klines(T0, 72)
    boundaries = [pd.Timestamp(T0) + i * STEP for i in range(3)]
    walk(con, engine.load_weights(con, ["bull", "bear"], CFG), klines, boundaries)
    walk(con, engine.load_weights(con, ["bull", "bear"], CFG, resume=False), klines, boundaries)
    assert con.execute("SELECT COUNT(*) FROM periods").fetchone()[0] == 3

def seed_posts(con, account, engagements):
    """History posts well before T0's scoring window."""
    db.save_posts(con, [{"post_id": f"{account}{i}", "account": account,
                         "created_at": f"2026-07-{10 + i:02d}T00:00:00+00:00", "text": "x",
                         "likes": 0, "score": 0.0, "engagement": e}
                        for i, e in enumerate(engagements)])

def test_engagement_weight_shape():
    assert engine.engagement_weight(50, None) == 1.0 # no history yet: neutral.
    assert engine.engagement_weight(10, 10.0) == 1.0 # at your own baseline: neutral.
    assert engine.engagement_weight(10_000, 10.0) == 3.0 # viral, but capped.
    assert engine.engagement_weight(0, 10_000.0) == 0.5 # flopped, but floored.

def test_engagement_is_normalized_per_account_not_by_absolute_reach():
    con = db.connect(":memory:")
    seed_posts(con, "whale", [1000, 1000, 1000])
    seed_posts(con, "minnow", [10, 10, 10])
    aw = engine.load_weights(con, ["whale", "minnow"], CFG)
    engine.score_period(con, aw, pd.Timestamp(T0),
                        {"whale": [(1.0, 1000), (-1.0, 1000)],
                         "minnow": [(1.0, 10), (-1.0, 10)]}, 100.0, "1d")
    sig = dict(con.execute("SELECT account, signal FROM account_signals"))
    assert sig["whale"] == sig["minnow"] == 0.0 # both at their usual: plain mean.

def test_overperforming_post_tilts_its_accounts_signal():
    con = db.connect(":memory:")
    seed_posts(con, "minnow", [10, 10, 10])
    aw = engine.load_weights(con, ["minnow"], CFG)
    engine.score_period(con, aw, pd.Timestamp(T0),
                        {"minnow": [(1.0, 90), (-1.0, 0)]}, 100.0, "1d")
    sig = con.execute("SELECT signal FROM account_signals").fetchone()[0]
    assert 0.5 < sig < 1.0 # the 9x-usual bullish post dominates the one that flopped.

def test_baseline_excludes_posts_inside_the_scored_window():
    con = db.connect(":memory:")
    db.save_posts(con, [{"post_id": "w1", "account": "a",
                         "created_at": (pd.Timestamp(T0) - pd.Timedelta(hours=2)).isoformat(),
                         "text": "x", "likes": 0, "score": 1.0, "engagement": 10_000}])
    cutoff = (pd.Timestamp(T0) - STEP).isoformat()
    assert db.engagement_history(con, "a", cutoff) == [] # in-window post: no look-ahead.

def test_resolve_leaves_periods_older_than_price_history_unresolved():
    con = db.connect(":memory:")
    con.execute("INSERT INTO periods (period_ts, horizon, price_now, resolved) "
                "VALUES ('2026-01-01T05:00:00+00:00', '1d', 50.0, 0)")
    aw = engine.load_weights(con, ["bull"], CFG)
    klines = make_klines(T0, 72) # starts months after that period matured.
    assert engine.resolve_matured(con, aw, klines, pd.Timestamp(T0) + STEP, STEP) == []
    assert con.execute("SELECT resolved FROM periods").fetchone()[0] == 0

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
        engine.score_period(con, aw, t, {"bull": [0.8], "bear": [-0.8]},
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

def test_signal_is_the_plain_mean_of_an_accounts_post_scores():
    con = db.connect(":memory:")
    aw = engine.load_weights(con, ["a"], CFG)
    engine.score_period(con, aw, pd.Timestamp(T0), {"a": [1.0, -1.0, 0.4]}, 100.0, "1d")
    sig = con.execute("SELECT signal, n_posts FROM account_signals").fetchone()
    assert abs(sig[0] - 0.4 / 3) < 1e-9 and sig[1] == 3 # every post counts equally.

def test_resolve_leaves_periods_older_than_price_history_unresolved():
    con = db.connect(":memory:")
    con.execute("INSERT INTO periods (period_ts, horizon, price_now, resolved) "
                "VALUES ('2026-01-01T05:00:00+00:00', '1d', 50.0, 0)")
    aw = engine.load_weights(con, ["bull"], CFG)
    klines = make_klines(T0, 72) # starts months after that period matured.
    assert engine.resolve_matured(con, aw, klines, pd.Timestamp(T0) + STEP, STEP) == []
    assert con.execute("SELECT resolved FROM periods").fetchone()[0] == 0

def test_irrelevant_posts_do_not_dilute_an_accounts_signal():
    con = db.connect(":memory:")
    aw = engine.load_weights(con, ["a"], CFG)
    # One conviction call among four off-topic posts the prompt scored 0.
    engine.score_period(con, aw, pd.Timestamp(T0), {"a": [0.0, 0.0, -0.8, 0.0, 0.0]},
                        100.0, "1d", relevance_min=0.0)
    sig, n = con.execute("SELECT signal, n_posts FROM account_signals").fetchone()
    assert abs(sig - (-0.8)) < 1e-9 # not -0.16, which the plain mean gave.
    assert n == 1 # n_posts counts what actually formed the signal.

def test_account_with_nothing_relevant_contributes_no_signal():
    con = db.connect(":memory:")
    aw = engine.load_weights(con, ["a", "b"], CFG)
    engine.score_period(con, aw, pd.Timestamp(T0), {"a": [0.0, 0.0], "b": [0.5]},
                        100.0, "1d", relevance_min=0.0)
    rows = [r[0] for r in con.execute("SELECT account FROM account_signals")]
    assert rows == ["b"] # silence on the topic is not a neutral vote.

def test_relevance_min_can_drop_weak_views_too():
    con = db.connect(":memory:")
    aw = engine.load_weights(con, ["a"], CFG)
    engine.score_period(con, aw, pd.Timestamp(T0), {"a": [0.02, 0.9]}, 100.0, "1d",
                        relevance_min=0.05)
    assert abs(con.execute("SELECT signal FROM account_signals").fetchone()[0] - 0.9) < 1e-9

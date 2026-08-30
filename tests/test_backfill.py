import sys; sys.path.insert(0, "src"); sys.path.insert(0, ".")
import pandas as pd
from datetime import timedelta
from sentiment_tracker import db, engine
from backfill import bucket_by_period

STEP = timedelta(days=1)
T0 = pd.Timestamp("2026-08-01T00:00:00+00:00")

def post(pid, account, created, score=0.5):
    return {"post_id": pid, "account": account, "created_at": created, "text": "x",
            "likes": 0, "score": score}

def test_bucketing_uses_cached_posts_a_short_fetch_would_have_missed():
    con = db.connect(":memory:")
    db.save_posts(con, [
        post("old1", "whale", "2026-08-01T09:00:00+00:00"),   # cached from an earlier run
        post("old2", "whale", "2026-08-02T09:00:00+00:00"),
        post("new1", "whale", "2026-08-03T09:00:00+00:00"),   # all today's short fetch returned
        post("other", "minnow", "2026-08-02T10:00:00+00:00"),
        post("dropped", "removed_handle", "2026-08-02T11:00:00+00:00"),  # no longer configured
    ])
    cached = db.scored_posts_in_range(con, ["whale", "minnow"],
                                      (T0 - STEP).isoformat(), (T0 + 3 * STEP).isoformat())
    assert len(cached) == 4 # the un-configured handle's post is excluded.

    periods = [T0 + i * STEP for i in range(4)]
    by_period = bucket_by_period(cached, periods, STEP)
    whale_days = sorted(t for t, accts in by_period.items() if "whale" in accts)
    assert len(whale_days) == 3 # history survives even though only "new1" was re-fetched.

def test_save_period_replaces_the_whole_signal_set():
    con = db.connect(":memory:")
    aw = engine.load_weights(con, ["a", "b"], {"eta": 2.0, "decay": 0.9, "floor": 0.2})
    t = T0
    engine.score_period(con, aw, t, {"a": [0.5], "b": [-0.5]}, 100.0, "1d")
    assert sorted(r[0] for r in con.execute(
        "SELECT account FROM account_signals WHERE period_ts=?", (t.isoformat(),))) == ["a", "b"]

    # Re-scoring the same period without "b" must drop b's stale row.
    engine.score_period(con, aw, t, {"a": [0.5]}, 100.0, "1d")
    assert [r[0] for r in con.execute(
        "SELECT account FROM account_signals WHERE period_ts=?", (t.isoformat(),))] == ["a"]

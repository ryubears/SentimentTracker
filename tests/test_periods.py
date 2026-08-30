import sys; sys.path.insert(0, "src")
from datetime import datetime, timezone
from sentiment_tracker import db
from sentiment_tracker.periods import HORIZON, anchor_hour, current_boundary

UTC = timezone.utc

def test_daily_runs_at_different_hours_land_on_the_same_boundary():
    a = current_boundary(datetime(2026, 8, 30, 5, 43, tzinfo=UTC), HORIZON["1d"])
    b = current_boundary(datetime(2026, 8, 30, 9, 15, tzinfo=UTC), HORIZON["1d"])
    assert a == b == datetime(2026, 8, 30, 0, 0, tzinfo=UTC)

def test_daily_respects_anchor_and_wraps_to_previous_day_before_it():
    step = HORIZON["1d"]
    assert current_boundary(datetime(2026, 8, 30, 9, 15, tzinfo=UTC), step, anchor=5) \
        == datetime(2026, 8, 30, 5, 0, tzinfo=UTC)
    assert current_boundary(datetime(2026, 8, 30, 3, 0, tzinfo=UTC), step, anchor=5) \
        == datetime(2026, 8, 29, 5, 0, tzinfo=UTC)
    assert current_boundary(datetime(2026, 8, 30, 5, 0, tzinfo=UTC), step, anchor=5) \
        == datetime(2026, 8, 30, 5, 0, tzinfo=UTC) # boundary itself counts.

def test_hourly_floors_to_the_hour_regardless_of_anchor():
    assert current_boundary(datetime(2026, 8, 30, 9, 15, 7, tzinfo=UTC), HORIZON["1h"], anchor=5) \
        == datetime(2026, 8, 30, 9, 0, tzinfo=UTC)

def test_anchor_hour_fresh_db_defaults_to_midnight():
    assert anchor_hour(db.connect(":memory:"), "1d") == 0

def test_anchor_hour_inferred_from_earliest_period_then_pinned_in_meta():
    con = db.connect(":memory:")
    con.execute("INSERT INTO periods (period_ts, horizon) VALUES ('2026-07-01T05:00:00+00:00', '1d')")
    con.execute("INSERT INTO periods (period_ts, horizon) VALUES ('2026-07-02T09:00:00+00:00', '1d')")
    assert anchor_hour(con, "1d") == 5 # earliest wins, later strays don't move it.
    assert anchor_hour(con, "1h") == 0 # another horizon's rows don't leak.
    con.execute("DELETE FROM periods")
    assert anchor_hour(con, "1d") == 5 # pinned in meta — pruning rows can't shift it.

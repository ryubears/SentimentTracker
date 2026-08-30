import sys; sys.path.insert(0, "src")
import sqlite3
from datetime import datetime, timedelta, timezone
from sentiment_tracker import db
from sentiment_tracker.fetch_x import engagement_fields

def post(pid, created, engagement=None, engagement_at=None, account="a"):
    return {"post_id": pid, "account": account, "created_at": created, "text": "x",
            "likes": 0, "score": 0.0, "engagement": engagement, "engagement_at": engagement_at}

def test_pre_engagement_db_gains_columns_with_likes_as_proxy(tmp_path):
    path = str(tmp_path / "old.sqlite")
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE posts (post_id TEXT PRIMARY KEY, account TEXT, "
                "created_at TEXT, text TEXT, likes INT, score REAL)")
    con.execute("INSERT INTO posts VALUES ('p1','a','2026-07-01T00:00:00+00:00','x',7,0.5)")
    con.commit(); con.close()

    con = db.connect(path)
    assert con.execute("SELECT engagement, engagement_at FROM posts").fetchone() == (7.0, None)
    db.save_posts(con, [post("p2", "2026-07-02T00:00:00+00:00", engagement=12.0,
                             engagement_at="2026-07-03T01:00:00+00:00")])
    assert db.engagement_history(con, "a", "2026-07-03T00:00:00+00:00") == [7.0, 12.0]

def test_stale_engagement_posts_and_refresh():
    con = db.connect(":memory:")
    db.save_posts(con, [
        # Measured a day+ after posting: settled, not stale.
        post("done", "2026-08-01T00:00:00+00:00", 50.0, "2026-08-02T06:00:00+00:00"),
        # Measured 2h after posting: early snapshot, due for refresh.
        post("early", "2026-08-02T00:00:00+00:00", 3.0, "2026-08-02T02:00:00+00:00"),
        # Never measured (was <1h old at fetch): due for refresh.
        post("young", "2026-08-03T00:00:00+00:00"),
        # Not yet a day old at the cutoff: not eligible.
        post("fresh", "2026-08-10T00:00:00+00:00"),
    ])
    cutoff = "2026-08-09T00:00:00+00:00"
    assert db.stale_engagement_posts(con, cutoff, limit=10) == ["early", "young"]
    assert db.stale_engagement_posts(con, cutoff, limit=1) == ["early"]

    # "early" got a fresh value; "young" is gone from X — keeps None but is stamped.
    db.update_engagement(con, ["early", "young"], {"early": 42.0}, "2026-08-09T01:00:00+00:00")
    rows = dict(con.execute("SELECT post_id, engagement FROM posts"))
    assert rows["early"] == 42.0 and rows["young"] is None
    assert db.stale_engagement_posts(con, cutoff, limit=10) == [] # neither is retried.

def test_engagement_fields_age_gate():
    class T:
        created_at = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
        public_metrics = {"like_count": 4, "retweet_count": 3, "reply_count": 1, "quote_count": 2}
    young = engagement_fields(T(), T.created_at + timedelta(minutes=30))
    assert young == {"engagement": None, "engagement_at": None}
    aged = engagement_fields(T(), T.created_at + timedelta(hours=2))
    assert aged["engagement"] == 4 + 2 * 3 + 1 + 2 and aged["engagement_at"] is not None

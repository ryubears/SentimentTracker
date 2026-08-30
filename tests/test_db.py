import sys; sys.path.insert(0, "src")
import sqlite3
from sentiment_tracker import db

def test_pre_engagement_db_gains_column_with_likes_as_proxy(tmp_path):
    path = str(tmp_path / "old.sqlite")
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE posts (post_id TEXT PRIMARY KEY, account TEXT, "
                "created_at TEXT, text TEXT, likes INT, score REAL)")
    con.execute("INSERT INTO posts VALUES ('p1','a','2026-07-01T00:00:00+00:00','x',7,0.5)")
    con.commit(); con.close()

    con = db.connect(path)
    assert con.execute("SELECT engagement FROM posts").fetchone()[0] == 7.0
    db.save_posts(con, [{"post_id": "p2", "account": "a", "created_at": "2026-07-02T00:00:00+00:00",
                         "text": "y", "likes": 3, "score": 0.1, "engagement": 12.0}])
    assert db.engagement_history(con, "a", "2026-07-03T00:00:00+00:00") == [7.0, 12.0]

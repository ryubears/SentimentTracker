import sys; sys.path.insert(0, "src")
import sqlite3
from sentiment_tracker import db

def post(pid, created, account="a", score=0.0):
    return {"post_id": pid, "account": account, "created_at": created, "text": "x",
            "likes": 0, "score": score}

def test_engagement_columns_are_dropped_from_an_existing_db(tmp_path):
    path = str(tmp_path / "old.sqlite")
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE posts (post_id TEXT PRIMARY KEY, account TEXT, created_at TEXT, "
                "text TEXT, likes INT, score REAL, engagement REAL, engagement_at TEXT)")
    con.execute("INSERT INTO posts VALUES ('p1','a','2026-07-01T00:00:00+00:00','x',7,0.5,42.0,"
                "'2026-07-03T00:00:00+00:00')")
    con.commit(); con.close()

    con = db.connect(path)
    cols = [r[1] for r in con.execute("PRAGMA table_info(posts)")]
    assert "engagement" not in cols and "engagement_at" not in cols
    assert con.execute("SELECT post_id, likes, score FROM posts").fetchone() == ("p1", 7, 0.5)

def test_connect_is_idempotent_on_an_already_migrated_db(tmp_path):
    path = str(tmp_path / "new.sqlite")
    db.connect(path).close()
    con = db.connect(path) # second open must not fail trying to drop absent columns.
    db.save_posts(con, [post("p2", "2026-07-02T00:00:00+00:00")])
    assert db.scored_posts_in_range(con, ["a"], "2026-07-01T00:00:00+00:00",
                                    "2026-07-03T00:00:00+00:00")[0]["account"] == "a"

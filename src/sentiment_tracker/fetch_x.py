"""Pull recent posts for each tracked account via the X API v2 (tweepy)."""
from __future__ import annotations

import os
from datetime import datetime, timezone

import tweepy


def fetch_posts(handles: list[str], since: datetime, max_per_account: int = 20) -> list[dict]:
    client = tweepy.Client(bearer_token=os.environ["X_BEARER_TOKEN"], wait_on_rate_limit=True)
    users = client.get_users(usernames=handles).data or []
    out = []
    for u in users:
        resp = client.get_users_tweets(
            id=u.id, max_results=max_per_account,
            start_time=since.astimezone(timezone.utc),
            exclude=["retweets", "replies"],
            tweet_fields=["created_at", "public_metrics"],
        )
        for t in resp.data or []:
            out.append({
                "post_id": str(t.id), "account": u.username,
                "created_at": t.created_at.isoformat(), "text": t.text,
                "likes": t.public_metrics.get("like_count", 0),
            })
    return out

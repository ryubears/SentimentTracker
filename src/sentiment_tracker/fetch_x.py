"""
Pull recent posts for each tracked account via the X API v2 (tweepy).
"""

from __future__ import annotations
from datetime import datetime, timezone
import os
import tweepy

def fetch_posts(handles: list[str], since: datetime, until: datetime | None = None,
                max_per_account: int = 20) -> list[dict]:
    client = tweepy.Client(bearer_token=os.environ["X_BEARER_TOKEN"], wait_on_rate_limit=True)
    users = client.get_users(usernames=handles).data or []
    out = []
    for u in users:
        resp = client.get_users_tweets(
            id=u.id, max_results=max_per_account,
            start_time=since.astimezone(timezone.utc),
            end_time=until.astimezone(timezone.utc) if until else None,
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

def fetch_historical_posts(handles: list[str], since: datetime, until: datetime | None = None,
                           max_per_account: int = 3200) -> list[dict]:
    """
    Backfill variant: paginates a user's timeline instead of taking one page.
    Note the 3200 tweet cap per account on this endpoint regardless of the time range for a high-volume account.
    """
    client = tweepy.Client(bearer_token=os.environ["X_BEARER_TOKEN"], wait_on_rate_limit=True)
    users = client.get_users(usernames=handles).data or []
    since_utc = since.astimezone(timezone.utc)
    until_utc = (until or datetime.now(timezone.utc)).astimezone(timezone.utc)
    out = []
    for u in users:
        n = 0
        for t in tweepy.Paginator(
            client.get_users_tweets, id=u.id, max_results=100,
            start_time=since_utc, end_time=until_utc,
            exclude=["retweets", "replies"],
            tweet_fields=["created_at", "public_metrics"],
        ).flatten(limit=max_per_account):
            out.append({
                "post_id": str(t.id), "account": u.username,
                "created_at": t.created_at.isoformat(), "text": t.text,
                "likes": t.public_metrics.get("like_count", 0),
            })
            n += 1
        print(f"  {u.username}: {n} posts")
    return out

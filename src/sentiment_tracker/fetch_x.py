"""
Pull recent posts for each tracked account via the X API v2 (tweepy).
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
import os
import tweepy

MIN_ENGAGEMENT_AGE = timedelta(hours=1)

def engagement(pm: dict) -> float:
    """Composite engagement for one post: reshares endorse harder than likes."""
    return (pm.get("like_count", 0) + 2 * pm.get("retweet_count", 0)
            + pm.get("reply_count", 0) + pm.get("quote_count", 0))

def engagement_fields(t, now: datetime) -> dict:
    """
    Engagement snapshot for a freshly fetched post. A post younger than an hour
    hasn't had time to earn engagement, so it gets None (weighted neutrally)
    until run_period.py's refresh pass re-fetches it a day after posting.
    """
    if now - t.created_at < MIN_ENGAGEMENT_AGE:
        return {"engagement": None, "engagement_at": None}
    return {"engagement": engagement(t.public_metrics), "engagement_at": now.isoformat()}

def fetch_engagement(post_ids: list[str]) -> dict[str, float]:
    """Current engagement for existing posts, batched 100 ids per API call.
    Deleted or protected posts are silently absent from the result."""
    client = tweepy.Client(bearer_token=os.environ["X_BEARER_TOKEN"], wait_on_rate_limit=True)
    out = {}
    for i in range(0, len(post_ids), 100):
        resp = client.get_tweets(ids=post_ids[i:i + 100], tweet_fields=["public_metrics"])
        for t in resp.data or []:
            out[str(t.id)] = engagement(t.public_metrics)
    return out

def fetch_posts(handles: list[str], since: datetime, until: datetime | None = None,
                max_per_account: int = 20) -> list[dict]:
    client = tweepy.Client(bearer_token=os.environ["X_BEARER_TOKEN"], wait_on_rate_limit=True)
    now = datetime.now(timezone.utc)
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
                **engagement_fields(t, now),
            })
    return out

def fetch_historical_posts(handles: list[str], since: datetime, until: datetime | None = None,
                           max_per_account: int = 3200) -> list[dict]:
    """
    Backfill variant: paginates a user's timeline instead of taking one page.
    Note the 3200 tweet cap per account on this endpoint regardless of the time range for a high-volume account.
    """
    client = tweepy.Client(bearer_token=os.environ["X_BEARER_TOKEN"], wait_on_rate_limit=True)
    now = datetime.now(timezone.utc)
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
                **engagement_fields(t, now),
            })
            n += 1
        print(f"  {u.username}: {n} posts")
    return out

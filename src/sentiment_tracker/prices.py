"""
BTC price history from Binance public klines (no API key needed), cached in
the db's klines table so each closed hourly bar is only ever fetched once.
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests

from . import db

BINANCE = "https://api.binance.com/api/v3/klines"

def price_at(df: pd.DataFrame, ts: pd.Timestamp) -> float:
    """Close of the last bar at or before ts."""
    return float(df.loc[df["ts"] <= ts, "close"].iloc[-1])

def fetch_klines_range(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp,
                       limit: int = 1000) -> pd.DataFrame:
    """
    Same shape as fetch_klines, but paginates startTime forward to cover an arbitrary
    [start, end] span instead of just the most recent `limit` bars. Used by backfill.py,
    which needs 60+ days of hourly bars which is more than one request's worth.
    """
    span_ms = {"1h": 3_600_000, "1d": 86_400_000}[interval]
    cur = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "qav", "trades", "tbbav", "tbqav", "ignore"]
    frames = []
    while cur < end_ms:
        r = requests.get(BINANCE, params={"symbol": symbol, "interval": interval,
                                          "startTime": cur, "limit": limit}, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        frames.append(pd.DataFrame(batch, columns=cols))
        cur = int(batch[-1][0]) + span_ms
        if len(batch) < limit:
            break
    if not frames:
        return pd.DataFrame(columns=["ts", "close"])
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="open_time")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close"] = df["close"].astype(float)
    return (df[["open_time", "close"]].rename(columns={"open_time": "ts"})
            .sort_values("ts").reset_index(drop=True))

def _spans(hours: list[pd.Timestamp]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Group sorted hourly timestamps into contiguous [first, last] spans."""
    spans: list[list[pd.Timestamp]] = []
    for t in hours:
        if spans and t - spans[-1][1] == pd.Timedelta(hours=1):
            spans[-1][1] = t
        else:
            spans.append([t, t])
    return [(a, b) for a, b in spans]

def get_klines(con, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Hourly closes covering [start, end], served from the db cache; only hours
    missing from it are fetched from Binance. Only closed bars are cached or
    returned, so a stored close never changes after the fact.
    """
    last_closed = pd.Timestamp(datetime.now(timezone.utc) - timedelta(hours=1)).floor("h")
    start = pd.Timestamp(start).floor("h")
    end = min(pd.Timestamp(end).floor("h"), last_closed)
    if end < start:
        return pd.DataFrame(columns=["ts", "close"])

    have = {pd.Timestamp(ts) for ts, _ in
            db.cached_klines(con, symbol, start.isoformat(), end.isoformat())}
    missing = [t for t in pd.date_range(start, end, freq="h") if t not in have]
    for a, b in _spans(missing):
        fetched = fetch_klines_range(symbol, "1h", a, b + pd.Timedelta(hours=1))
        fetched = fetched[(fetched["ts"] >= a) & (fetched["ts"] <= b)]
        db.save_klines(con, symbol,
                       [(t.isoformat(), c) for t, c in zip(fetched["ts"], fetched["close"])])

    rows = db.cached_klines(con, symbol, start.isoformat(), end.isoformat())
    return pd.DataFrame(rows, columns=["ts", "close"]).assign(ts=lambda d: pd.to_datetime(d["ts"]))

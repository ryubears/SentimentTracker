"""BTC price history from Binance public klines (no API key needed)."""
from __future__ import annotations

import pandas as pd
import requests

BINANCE = "https://api.binance.com/api/v3/klines"


def fetch_klines(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 1000) -> pd.DataFrame:
    r = requests.get(BINANCE, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=30)
    r.raise_for_status()
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "qav", "trades", "tbbav", "tbqav", "ignore"]
    df = pd.DataFrame(r.json(), columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close"] = df["close"].astype(float)
    return df[["open_time", "close"]].rename(columns={"open_time": "ts"})


def price_at(df: pd.DataFrame, ts: pd.Timestamp) -> float:
    """Close of the last bar at or before ts."""
    return float(df.loc[df["ts"] <= ts, "close"].iloc[-1])


def fetch_klines_range(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp,
                       limit: int = 1000) -> pd.DataFrame:
    """Same shape as fetch_klines, but paginates startTime forward to cover an arbitrary
    [start, end] span instead of just the most recent `limit` bars. Used by backfill.py,
    which needs 60+ days of hourly bars — more than one request's worth."""
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

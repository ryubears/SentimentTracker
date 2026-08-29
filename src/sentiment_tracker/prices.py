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

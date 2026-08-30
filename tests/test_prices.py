import sys; sys.path.insert(0, "src")
import pandas as pd
from sentiment_tracker import db, prices

T0 = pd.Timestamp("2026-08-01T00:00:00+00:00")
H = pd.Timedelta(hours=1)

def install_fake_binance(monkeypatch, calls):
    """Serve synthetic hourly bars (close = 100 + hours since T0) and log each request span."""
    def fake_range(symbol, interval, start, end, limit=1000):
        calls.append((start, end))
        hours = pd.date_range(start.ceil("h"), end.floor("h"), freq="h")
        return pd.DataFrame({"ts": hours, "close": [100.0 + (t - T0) / H for t in hours]})
    monkeypatch.setattr(prices, "fetch_klines_range", fake_range)

def test_cache_hit_skips_binance_and_misses_fetch_only_the_gaps(monkeypatch):
    con, calls = db.connect(":memory:"), []
    install_fake_binance(monkeypatch, calls)

    df = prices.get_klines(con, "BTCUSDT", T0, T0 + 48 * H)
    assert len(df) == 49 and len(calls) == 1

    calls.clear()
    df = prices.get_klines(con, "BTCUSDT", T0, T0 + 48 * H)
    assert len(df) == 49 and calls == [] # fully cached: no network.
    assert prices.price_at(df, T0 + 30 * H) == 130.0

    # Extending the range backwards fetches only the missing prefix.
    df = prices.get_klines(con, "BTCUSDT", T0 - 24 * H, T0 + 48 * H)
    assert len(df) == 73 and calls == [(T0 - 24 * H, T0)]

def test_interior_gaps_are_fetched_as_separate_spans(monkeypatch):
    con, calls = db.connect(":memory:"), []
    install_fake_binance(monkeypatch, calls)
    prices.get_klines(con, "BTCUSDT", T0, T0 + 48 * H)
    con.execute("DELETE FROM klines WHERE ts IN (?,?)",
                ((T0 + 10 * H).isoformat(), (T0 + 20 * H).isoformat()))
    calls.clear()
    df = prices.get_klines(con, "BTCUSDT", T0, T0 + 48 * H)
    assert len(df) == 49 and len(calls) == 2 # two one-hour holes, two fetches.

def test_open_bar_is_never_cached_or_returned(monkeypatch):
    con, calls = db.connect(":memory:"), []
    install_fake_binance(monkeypatch, calls)
    real_now = pd.Timestamp.now(tz="UTC")
    df = prices.get_klines(con, "BTCUSDT", real_now - 5 * H, real_now)
    assert df["ts"].max() <= real_now - H # the still-open bar's close would drift.
    assert con.execute("SELECT MAX(ts) FROM klines").fetchone()[0] <= (real_now - H).isoformat()

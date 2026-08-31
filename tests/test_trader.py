import sys; sys.path.insert(0, "src")
from sentiment_tracker import db, trader

CFG = {"buy_threshold": 0.05, "sell_threshold": -0.02, "order_size_usd": 25.0,
       "max_position_usd": 100.0, "min_base_size": 1e-6}

def test_flat_aggregate_never_trades():
    # 0.0 means nobody said anything relevant - absence of evidence, not bearish.
    assert trader.decide(0.0, 0.0, 0.0, CFG).action == trader.HOLD
    assert trader.decide(0.0, 0.5, 40_000.0, CFG).action == trader.HOLD

def test_buys_a_slice_when_bullish_enough():
    d = trader.decide(0.09, 0.0, 0.0, CFG)
    assert d.action == trader.BUY and d.quote_size == 25.0

def test_buy_is_capped_by_max_position():
    assert trader.decide(0.09, 0.002, 100.0, CFG).action == trader.HOLD  # already at cap
    d = trader.decide(0.09, 0.001, 90.0, CFG)
    assert d.action == trader.BUY and d.quote_size == 10.0 # only the remaining room

def test_sells_the_whole_position_when_bearish():
    d = trader.decide(-0.05, 0.004, 200.0, CFG)
    assert d.action == trader.SELL and d.base_size == 0.004

def test_bearish_without_a_position_is_a_hold():
    assert trader.decide(-0.05, 0.0, 0.0, CFG).action == trader.HOLD

def test_between_thresholds_is_a_hold():
    assert trader.decide(0.01, 0.0, 0.0, CFG).action == trader.HOLD


class FakeBroker:
    def __init__(self, base=0.0, price=50_000.0):
        self.base, self.price, self.calls = base, price, []
    def position(self): return self.base, self.base * self.price
    def buy(self, quote_size, order_id):
        self.calls.append(("buy", quote_size, order_id)); return {"order_id": "ok-buy"}
    def sell(self, base_size, order_id):
        self.calls.append(("sell", base_size, order_id)); return {"order_id": "ok-sell"}

def cfg(**over):
    t = {"enabled": True, "dry_run": True, **CFG}; t.update(over)
    return {"trading": t}

def test_dry_run_records_the_decision_but_sends_no_order():
    con, b = db.connect(":memory:"), FakeBroker()
    d = trader.act(con, "2026-08-30T00:00:00+00:00", 0.09, cfg(dry_run=True), broker=b)
    assert d.action == trader.BUY and b.calls == [] # nothing sent
    row = con.execute("SELECT action, status, quote_size FROM trades").fetchone()
    assert row == ("buy", "dry-run", 25.0) # but the audit trail is identical

def test_disabled_never_sends_even_if_dry_run_is_false():
    con, b = db.connect(":memory:"), FakeBroker()
    trader.act(con, "2026-08-30T01:00:00+00:00", 0.09, cfg(enabled=False, dry_run=False), broker=b)
    assert b.calls == []
    assert con.execute("SELECT status FROM trades").fetchone()[0] == "dry-run"

def test_live_sends_the_order_with_an_idempotent_id():
    con, b = db.connect(":memory:"), FakeBroker()
    trader.act(con, "2026-08-30T02:00:00+00:00", 0.09, cfg(dry_run=False), broker=b)
    assert b.calls == [("buy", 25.0, "st-2026-08-30T02:00:00+00:00-buy")]
    assert con.execute("SELECT status FROM trades").fetchone()[0] == "placed"

def test_a_failing_order_is_recorded_and_does_not_raise():
    class Boom(FakeBroker):
        def buy(self, *a): raise RuntimeError("exchange down")
    con = db.connect(":memory:")
    d = trader.act(con, "2026-08-30T03:00:00+00:00", 0.09, cfg(dry_run=False), broker=Boom())
    assert d.action == trader.BUY
    status, detail = con.execute("SELECT status, detail FROM trades").fetchone()
    assert status == "error" and "exchange down" in detail

def test_holds_are_recorded_too():
    con = db.connect(":memory:")
    trader.act(con, "2026-08-30T04:00:00+00:00", 0.0, cfg(), broker=FakeBroker())
    assert con.execute("SELECT action, status FROM trades").fetchone() == ("hold", "no-op")

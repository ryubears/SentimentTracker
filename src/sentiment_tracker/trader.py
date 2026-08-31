"""
Act on the aggregate signal through Coinbase Advanced Trade.

Long-only and deliberately dull: buy a fixed dollar amount when the aggregate
turns bullish enough, sell the whole position when it turns bearish. Anything
in between is a hold.

Safety properties, in order of how much they matter:

  * Dry run is the default. Placing a real order needs trading.enabled AND
    trading.dry_run: false in config. A missing trading block means disabled.
  * Credentials are read only from the environment (COINBASE_API_KEY /
    COINBASE_API_SECRET) and never from config, argv, or the db.
  * Orders are idempotent per period: client_order_id is derived from the
    period timestamp, so re-running a period cannot buy twice.
  * Every decision is recorded in the trades table whether or not an order was
    placed, so a dry run produces the same audit trail a live run would.
  * Buys are capped by max_position_usd; a buy that would breach it is refused.

decide() is a pure function and carries the whole policy, so the behaviour can
be tested without credentials or network.
"""

from __future__ import annotations
from dataclasses import dataclass
import os

BUY, SELL, HOLD = "buy", "sell", "hold"

@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    quote_size: float = 0.0   # USD to spend on a buy
    base_size: float = 0.0    # BTC to sell

def decide(agg: float, position_base: float, position_value_usd: float, cfg: dict) -> Decision:
    """
    Policy, given the period's aggregate score and the current BTC position.

    Bullish above buy_threshold and not already at the cap -> buy a fixed slice.
    Bearish below sell_threshold while holding -> sell everything.
    A flat aggregate (exactly 0, i.e. nobody said anything relevant) is never a
    trade: it is absence of evidence, not a bearish view.
    """
    buy_at = float(cfg.get("buy_threshold", 0.05))
    sell_at = float(cfg.get("sell_threshold", -0.02))
    slice_usd = float(cfg.get("order_size_usd", 25.0))
    cap_usd = float(cfg.get("max_position_usd", 100.0))
    dust = float(cfg.get("min_base_size", 1e-6))

    if agg == 0.0:
        return Decision(HOLD, "aggregate is flat - no directional call this period")
    if agg <= sell_at:
        if position_base > dust:
            return Decision(SELL, f"aggregate {agg:+.4f} <= sell_threshold {sell_at:+.4f}",
                            base_size=position_base)
        return Decision(HOLD, f"bearish ({agg:+.4f}) but no position to sell")
    if agg >= buy_at:
        room = cap_usd - position_value_usd
        if room <= 0:
            return Decision(HOLD, f"bullish ({agg:+.4f}) but position is at the "
                                  f"{cap_usd:.2f} USD cap")
        size = min(slice_usd, room)
        return Decision(BUY, f"aggregate {agg:+.4f} >= buy_threshold {buy_at:+.4f}",
                        quote_size=round(size, 2))
    return Decision(HOLD, f"aggregate {agg:+.4f} is between "
                          f"{sell_at:+.4f} and {buy_at:+.4f}")


class CoinbaseBroker:
    """Thin wrapper over coinbase-advanced-py. Constructed only when trading is
    live; dry runs never touch the network or the credentials."""

    def __init__(self, product_id: str = "BTC-USD"):
        from coinbase.rest import RESTClient # imported late: optional dependency
        key, secret = os.environ.get("COINBASE_API_KEY"), os.environ.get("COINBASE_API_SECRET")
        if not key or not secret:
            raise RuntimeError("COINBASE_API_KEY / COINBASE_API_SECRET are not set; "
                               "add them to .env or run with dry_run")
        self.client = RESTClient(api_key=key, api_secret=secret)
        self.product_id = product_id
        self.base_ccy = product_id.split("-")[0]

    def position(self) -> tuple[float, float]:
        """(base held, its value in quote currency) from the exchange itself, so
        the position is never inferred from our own record of what we sent."""
        held = 0.0
        for acct in (self.client.get_accounts().to_dict().get("accounts") or []):
            if acct.get("currency") == self.base_ccy:
                held += float((acct.get("available_balance") or {}).get("value", 0.0))
        price = float(self.client.get_product(self.product_id).to_dict()["price"])
        return held, held * price

    def buy(self, quote_size: float, order_id: str) -> dict:
        return self.client.market_order_buy(client_order_id=order_id,
                                            product_id=self.product_id,
                                            quote_size=f"{quote_size:.2f}").to_dict()

    def sell(self, base_size: float, order_id: str) -> dict:
        return self.client.market_order_sell(client_order_id=order_id,
                                             product_id=self.product_id,
                                             base_size=f"{base_size:.8f}").to_dict()


def act(con, period_ts: str, agg: float, cfg: dict, broker=None) -> Decision:
    """
    Apply the policy for one period and record what happened.

    Returns the Decision. Placing an order requires enabled and not dry_run; in
    every other case the decision is still written to the trades table, so a dry
    run leaves exactly the audit trail a live run would.
    """
    from . import db
    tcfg = cfg.get("trading") or {}
    enabled = bool(tcfg.get("enabled", False))
    dry_run = bool(tcfg.get("dry_run", True))
    live = enabled and not dry_run

    if live and broker is None:
        broker = CoinbaseBroker(tcfg.get("product_id", "BTC-USD"))
    position_base, position_usd = broker.position() if broker else db.simulated_position(con)

    d = decide(agg, position_base, position_usd, tcfg)
    order_id = f"st-{period_ts}-{d.action}" # idempotent: one order per period per side
    status, detail = ("dry-run" if not live else "placed"), ""

    if d.action != HOLD and live:
        try:
            resp = (broker.buy(d.quote_size, order_id) if d.action == BUY
                    else broker.sell(d.base_size, order_id))
            detail = str(resp.get("order_id") or resp.get("success_response", {}).get("order_id", ""))
        except Exception as e: # a failed order must not take the scoring run down
            status, detail = "error", f"{type(e).__name__}: {e}"
    elif d.action == HOLD:
        status = "no-op"

    db.save_trade(con, period_ts, d.action, agg, d.quote_size, d.base_size,
                  status, d.reason, detail)
    return d

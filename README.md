# BTC Sentiment Tracker

Scores a curated list of crypto X accounts every period, combines them into one
sentiment number, and measures — honestly — whether that number predicts the next
BTC move.

**It does not.** Across daily and hourly horizons, 28/50/66-account rosters, with
and without engagement weighting, the correlation has never escaped its own noise
band. The current hourly series gives Pearson **0.018** against a shuffled-null
95th percentile of **0.058**. The value here is the measurement apparatus, not an
edge. See [`RESULTS.md`](RESULTS.md).

## How it works

```
X posts ─► per-post sentiment (LLM or VADER) ─► per-account signal
                                                (mean of relevant posts)
                                                        │
                                        aggregate = mean across accounts
                                                        │
BTC price (Binance) ─► realized return over the horizon ─┴─► evaluation
                                                        └─► optional Coinbase order
```

**Relevance filter.** The scoring prompt gives an off-topic post `score 0`, and
~60% of posts land there. Averaging those in would treat "said nothing about BTC"
as "said neutral", dragging real views toward zero — so only posts with
`|score| > signal.relevance_min` count. An account with nothing relevant to say
contributes nothing, exactly like one that did not post.

**Every account gets one vote.** An earlier version weighted accounts by recent
accuracy (hedge / multiplicative weights). It was removed: with a uniform floor
spread across 50 accounts the weights only ever spanned 2.08%–2.30%, so the
adaptive and uniform aggregates agreed to three decimals. It was inert *as
configured* — a zero floor with sharper `eta` was never tested — not disproven.

**No look-ahead.** A period's outcome is only ever read after its horizon has
elapsed, and `backfill.py` replays the same Phase A / Phase B loop as
`run_period.py` one boundary at a time, so backfilled history is built exactly
the way live history is.

**Honest evaluation.** Every run reports a shuffled-label null alongside the real
correlation, so a number that looks like signal can be checked against what noise
produces on the same data. Hit rate counts only periods with a directional call —
a flat aggregate is no call, not a wrong one.

## Run locally

```bash
/usr/bin/python3 -m pip install -r requirements.txt
cp .env.example .env            # add X_BEARER_TOKEN, ANTHROPIC_API_KEY
/usr/bin/python3 -m pytest tests
/usr/bin/python3 run_period.py  # one period
/usr/bin/python3 report.py      # refresh RESULTS.md
/usr/bin/python3 dashboard.py   # writes dashboard.html
```

Use an explicit interpreter path if more than one Python is installed — packages
installed for one are invisible to the other, and a scheduled job that hardcodes
`/usr/bin/python3` will not see them otherwise.

## Backfill

```bash
/usr/bin/python3 backfill.py --days 60 --workers 24
```

Paginates historical X posts and BTC prices, then walks the period boundaries.
Scoring is concurrent (~8 posts/sec at 24 workers, versus 0.42 serially).

Two properties worth knowing:

- **Scores are cached per post**, so re-running a window costs no LLM calls. But
  the scoring prompt embeds the horizon, so switching `1d` ↔ `1h` means re-scoring
  every post — cached scores answer a different question.
- **Periods are bucketed from the database**, not from one fetch's results, so a
  short response from the X API cannot silently drop an account's history.

## Configuration

`config.yaml`:

| Section | What it controls |
|---|---|
| `accounts` | the handles to track (currently 50) |
| `horizon` | `"1d"` or `"1h"` — how far ahead the score should predict |
| `signal.relevance_min` | posts at or below this magnitude are "no view" |
| `signal.deadband` | only act when the aggregate clears this |
| `sentiment` | `llm` (Claude) or `vader` |
| `trading` | Coinbase execution — **`enabled: false` by default** |
| `db_path` | SQLite file |

Switching horizons also means pointing `db_path` somewhere new: the `periods`
table keys on the timestamp alone, so two horizons in one database collide at
shared boundaries.

## Trading

`trader.py` can place Coinbase Advanced Trade orders: buy a fixed dollar slice
when the aggregate is bullish past `buy_threshold`, sell the position when it
turns bearish. **Dry run is the default** — a live order needs
`trading.enabled: true` *and* `trading.dry_run: false`, plus `COINBASE_API_KEY` /
`COINBASE_API_SECRET` in the environment.

Orders are idempotent per period, buys are capped by `max_position_usd`, a flat
aggregate never trades, and every decision — including holds and dry runs — is
written to the `trades` table. Given the signal has no measured predictive power,
treat this as plumbing, not a strategy.

## Deploying

[`DEPLOY.md`](DEPLOY.md) covers running it on AWS: an hourly Lambda with the
SQLite database synced to S3.

## Layout

```
run_period.py      one period: resolve matured, score current, maybe trade
backfill.py        replay N days of history through the same loop
report.py          metrics -> RESULTS.md
dashboard.py       self-contained HTML dashboard
lambda_handler.py  AWS entry point (pull db -> run -> push db)
src/sentiment_tracker/
  engine.py     Phase A/B core shared by live and backfill
  sentiment.py  per-post scoring, concurrent
  fetch_x.py    X API
  prices.py     Binance klines, cached in the db
  periods.py    period boundaries and anchoring
  evaluate.py   correlations, Sortino, shuffled null
  trader.py     Coinbase execution policy
  db.py         SQLite persistence and migrations
  runtime.py    config / secrets / paths (local vs Lambda)
  storage.py    S3 sync for the database
```

## What's been tried

- [x] 60-day backfill so evaluation has history
- [x] Hourly horizon — 1,440 periods tightened the 95% CI on Pearson from ±0.254 to ±0.052
- [x] Engagement-weighted signals — removed; a post fetched minutes after publication
      has not earned engagement yet, and it changed nothing measurable
- [x] Relevance filter — doubled aggregate dispersion (std 0.028 → 0.058)
- [x] Adaptive per-account weights — removed as inert
- [x] Coinbase execution, dry-run by default
- [x] AWS Lambda deployment
- [ ] Forward validation: the deadband was tuned in-sample, so only periods
      collected *after* it was chosen can confirm or kill it

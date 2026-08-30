# BTC Sentiment Tracker

Scores a curated list of crypto X accounts every period, combines them into one
sentiment number using **adaptive per-account weights**, and tracks whether that
number predicts the next BTC move. Weights update automatically from realized returns.

Live results: see [`RESULTS.md`](RESULTS.md) (auto-updated by GitHub Actions).

## How it works

```
X posts ─► per-post sentiment (LLM or VADER) ─► per-account signal (mean)
                                                       │
BTC price (Binance) ─► realized return ─► weight update ◄┘
                                                       │
                                        weighted aggregate score ─► evaluation
```

**Weight update** — Hedge / multiplicative-weights with recency decay and a uniform floor:

- score: `S_a ← λ·S_a + (1−λ)·(signal_a · sign(return))`
- weight: `w_a = (1−ε)·softmax(η·S)_a + ε/N`

Recent accuracy dominates (λ), confident wrong calls are penalised more than hedged
ones, and the floor ε keeps every account alive so it can recover after a bad streak.

**No look-ahead.** Weights are snapshotted each period and the aggregate uses only
weights known at scoring time. Evaluation compares against a uniform-weight baseline
and a shuffled-label null.

## Run locally
```bash
pip install -r requirements.txt
cp .env.example .env            # add X_BEARER_TOKEN, ANTHROPIC_API_KEY
python -m pytest tests
python run_period.py            # one period; schedule with cron or the included workflow
python report.py
```

## Backfill & tune
```bash
python backfill.py --days 60          # bootstrap history before going "live" (--resume to extend it)
python sweep.py                       # grid sweep over eta/decay/floor, walk-forward validated
```
`backfill.py` paginates historical X posts and BTC price history, then replays the same
Phase A / Phase B loop as `run_period.py`, one period boundary at a time — so weights warm
up under the identical no-look-ahead rule used live, just without waiting on a real clock.

`sweep.py` replays already-recorded (signal, realized_return) pairs — no API calls — under
each candidate (η, λ, ε), then scores each combo on forward-chained folds after an initial
burn-in stretch, ranking by worst-fold correlation so a combo has to hold up across time,
not just fit one lucky window.

## Roadmap
- [x] Bootstrap 60+ days of history so weights are meaningful before going "live" (`backfill.py`)
- [ ] Hourly horizon experiment (set `horizon: "1h"`, `decay: 0.97`)
- [x] Hyperparameter sweep for η, λ, ε with walk-forward validation (`sweep.py`)
- [ ] Engagement-weighted signals (likes/reposts) vs. plain mean
- [ ] Dashboard of weight trajectories per account

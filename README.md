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

## Roadmap
- [ ] Bootstrap 60+ days of history so weights are meaningful before going "live"
- [ ] Hourly horizon experiment (set `horizon: "1h"`, `decay: 0.97`)
- [ ] Hyperparameter sweep for η, λ, ε with walk-forward validation
- [ ] Engagement-weighted signals (likes/reposts) vs. plain mean
- [ ] Dashboard of weight trajectories per account

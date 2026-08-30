"""
Grid sweep over (eta, decay, floor) with walk-forward validation.

The weight-update rule is already online and look-ahead-free by construction:
the score saved for period t only ever depends on weights derived from periods
before t (see weights.py / run_period.py). That means there's no separate
train step to walk forward through per fold for a given hyperparameter
combo we just replay the entire historical sequence once, in period order,
which is cheap (pure numpy over signals/returns already recorded in the db,
no API calls, no LLM). Walk-forward here is purely about evaluation: slice
the resulting (agg_score, realized_return) series into forward-chained folds,
skip an initial burn-in stretch where a cold-started weight vector is still
close to uniform, and require a combo to hold up across folds before trusting 
it. That guards against picking eta/decay/floor that happen to fit one lucky 
stretch of the timeline.

Needs history to sweep over: run `python backfill.py --days 60` first (or let
run_period.py accumulate periods over time).

Usage:
  python sweep.py
  python sweep.py --eta 1,2,4,8 --decay 0.8,0.9,0.95 --floor 0.1,0.2,0.3 --folds 6
"""
from __future__ import annotations
import argparse
import itertools
import sys
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "src")
from sentiment_tracker import db  # noqa: E402
from sentiment_tracker.evaluate import metrics  # noqa: E402
from sentiment_tracker.weights import AccountWeights, aggregate  # noqa: E402

def load_history(con) -> tuple[pd.DataFrame, pd.DataFrame]:
    periods = pd.read_sql(
        "SELECT period_ts, realized_return FROM periods WHERE resolved=1 ORDER BY period_ts", con)
    signals = pd.read_sql("SELECT period_ts, account, signal FROM account_signals", con)
    return periods, signals

def replay(periods: pd.DataFrame, by_period: dict[str, dict[str, float]], accounts: list[str],
          eta: float, decay: float, floor: float) -> np.ndarray:
    """
    Recompute the adaptive aggregate score for every resolved period under one
    hyperparameter combo. reward = signal * sign(return) doesn't depend on
    (eta, decay, floor) at all, so this needs only the signals/returns already
    on disk and no re-fetching required to sweep hundreds of combos.
    """
    aw = AccountWeights(accounts, eta=eta, decay=decay, floor=floor)
    scores = np.empty(len(periods))
    for i, row in enumerate(periods.itertuples()):
        sig = by_period.get(row.period_ts, {})
        w = aw.weights()                      # snapshot known *before* this period's outcome
        scores[i] = aggregate(sig, w)
        aw.update(sig, row.realized_return)   # now fold in the outcome
    return scores

def walk_forward(periods: pd.DataFrame, by_period: dict, accounts: list[str],
                 eta: float, decay: float, floor: float,
                 n_folds: int, burn_in_frac: float) -> dict:
    scores = replay(periods, by_period, accounts, eta, decay, floor)
    ret = periods["realized_return"].values
    n = len(periods)
    burn_in = int(n * burn_in_frac)
    idx = np.arange(burn_in, n)
    fold_metrics = [metrics(scores[fold], ret[fold])
                    for fold in np.array_split(idx, n_folds) if len(fold) >= 2]
    row = {"eta": eta, "decay": decay, "floor": floor, "n_folds_used": len(fold_metrics)}
    if not fold_metrics:
        return row
    pearsons = [m["pearson"] for m in fold_metrics]
    row.update({
        "mean_pearson": float(np.nanmean(pearsons)),
        "worst_fold_pearson": float(np.nanmin(pearsons)),
        "mean_hit_rate": float(np.mean([m["hit_rate"] for m in fold_metrics])),
        "pooled_pearson": metrics(scores[idx], ret[idx])["pearson"],
    })
    return row

def main(cfg_path: str = "config.yaml", eta_grid=(1.0, 2.0, 4.0, 8.0), decay_grid=(0.8, 0.9, 0.95),
        floor_grid=(0.1, 0.2, 0.3), n_folds: int = 5, burn_in_frac: float = 0.2) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    con = db.connect(cfg["db_path"])
    periods, signals = load_history(con)
    if len(periods) < n_folds * 5:
        print(f"only {len(periods)} resolved periods — need more history for a meaningful "
             f"walk-forward sweep (try `python backfill.py --days 60`, or fewer --folds)")
        return

    accounts = sorted(set(signals["account"]))
    by_period = {t: dict(zip(g["account"], g["signal"])) for t, g in signals.groupby("period_ts")}

    combos = list(itertools.product(eta_grid, decay_grid, floor_grid))
    rows = [walk_forward(periods, by_period, accounts, eta, decay, floor, n_folds, burn_in_frac)
           for eta, decay, floor in combos]
    df = pd.DataFrame(rows)
    ranked = df[df["n_folds_used"] > 0].sort_values(
        ["worst_fold_pearson", "mean_pearson"], ascending=False)
    df.to_csv("sweep_results.csv", index=False)

    cur = cfg["weights"]
    baseline = walk_forward(periods, by_period, accounts, cur["eta"], cur["decay"], cur["floor"],
                            n_folds, burn_in_frac)
    print(f"{len(periods)} resolved periods, {n_folds} walk-forward folds "
         f"({burn_in_frac:.0%} burn-in excluded), {len(combos)} combos swept\n")
    if baseline.get("n_folds_used"):
        print(f"current config.yaml (eta={cur['eta']}, decay={cur['decay']}, floor={cur['floor']}): "
             f"mean_pearson={baseline['mean_pearson']:.3f}  "
             f"worst_fold={baseline['worst_fold_pearson']:.3f}  "
             f"hit_rate={baseline['mean_hit_rate']:.3f}\n")

    print(f"top 10 (ranked by worst-fold pearson, tie-broken by mean — a combo that only "
         f"wins on one fold is ranked below one that's merely good everywhere):")
    cols = ["eta", "decay", "floor", "mean_pearson", "worst_fold_pearson", "mean_hit_rate"]
    print(ranked[cols].head(10).to_string(index=False))
    print("\nfull grid written to sweep_results.csv")

if __name__ == "__main__":
    parse_list = lambda s: tuple(float(x) for x in s.split(","))  # noqa: E731
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--eta", type=parse_list, default=(1.0, 2.0, 4.0, 8.0))
    ap.add_argument("--decay", type=parse_list, default=(0.8, 0.9, 0.95))
    ap.add_argument("--floor", type=parse_list, default=(0.1, 0.2, 0.3))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--burn-in", type=float, default=0.2)
    args = ap.parse_args()
    main(args.config, args.eta, args.decay, args.floor, args.folds, args.burn_in)

"""Does the score predict BTC? Compares adaptive vs uniform weights on resolved periods only.
Both aggregates were computed with weights known at the time, so there is no look-ahead."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def metrics(x: np.ndarray, r: np.ndarray) -> dict:
    """Score-vs-realized-return metrics for one (score, return) series. Pulled out of
    evaluate() so sweep.py's walk-forward folds score themselves the same way."""
    x, r = np.asarray(x, dtype=float), np.asarray(r, dtype=float)
    pearson = float(stats.pearsonr(x, r)[0]) if len(x) > 1 and x.std() > 0 else float("nan")
    spearman = float(stats.spearmanr(x, r)[0]) if len(x) > 1 and x.std() > 0 else float("nan")
    return {
        "pearson": pearson,
        "spearman": spearman,
        "hit_rate": float(np.mean(np.sign(x) == np.sign(r))),
        "mean_ret_when_bullish": float(r[x > 0].mean()) if (x > 0).any() else None,
        "mean_ret_when_bearish": float(r[x < 0].mean()) if (x < 0).any() else None,
    }


def evaluate(con) -> dict:
    df = pd.read_sql("SELECT * FROM periods WHERE resolved=1 ORDER BY period_ts", con)
    if len(df) < 5:
        return {"n": len(df), "note": "need >=5 resolved periods"}
    out = {"n": int(len(df))}
    for col in ("agg_score", "agg_uniform"):
        out[col] = metrics(df[col].values, df["realized_return"].values)
    # sanity baseline: shuffle scores; correlation should collapse to ~0
    rng = np.random.default_rng(0)
    shuf = [stats.pearsonr(rng.permutation(df["agg_score"]), df["realized_return"])[0] for _ in range(200)]
    out["shuffled_pearson_95pct"] = float(np.percentile(np.abs(shuf), 95))
    return out

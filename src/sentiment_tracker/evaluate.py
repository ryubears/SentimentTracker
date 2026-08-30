"""
Does the score predict BTC? Compares adaptive vs uniform weights on resolved periods only.
Both aggregates were computed with weights known at the time, so there is no look-ahead.
"""

from __future__ import annotations
from scipy import stats
import numpy as np
import pandas as pd

def metrics(x: np.ndarray, r: np.ndarray) -> dict:
    """
    Score-vs-realized-return metrics for one (score, return) series. Pulled out of
    evaluate() so sweep.py's walk-forward folds score themselves the same way.
    """
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

def strategy_returns(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Per-period return from following the score: long when bullish, short when
    bearish, flat when the score is exactly 0."""
    x, r = np.asarray(x, dtype=float), np.asarray(r, dtype=float)
    return np.sign(x) * r

def sortino(returns, periods_per_year: float = 365.0) -> float | None:
    """Annualized Sortino ratio against a 0 target: sqrt(ppy) * mean / downside
    deviation. None when there are no returns or no downside to measure."""
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return None
    downside = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)))
    if downside == 0.0:
        return None
    return float(np.sqrt(periods_per_year) * r.mean() / downside)

def rolling_corr(x, r, window: int = 14) -> dict[str, list[float | None]]:
    """Rolling Pearson/Spearman between score and realized return, aligned to the
    input (None until the window fills or while a window has no variance)."""
    x, r = np.asarray(x, dtype=float), np.asarray(r, dtype=float)
    out: dict[str, list[float | None]] = {"pearson": [], "spearman": []}
    for i in range(len(x)):
        wx, wr = x[max(0, i + 1 - window):i + 1], r[max(0, i + 1 - window):i + 1]
        if i + 1 < window or wx.std() == 0 or wr.std() == 0:
            out["pearson"].append(None)
            out["spearman"].append(None)
        else:
            out["pearson"].append(float(stats.pearsonr(wx, wr)[0]))
            out["spearman"].append(float(stats.spearmanr(wx, wr)[0]))
    return out

def evaluate(con) -> dict:
    df = pd.read_sql("SELECT * FROM periods WHERE resolved=1 ORDER BY period_ts", con)
    if len(df) < 5:
        return {"n": len(df), "note": "need >=5 resolved periods"}
    out = {"n": int(len(df))}
    for col in ("agg_score", "agg_uniform"):
        out[col] = metrics(df[col].values, df["realized_return"].values)

    # Sanity baseline: shuffle scores; correlation should collapse to ~0.
    rng = np.random.default_rng(0)
    shuf = [stats.pearsonr(rng.permutation(df["agg_score"]), df["realized_return"])[0] for _ in range(200)]
    out["shuffled_pearson_95pct"] = float(np.percentile(np.abs(shuf), 95))
    return out

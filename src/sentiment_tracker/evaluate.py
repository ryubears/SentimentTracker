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
    # A score of exactly 0 is "no directional call", not a wrong one. sign(0) never
    # matches sign(r), so counting flat periods as misses silently understates the
    # hit rate — badly once periods can be empty (hourly runs are ~29% flat).
    directional = x != 0
    return {
        "pearson": pearson,
        "spearman": spearman,
        "hit_rate": (float(np.mean(np.sign(x[directional]) == np.sign(r[directional])))
                     if directional.any() else None),
        "n_directional": int(directional.sum()),
        "mean_ret_when_bullish": float(r[x > 0].mean()) if (x > 0).any() else None,
        "mean_ret_when_bearish": float(r[x < 0].mean()) if (x < 0).any() else None,
    }

def strategy_returns(x: np.ndarray, r: np.ndarray, deadband: float = 0.0) -> np.ndarray:
    """
    Per-period return from following the score: long when bullish, short when
    bearish, flat when |score| <= deadband.

    The deadband exists because a near-zero aggregate carries no information —
    its sign is a coin flip — so taking a position on it is trading noise. Note
    that any deadband tuned on the same history it is evaluated over is an
    in-sample choice and will flatter itself; validate it on later periods.
    """
    x, r = np.asarray(x, dtype=float), np.asarray(r, dtype=float)
    return np.where(np.abs(x) > deadband, np.sign(x) * r, 0.0)

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

def evaluate(con, deadband: float = 0.0) -> dict:
    df = pd.read_sql("SELECT * FROM periods WHERE resolved=1 ORDER BY period_ts", con)
    if len(df) < 5:
        return {"n": len(df), "note": "need >=5 resolved periods"}
    out = {"n": int(len(df))}
    out["agg_score"] = metrics(df["agg_score"].values, df["realized_return"].values)

    # Gated view: only the periods with enough conviction to act on. Reported
    # alongside the ungated metrics so the filter can never hide the full picture.
    if deadband > 0:
        x, r = df["agg_score"].values, df["realized_return"].values
        act = np.abs(x) > deadband
        out["gated"] = {"deadband": deadband, "n_active": int(act.sum()),
                        "share_active": float(act.mean())}
        if act.sum() >= 5:
            out["gated"].update(metrics(x[act], r[act]))
            out["gated"]["mean_return_per_active_period"] = float(
                np.mean(np.sign(x[act]) * r[act]))

    # Sanity baseline: shuffle scores; correlation should collapse to ~0.
    rng = np.random.default_rng(0)
    shuf = [stats.pearsonr(rng.permutation(df["agg_score"]), df["realized_return"])[0] for _ in range(200)]
    out["shuffled_pearson_95pct"] = float(np.percentile(np.abs(shuf), 95))
    return out

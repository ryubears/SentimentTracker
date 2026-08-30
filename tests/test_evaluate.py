import sys; sys.path.insert(0, "src")
import numpy as np
from sentiment_tracker.evaluate import rolling_corr, sortino, strategy_returns

def test_strategy_returns_follow_the_score_direction():
    out = strategy_returns([0.5, -0.3, 0.0], [0.02, 0.02, 0.02])
    assert list(out) == [0.02, -0.02, 0.0] # long, short, flat.

def test_sortino():
    assert sortino([]) is None
    assert sortino([0.01, 0.02]) is None # no downside to measure.
    assert abs(sortino([0.02, -0.01], periods_per_year=1)
               - (0.005 / np.sqrt(0.0001 / 2))) < 1e-12
    assert sortino([0.01, -0.01], periods_per_year=365) == 0.0 # zero mean.

def test_rolling_corr_fills_after_window_and_tracks_correlation():
    x = list(np.linspace(-1, 1, 20))
    out = rolling_corr(x, x, window=5) # perfectly correlated with itself.
    assert len(out["pearson"]) == 20
    assert out["pearson"][:4] == [None] * 4 and out["spearman"][:4] == [None] * 4
    assert all(abs(v - 1.0) < 1e-9 for v in out["pearson"][4:])
    anti = rolling_corr(x, [-v for v in x], window=5)
    assert all(abs(v + 1.0) < 1e-9 for v in anti["spearman"][4:])

def test_rolling_corr_returns_none_for_flat_windows():
    out = rolling_corr([1.0] * 6, [0.1, -0.2, 0.3, -0.1, 0.2, 0.0], window=3)
    assert out["pearson"] == [None] * 6

def test_deadband_flattens_low_conviction_periods():
    x = [0.5, 0.005, -0.005, -0.5]
    r = [0.02, 0.02, 0.02, 0.02]
    assert list(strategy_returns(x, r)) == [0.02, 0.02, -0.02, -0.02] # ungated
    assert list(strategy_returns(x, r, deadband=0.01)) == [0.02, 0.0, 0.0, -0.02]

def test_deadband_of_zero_matches_ungated_behaviour():
    x, r = [0.3, -0.2], [0.01, 0.01]
    assert list(strategy_returns(x, r, 0.0)) == list(strategy_returns(x, r))

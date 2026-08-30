import sys; sys.path.insert(0, "src")
from sentiment_tracker.weights import AccountWeights, aggregate

def test_accurate_account_gains_weight_but_nobody_hits_zero():
    aw = AccountWeights(["good", "bad", "quiet"], eta=2.0, decay=0.8, floor=0.2)
    for _ in range(20):
        aw.update({"good": 0.8, "bad": -0.8}, realized_return=0.02)
    w = aw.weights()
    assert w["good"] > w["quiet"] > w["bad"]
    assert min(w.values()) >= 0.2 / 3 - 1e-9 # floor guarantees survival.

def test_recovery_after_streak():
    aw = AccountWeights(["a", "b"], decay=0.7)
    for _ in range(10):
        aw.update({"a": 1.0, "b": -1.0}, 0.01)
    for _ in range(10):
        aw.update({"a": -1.0, "b": 1.0}, 0.01) # b becomes right.
    assert aw.weights()["b"] > aw.weights()["a"]

def test_aggregate_ignores_silent_accounts():
    assert aggregate({"a": 1.0}, {"a": 0.1, "b": 0.9}) == 1.0

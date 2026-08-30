import sys; sys.path.insert(0, "src")
import threading
from sentiment_tracker import sentiment

LLM = {"backend": "llm", "model": "m", "horizon": "1d"}

def test_score_many_preserves_order_under_concurrency(monkeypatch):
    # Reply with a score derived from the text, after a stagger that guarantees
    # completions arrive out of submission order.
    def fake(text, cfg):
        n = int(text)
        threading.Event().wait(0.02 if n % 2 else 0.001)
        return n / 100.0
    monkeypatch.setattr(sentiment, "score_post", fake)
    texts = [str(i) for i in range(40)]
    assert sentiment.score_many(texts, LLM, workers=8, progress_every=0) == \
        [i / 100.0 for i in range(40)]

def test_score_many_returns_none_for_failures_without_losing_the_rest(monkeypatch):
    def fake(text, cfg):
        if text == "boom":
            raise RuntimeError("api exploded")
        return 0.5
    monkeypatch.setattr(sentiment, "score_post", fake)
    out = sentiment.score_many(["ok", "boom", "ok"], LLM, workers=4, progress_every=0)
    assert out == [0.5, None, 0.5] # the failure is isolated, not fatal.

def test_score_many_runs_serially_for_vader(monkeypatch):
    calls = []
    monkeypatch.setattr(sentiment, "score_post", lambda t, c: calls.append(t) or 0.1)
    assert sentiment.score_many(["a", "b"], {"backend": "vader"}, progress_every=0) == [0.1, 0.1]
    assert calls == ["a", "b"]

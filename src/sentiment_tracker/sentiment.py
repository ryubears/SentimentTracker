"""
Per-post sentiment in [-1, 1]. Two backends so you can show a baseline vs LLM comparison.
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
import json
import os

DEFAULT_WORKERS = 12

PROMPT = """You are scoring crypto social-media posts for their implied view on BTC price
over the next {horizon}. Respond with ONLY a JSON object: {{"score": float in [-1,1], "confidence": float in [0,1]}}.
-1 = strongly bearish, 0 = neutral/irrelevant, +1 = strongly bullish. Irrelevant posts get score 0, confidence 0.

Post:
{text}"""

def score_vader(text: str) -> float:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer().polarity_scores(text)["compound"]

@functools.lru_cache(maxsize=1)
def _client():
    """One shared client for the process. Constructing an Anthropic() per call
    builds a fresh connection pool each time, which dominates latency once you
    are scoring thousands of posts. The sync client is thread-safe."""
    import anthropic
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=5)

def score_llm(text: str, model: str, horizon: str) -> float:
    client = _client()
    msg = client.messages.create(
        model=model, max_tokens=100,
        messages=[{"role": "user", "content": PROMPT.format(text=text, horizon=horizon)}],
    )
    raw = msg.content[0].text.strip().strip("`").removeprefix("json").strip()
    d = json.loads(raw)
    # Weight the score by confidence so hedged posts count less.
    return max(-1.0, min(1.0, float(d["score"]) * float(d.get("confidence", 1.0))))

def score_post(text: str, cfg: dict) -> float:
    if cfg["backend"] == "llm":
        return score_llm(text, cfg["model"], cfg.get("horizon", "24 hours"))
    return score_vader(text)

def score_many(texts: list[str], cfg: dict, workers: int = DEFAULT_WORKERS,
               progress_every: int = 250) -> list[float | None]:
    """
    Score many posts, in order. LLM scoring is one network round trip per post,
    so a backfill of thousands runs for hours serially — these are independent
    and I/O-bound, so they go through a thread pool instead.

    A post whose scoring keeps failing gets None rather than a fabricated score:
    callers drop those so a later run retries them, instead of persisting a wrong
    value that the cache would then serve forever.
    """
    if cfg["backend"] != "llm" or len(texts) < 2:
        return [score_post(t, cfg) for t in texts]

    scores: list[float | None] = [None] * len(texts)
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(score_post, t, cfg): i for i, t in enumerate(texts)}
        for n, fut in enumerate(as_completed(futures), start=1):
            try:
                scores[futures[fut]] = fut.result()
            except Exception as e:
                failures += 1
                if failures <= 3: # A handful is enough to diagnose; don't spam.
                    print(f"  scoring error: {type(e).__name__}: {e}", flush=True)
            if progress_every and (n % progress_every == 0 or n == len(texts)):
                print(f"  scored {n}/{len(texts)}", flush=True)
    if failures:
        print(f"  {failures} post(s) unscored; a later run will retry them", flush=True)
    return scores

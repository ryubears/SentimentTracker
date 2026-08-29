"""Per-post sentiment in [-1, 1]. Two backends so you can show a baseline vs LLM comparison."""
from __future__ import annotations

import json
import os

PROMPT = """You are scoring crypto social-media posts for their implied view on BTC price
over the next {horizon}. Respond with ONLY a JSON object: {{"score": float in [-1,1], "confidence": float in [0,1]}}.
-1 = strongly bearish, 0 = neutral/irrelevant, +1 = strongly bullish. Irrelevant posts get score 0, confidence 0.

Post:
{text}"""


def score_vader(text: str) -> float:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer().polarity_scores(text)["compound"]


def score_llm(text: str, model: str, horizon: str) -> float:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=model, max_tokens=100,
        messages=[{"role": "user", "content": PROMPT.format(text=text, horizon=horizon)}],
    )
    raw = msg.content[0].text.strip().strip("`").removeprefix("json").strip()
    d = json.loads(raw)
    # confidence-weight the score so hedged posts count less
    return max(-1.0, min(1.0, float(d["score"]) * float(d.get("confidence", 1.0))))


def score_post(text: str, cfg: dict) -> float:
    if cfg["backend"] == "llm":
        return score_llm(text, cfg["model"], cfg.get("horizon", "24 hours"))
    return score_vader(text)

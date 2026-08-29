"""Adaptive per-account weights.

Design goals (from the project brief):
  1. Accounts that were recently accurate get higher weight.
  2. Nobody is ever thrown away — every account keeps a minimum weight.

Mechanism: each account carries an exponentially-decayed performance score
    S_a <- decay * S_a + (1 - decay) * reward_a
where reward_a = signal_a * sign(realized_return) lies in [-1, 1]
(agreeing with a big move and disagreeing with one are rewarded/penalised by the
signal's magnitude, so a confident wrong call hurts more than a lukewarm one).

Weights are a softmax over eta * S_a, mixed with a uniform distribution:
    w_a = (1 - floor) * softmax(eta * S)_a + floor / N
This is the Hedge / multiplicative-weights algorithm with recency decay plus a
uniform floor, which keeps every account "alive" so it can recover.
Accounts that did not post in a period are left untouched.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AccountWeights:
    accounts: list[str]
    eta: float = 2.0
    decay: float = 0.9
    floor: float = 0.2
    scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for a in self.accounts:
            self.scores.setdefault(a, 0.0)
        if not 0.0 <= self.floor < 1.0:
            raise ValueError("floor must be in [0, 1)")

    # ------------------------------------------------------------------ core
    def weights(self) -> dict[str, float]:
        names = list(self.scores)
        s = np.array([self.scores[a] for a in names], dtype=float)
        z = np.exp(self.eta * (s - s.max()))          # stable softmax
        soft = z / z.sum()
        n = len(names)
        w = (1.0 - self.floor) * soft + self.floor / n
        return dict(zip(names, w.tolist()))

    def update(self, signals: dict[str, float], realized_return: float) -> None:
        """signals: account -> sentiment in [-1, 1] for the period just resolved.
        realized_return: BTC return over the horizon that followed that period."""
        direction = float(np.sign(realized_return))
        if direction == 0.0:
            return
        for a, sig in signals.items():
            if a not in self.scores:
                self.scores[a] = 0.0        # new account starts neutral
            reward = float(np.clip(sig, -1, 1)) * direction
            self.scores[a] = self.decay * self.scores[a] + (1 - self.decay) * reward

    def add_account(self, name: str) -> None:
        self.scores.setdefault(name, 0.0)

    # ------------------------------------------------------------ persistence
    def to_json(self) -> str:
        return json.dumps({"eta": self.eta, "decay": self.decay,
                           "floor": self.floor, "scores": self.scores})

    @classmethod
    def from_json(cls, text: str) -> "AccountWeights":
        d = json.loads(text)
        return cls(accounts=list(d["scores"]), eta=d["eta"], decay=d["decay"],
                   floor=d["floor"], scores=d["scores"])


def aggregate(signals: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted mean sentiment over accounts that posted this period."""
    active = {a: s for a, s in signals.items() if a in weights}
    if not active:
        return 0.0
    num = sum(weights[a] * s for a, s in active.items())
    den = sum(weights[a] for a in active)
    return num / den if den else 0.0

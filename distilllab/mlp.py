"""One-hidden-layer tanh MLP classifier with hand-derived backprop."""

from __future__ import annotations

import math
import random
from typing import List, Tuple


def softmax(logits: List[float], temperature: float = 1.0) -> List[float]:
    z = [t / temperature for t in logits]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


Grads = Tuple[List[List[float]], List[float], List[List[float]], List[float]]


class MLP:
    def __init__(self, dim: int, hidden: int, n_classes: int, rng: random.Random):
        self.dim, self.hidden, self.n_classes = dim, hidden, n_classes
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(dim)] for _ in range(hidden)]
        self.b1 = [0.0] * hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(hidden)] for _ in range(n_classes)]
        self.b2 = [0.0] * n_classes

    def hidden_act(self, x: List[float]) -> List[float]:
        return [math.tanh(sum(w * xi for w, xi in zip(row, x)) + b)
                for row, b in zip(self.W1, self.b1)]

    def logits(self, x: List[float]) -> List[float]:
        h = self.hidden_act(x)
        return [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]

    def predict(self, x: List[float]) -> int:
        lg = self.logits(x)
        return lg.index(max(lg))

    def accuracy(self, data) -> float:
        return sum(1 for x, y in data if self.predict(x) == y) / max(1, len(data))

    def param_count(self) -> int:
        return self.hidden * self.dim + self.hidden + self.n_classes * self.hidden + self.n_classes

    def apply_grads(self, grads: Grads, lr: float) -> None:
        gW1, gb1, gW2, gb2 = grads
        for k in range(self.n_classes):
            for i in range(self.hidden):
                self.W2[k][i] -= lr * gW2[k][i]
            self.b2[k] -= lr * gb2[k]
        for i in range(self.hidden):
            for j in range(self.dim):
                self.W1[i][j] -= lr * gW1[i][j]
            self.b1[i] -= lr * gb1[i]

    def backprop(self, x: List[float], dl: List[float]) -> Grads:
        """Given d loss / d logits, backprop to parameter gradients."""
        h = self.hidden_act(x)
        gW2 = [[dl[k] * h[i] for i in range(self.hidden)] for k in range(self.n_classes)]
        gb2 = list(dl)
        dh = [sum(dl[k] * self.W2[k][i] for k in range(self.n_classes)) * (1.0 - h[i] ** 2)
              for i in range(self.hidden)]
        gW1 = [[dh[i] * x[j] for j in range(self.dim)] for i in range(self.hidden)]
        gb1 = dh
        return gW1, gb1, gW2, gb2

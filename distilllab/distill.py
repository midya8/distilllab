"""Distillation losses and gradients, all in closed form.

Combined objective per sample:
    L = alpha * T^2 * CE(softmax(z_s / T), softmax(z_t / T)) + (1 - alpha) * CE(softmax(z_s), onehot)

d CE(softmax(z/T), p) / dz = (softmax(z/T) - p) / T, so the soft term's
logit gradient is alpha * T * (p_s^T - p_t^T): the T^2 prefactor restores the
scale that dividing logits by T removes, leaving one factor of T from the
chain rule. The hard term's gradient is (1 - alpha) * (p_s - onehot).
"""

from __future__ import annotations

import math
import random
from typing import List

from .mlp import MLP, softmax


def onehot(cls: int, n_classes: int) -> List[float]:
    return [1.0 if k == cls else 0.0 for k in range(n_classes)]


def soft_targets(teacher: MLP, x: List[float], temperature: float) -> List[float]:
    return softmax(teacher.logits(x), temperature)


def distill_loss(student: MLP, x: List[float], soft: List[float], cls: int,
                 temperature: float, alpha: float) -> float:
    lg = student.logits(x)
    pT = softmax(lg, temperature)
    p1 = softmax(lg)
    soft_ce = -sum(t * math.log(max(p, 1e-12)) for t, p in zip(soft, pT))
    hard_ce = -math.log(max(p1[cls], 1e-12))
    return alpha * temperature ** 2 * soft_ce + (1.0 - alpha) * hard_ce


def distill_grads(student: MLP, x: List[float], soft: List[float], cls: int,
                  temperature: float, alpha: float):
    lg = student.logits(x)
    pT = softmax(lg, temperature)
    p1 = softmax(lg)
    dl = [alpha * temperature * (p - t) + (1.0 - alpha) * (p_ - (1.0 if k == cls else 0.0))
          for k, (p, t, p_) in enumerate(zip(pT, soft, p1))]
    return student.backprop(x, dl)


def hard_loss(student: MLP, x: List[float], cls: int) -> float:
    p = softmax(student.logits(x))
    return -math.log(max(p[cls], 1e-12))


def hard_grads(student: MLP, x: List[float], cls: int):
    p = softmax(student.logits(x))
    dl = [pi - (1.0 if k == cls else 0.0) for k, pi in enumerate(p)]
    return student.backprop(x, dl)


def mean_kl(student: MLP, teacher: MLP, data, temperature: float = 1.0) -> float:
    """Mean KL(teacher || student) over the data at the given temperature."""
    total = 0.0
    for x, _ in data:
        pt = softmax(teacher.logits(x), temperature)
        ps = softmax(student.logits(x), temperature)
        total += sum(t * math.log(max(t, 1e-12) / max(s, 1e-12)) for t, s in zip(pt, ps))
    return total / max(1, len(data))

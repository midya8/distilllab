"""Synthetic classification fixture with ring-structured class similarity.

Eight classes on a ring in 32 dimensions: each class center blends its own
random unit direction with its predecessor's, so neighboring classes are
genuinely similar and the teacher's soft targets carry real information about
that similarity (dark knowledge) which one-hot labels cannot. Inputs are class
centers plus seeded Gaussian noise. The student sees only a small subsample,
so per-sample information content is what separates training regimes.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

N_CLASSES = 8
DIM = 32
NOISE = 0.3


def _centers() -> List[List[float]]:
    rng = random.Random(7)
    dirs = [[rng.gauss(0.0, 1.0) for _ in range(DIM)] for _ in range(N_CLASSES)]
    for d in dirs:
        n = math.sqrt(sum(v * v for v in d))
        for i in range(DIM):
            d[i] /= n
    return [[1.2 * (0.75 * dirs[k][i] + 0.25 * dirs[(k - 1) % N_CLASSES][i])
             for i in range(DIM)] for k in range(N_CLASSES)]


CENTERS = _centers()


def make_split(n_per_class: int, seed: int) -> List[Tuple[List[float], int]]:
    rng = random.Random(seed)
    out = []
    for cls in range(N_CLASSES):
        for _ in range(n_per_class):
            x = [c + rng.gauss(0.0, NOISE) for c in CENTERS[cls]]
            out.append((x, cls))
    rng.shuffle(out)
    return out


# Teacher sees the large split; students see a small disjoint one; test is held out.
TEACHER_SPLIT = (100, 1)
STUDENT_SPLIT = (10, 3)
TEST_SPLIT = (40, 2)

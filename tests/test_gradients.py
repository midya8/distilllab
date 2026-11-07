import math
import random
import unittest

from distilllab.data import DIM, N_CLASSES, make_split
from distilllab.distill import (distill_grads, distill_loss, hard_grads, hard_loss,
                                soft_targets)
from distilllab.mlp import MLP


def _flat_params(net):
    return ([v for row in net.W1 for v in row] + list(net.b1)
            + [v for row in net.W2 for v in row] + list(net.b2))


def _set_flat(net, flat):
    it = iter(flat)
    net.W1 = [[next(it) for _ in range(net.dim)] for _ in range(net.hidden)]
    net.b1 = [next(it) for _ in range(net.hidden)]
    net.W2 = [[next(it) for _ in range(net.hidden)] for _ in range(net.n_classes)]
    net.b2 = [next(it) for _ in range(net.n_classes)]


def _flat_grads(g):
    gW1, gb1, gW2, gb2 = g
    return [v for row in gW1 for v in row] + list(gb1) + [v for row in gW2 for v in row] + list(gb2)


def _fd_check(loss_fn, grads_fn, net, tol=1e-5):
    base = _flat_params(net)
    analytic = _flat_grads(grads_fn())
    eps = 1e-6
    for i in range(len(base)):
        plus = list(base); plus[i] += eps
        minus = list(base); minus[i] -= eps
        _set_flat(net, plus); lp = loss_fn()
        _set_flat(net, minus); lm = loss_fn()
        numeric = (lp - lm) / (2 * eps)
        if abs(numeric - analytic[i]) > tol:
            return i, numeric, analytic[i]
    _set_flat(net, base)
    return None


class TestGradients(unittest.TestCase):
    def test_hard_loss_grads_fd_verified(self):
        data = make_split(2, 11)
        net = MLP(DIM, 6, N_CLASSES, random.Random(5))
        x, y = data[0]
        bad = _fd_check(lambda: hard_loss(net, x, y), lambda: hard_grads(net, x, y), net)
        self.assertIsNone(bad, f"hard-loss gradient mismatch at {bad}")

    def test_distill_loss_grads_fd_verified(self):
        data = make_split(2, 11)
        teacher = MLP(DIM, 8, N_CLASSES, random.Random(0))
        net = MLP(DIM, 6, N_CLASSES, random.Random(5))
        x, y = data[0]
        soft = soft_targets(teacher, x, 4.0)
        bad = _fd_check(lambda: distill_loss(net, x, soft, y, 4.0, 0.95),
                        lambda: distill_grads(net, x, soft, y, 4.0, 0.95), net)
        self.assertIsNone(bad, f"distill-loss gradient mismatch at {bad}")

    def test_soft_targets_match_teacher_argmax(self):
        data = make_split(2, 11)
        teacher = MLP(DIM, 8, N_CLASSES, random.Random(0))
        x, _ = data[0]
        soft = soft_targets(teacher, x, 4.0)
        self.assertEqual(soft.index(max(soft)), teacher.predict(x))
        self.assertAlmostEqual(sum(soft), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()

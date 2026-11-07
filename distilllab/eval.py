"""The measured experiment: teacher, direct student, distilled student, control.

- Teacher: hidden 64, trained on the 100-per-class split.
- Students: hidden 6, trained on the 10-per-class split with the same seed,
  optimizer budget, and learning rate; only the targets differ.
- direct: one-hot labels. distill: teacher soft targets at temperature T
  blended with the label (alpha). control: teacher soft targets for a
  DIFFERENT, deterministically permuted input - same target distribution,
  zero input-specific information. ablation_t1: same as distill but T=1,
  where soft targets carry almost no similarity information.
"""

from __future__ import annotations

import random

from .data import N_CLASSES, DIM, TEACHER_SPLIT, STUDENT_SPLIT, TEST_SPLIT, make_split
from .distill import distill_grads, hard_grads, mean_kl, soft_targets
from .mlp import MLP

TEACHER_HIDDEN, TEACHER_EPOCHS = 64, 100
STUDENT_HIDDEN, STUDENT_EPOCHS = 6, 150
LR = 0.02
TEMPERATURE, ALPHA, CONTROL_TEMPERATURE = 4.0, 0.95, 3.0


def train_teacher(data) -> MLP:
    t = MLP(DIM, TEACHER_HIDDEN, N_CLASSES, random.Random(0))
    for _ in range(TEACHER_EPOCHS):
        for x, y in data:
            t.apply_grads(hard_grads(t, x, y), LR)
    return t


def train_student(data, mode: str, teacher: MLP, temperature: float = TEMPERATURE,
                  alpha: float = ALPHA, perm=None) -> MLP:
    s = MLP(DIM, STUDENT_HIDDEN, N_CLASSES, random.Random(5))
    for _ in range(STUDENT_EPOCHS):
        for i, (x, y) in enumerate(data):
            if mode == "direct":
                s.apply_grads(hard_grads(s, x, y), LR)
            elif mode == "distill":
                s.apply_grads(distill_grads(s, x, soft_targets(teacher, x, temperature), y,
                                            temperature, alpha), LR)
            elif mode == "control":
                xc, _ = data[perm[i]]
                s.apply_grads(distill_grads(s, x, soft_targets(teacher, xc, temperature), y,
                                            temperature, alpha), LR)
            else:
                raise ValueError(mode)
    return s


def run_evaluation() -> dict:
    big = make_split(*TEACHER_SPLIT)
    small = make_split(*STUDENT_SPLIT)
    test = make_split(*TEST_SPLIT)

    teacher = train_teacher(big)
    perm = list(range(len(small)))
    random.Random(99).shuffle(perm)

    direct = train_student(small, "direct", teacher)
    distilled = train_student(small, "distill", teacher)
    control = train_student(small, "control", teacher, temperature=CONTROL_TEMPERATURE, perm=perm)
    abl_t1 = train_student(small, "distill", teacher, temperature=1.0)

    r = {
        "teacher_acc": teacher.accuracy(test),
        "teacher_params": teacher.param_count(),
        "student_params": direct.param_count(),
        "direct_acc": direct.accuracy(test),
        "direct_train_acc": direct.accuracy(small),
        "distilled_acc": distilled.accuracy(test),
        "distilled_train_acc": distilled.accuracy(small),
        "control_acc": control.accuracy(test),
        "ablation_t1_acc": abl_t1.accuracy(test),
        "kl_direct": mean_kl(direct, teacher, test),
        "kl_distilled": mean_kl(distilled, teacher, test),
    }
    r["distill_gain"] = r["distilled_acc"] - r["direct_acc"]
    r["overfit_gap_direct"] = r["direct_train_acc"] - r["direct_acc"]
    r["overfit_gap_distilled"] = r["distilled_train_acc"] - r["distilled_acc"]
    return r


def format_terminal(r: dict) -> str:
    return "\n".join([
        f"Teacher test accuracy:          {r['teacher_acc']:.4f}   ({r['teacher_params']} params)",
        f"Student (direct) test accuracy: {r['direct_acc']:.4f}   train {r['direct_train_acc']:.4f}",
        f"Student (distilled) accuracy:   {r['distilled_acc']:.4f}   train {r['distilled_train_acc']:.4f}   ({r['student_params']} params)",
        f"Distillation gain:              {r['distill_gain']:+.4f}",
        f"Control (mismatched targets):   {r['control_acc']:.4f}",
        f"Ablation T=1:                   {r['ablation_t1_acc']:.4f}   (vs T=4: {r['distilled_acc']:.4f})",
        f"KL(teacher||student):           direct {r['kl_direct']:.4f}   distilled {r['kl_distilled']:.4f}",
        f"Overfit gap (train - test):     direct {r['overfit_gap_direct']:.4f}   distilled {r['overfit_gap_distilled']:.4f}",
    ])

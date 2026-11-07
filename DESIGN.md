# DistillLab design

Knowledge distillation, dependency-free.

- data.py: 8 classes on a ring in 32 dims; center_k = 1.2 * (0.75 d_k + 0.25 d_{k-1})
  over random unit directions, so neighbors are similar. Noise 0.3. Splits:
  teacher 100/class (seed 1), student 10/class (seed 3), test 40/class (seed 2).
- mlp.py: one-hidden-layer tanh MLP; backprop takes d loss / d logits and
  returns parameter gradients; apply_grads steps. Everything seeded.
- distill.py: combined loss alpha T^2 CE(softmax(z/T), soft) + (1-a) CE(softmax(z), y).
  Logit gradient: alpha T (p_s^T - p_t^T) + (1-a) (p_s - onehot). The T^2
  restores scale lost by dividing logits by T; one T survives the chain rule.
  FD-verified in tests/test_gradients.py.
- eval.py: teacher hidden 64 / 100 epochs; students hidden 6 / 150 epochs, lr 0.02.
  Arms: direct, distill (T=4, alpha=0.95), control (targets from permuted inputs,
  perm seed 99), pseudo (teacher argmax as hard labels), smoothing (eps=0.1
  toward uniform), T=1 ablation. Metrics: accuracy, train/test gap, KL, params.
  Gain decomposition: pseudo - direct (teacher-signal), distill - pseudo
  (non-target-class distribution content, i.e. dark knowledge on this fixture).
- cli.py: evaluate (terminal/json) and gates on the primary arms
  (teacher strength, distillation gain); controls are reported, not gated.

Measured on the checked-in fixture: teacher 0.8344; direct 0.6813 (train 1.0);
distilled 0.7469 (train 0.8); pseudo 0.7094; smoothing 0.6781; control 0.1250
= chance; T=1 0.7219; KL 1.0932 -> 0.9563; gain +0.0656 decomposed as
+0.0281 teacher-signal + +0.0375 dark-knowledge component. Pinned in tests.

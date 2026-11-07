# DistillLab

Knowledge distillation from scratch: a large teacher trained on abundant data, a 10x smaller student trained on a small split, and the measured gain decomposed by matched control arms into teacher-prediction transfer and the non-target-class distribution content beyond it. No ML frameworks.

## What it does

- Builds a ring-structured classification fixture: eight classes in 32 dimensions where each class center blends in its predecessor's direction, so neighboring classes are genuinely similar. A teacher's soft targets then carry per-sample information about class similarity that one-hot labels cannot.
- Trains the teacher (hidden 64) on 100 samples per class, and students (hidden 6, 254 params vs 2632) on only 10 per class - the regime where distillation is supposed to matter.
- Implements the full combined loss by hand: `alpha * T^2 * CE(softmax(z/T), soft_targets) + (1 - alpha) * CE(softmax(z), label)`, with the logit gradient `alpha * T * (p_s^T - p_t^T) + (1 - alpha) * (p_s - onehot)` derived and finite-difference verified in the suite.
- Runs five student arms with identical seeds, capacity, and optimizer budget: direct (one-hot), distilled (teacher soft targets at T=4, alpha=0.95), pseudo-labels (the teacher's argmax predictions as hard labels on the same inputs), uniform label smoothing (input-independent softening), and a control whose soft targets come from *different, permuted inputs* - same target distribution, zero input-specific information.

## Measured results (checked-in fixture, pinned in tests)

```
Teacher test accuracy:          0.8344   (2632 params)
Student (direct) test accuracy: 0.6813   train 1.0000
Student (distilled) accuracy:   0.7469   train 0.8000   (254 params)
Distillation gain:              +0.0656
Pseudo-labels (teacher argmax): 0.7094   -> teacher-signal component +0.0281
Dark-knowledge component:       +0.0375   (distilled minus pseudo)
Label smoothing (uniform eps):  0.6781
Control (mismatched targets):   0.1250   (chance is 1/8 = 0.125)
Ablation T=1:                   0.7219   (vs T=4: 0.7469)
KL(teacher||student):           direct 1.0932   distilled 0.9563
Overfit gap (train - test):     direct 0.3187   distilled 0.0531
```

What the fixture shows, stated plainly:

- Distillation buys real generalization on this fixture: +0.0656 test accuracy over direct training at identical capacity and budget, recovering part of the gap toward the teacher (0.8344) with a 10x smaller model and a tenth of its training data.
- The mechanism is regularization, visible in the overfit gap: the direct student memorizes its 80 training samples (train 1.0000, gap 0.3187) while the distilled student does not (train 0.8000, gap 0.0531) and generalizes better.
- The gain decomposes on this fixture. Training on the teacher's own argmax predictions (pseudo-labels) accounts for +0.0281; the full soft distribution adds a further +0.0375 beyond them - that second component is the non-target-class similarity information the term dark knowledge refers to. Uniform label smoothing adds nothing (-0.0032 vs direct), so input-independent softening is not the mechanism. And the permuted-target control lands at exactly chance (0.1250), so input-mismatched teacher outputs carry nothing either. Each comparison is a fixture measurement between matched training arms, and the suite pins the two positive components from below.
- Temperature matters, modestly here: T=1 keeps the teacher's argmax supervision and still helps (0.7219 vs 0.6813 direct), and T=4 adds +0.0250 more. The suite pins the direction, not a large effect.
- The distilled student's predictive distribution is closer to the teacher's (KL 0.9563 vs 1.0932) - it is genuinely imitating, not just scoring higher.

## CLI

```bash
python -m distilllab evaluate
python -m distilllab gates --min-teacher-acc 0.80 --min-distill-gain 0.04
```

Gates exit 1 with named failures; the demo asserts both sides (an impossible distillation gain fails).

## Development

```bash
python -m unittest discover -s tests
bash examples/demo.sh
```

12 tests. No dependencies beyond Python 3.10+.

## Limitations

Everything here is a small synthetic classification fixture: real distillation runs over language-model logits, uses much larger capacity gaps, and often drops the hard-label term entirely. The transferable mechanics are the measured ones: the temperature-scaled combined loss with its exact gradient, the per-sample regularization signature (train-accuracy gap), and the matched control arms that decompose the gain. One honest caveat: the fixture's distillation gain, while consistent and gated, is a few accuracy points - it demonstrates the mechanism, not a production-scale win.

## License

MIT

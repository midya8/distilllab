# DistillLab

Knowledge distillation from scratch: a large teacher trained on abundant data, a 10x smaller student trained on a small split, and the measured gain compared against matched control arms. No ML frameworks.

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
Pseudo-labels (teacher argmax): 0.7094   (+0.0281 over direct)
Distilled vs pseudo-labels:     +0.0375   (descriptive, not causal attribution)
Label smoothing (uniform eps):  0.6781
Control (mismatched targets):   0.1250
Ablation T=1:                   0.7219   (vs T=4: 0.7469)
KL(teacher||student):           direct 1.0932   distilled 0.9563
Overfit gap (train - test):     direct 0.3187   distilled 0.0531
```

What the fixture shows, stated plainly:

- Distillation buys real generalization on this fixture: +0.0656 test accuracy over direct training at identical capacity and budget, recovering part of the gap toward the teacher (0.8344) with a 10x smaller model and a tenth of its training data.
- The measured train-test gap is smaller: the direct student memorizes its 80 training samples (train 1.0000, gap 0.3187) while the distilled student does not (train 0.8000, gap 0.0531) and generalizes better.
- The arms separate on this fixture, but the differences are measured comparisons, not causal identification. Pseudo-label training on the teacher's argmax exceeds direct by +0.0281, and full soft-label distillation exceeds pseudo-labels by a further +0.0375 - consistent with the non-target-class distribution carrying usable information, though this comparison alone does not isolate why (soft targets also change target confidence and the optimization problem). Uniform label smoothing at eps=0.1 adds nothing (-0.0032 vs direct) - one setting, which cannot rule out input-dependent smoothing or other settings. The permuted-target control lands at exactly chance (0.1250). The suite pins the arm ordering from below as deterministic fixture regression.
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

Everything here is a small synthetic classification fixture: real distillation runs over language-model logits, uses much larger capacity gaps, and often drops the hard-label term entirely. The transferable mechanics are the measured ones: the temperature-scaled combined loss with its exact gradient, the per-sample regularization signature (train-accuracy gap), and the matched control arms that bound the between-arm comparisons. One honest caveat: the fixture's distillation gain, while consistent and gated, is a few accuracy points - it demonstrates the effect on this fixture, not a production-scale win.

## License

MIT

## v0.2: inspect one real student update

[Live gradient inspector](https://midya8.github.io/distilllab/) · [Equations, input/report schema and limits](STEP.md)

Select one of the six original trained student arms (or its untrained initialization), any of the 80 original student-training rows, T, alpha and a step size. Independent native and browser arithmetic expose the hard/soft class derivatives, all 254 parameter updates, hidden activations, selected-row CE/KL and every prediction on the original 80 training/320 test rows. Every entered state starts from the selected original model. No test label enters its gradient. Full input/report JSON is copyable/downloadable and compatible with the new `step-audit` command.

A supplied two-hidden-unit control visibly overshoots: loss rises from 1.3466 to 3.6897 at eta=2 despite a negative local directional derivative. This is an arithmetic counterexample, not a new trained arm. The inspector uses stable, unclipped logit-space losses; the original probability-clipped helper functions, original gradients, training, controls and results stay unchanged. Their distinction below the 1e-12 probability floor is explicit in STEP.md.

24 Python tests (the original 12 retained), four Node/actual-Chrome tests, 36 complete native/browser witnesses, 160 independent 70-digit Decimal logit controls, 80 uniform-class controls and all-parameter finite differences check the implementation. Witnesses expose 8,900 parameter updates and 14,001 evaluation positions. Installed wheel reproduction rebuilds the original teacher/six student fits, full data/logits/report and every witness outside the checkout.

```bash
python -m distilllab step-audit input.json --temperature 4 --alpha .95 --step .02
python -m distilllab step-demo regenerated
node --test tests/test_step.cjs tests/test_step_browser.cjs
bash examples/verify_wheel.sh
```

One finite step is not a training run, a guarantee of improvement or a causal explanation of distillation's gain. Temperature changes both targets and objective scale. Original synthetic test metrics are descriptive, not new held-out evidence or a deployment guarantee. Standard distillation and backpropagation are attributed to the sources in STEP.md; no new method is claimed.

## v0.3: keep the confidence, change the tail

A separate target/rest KL workbench audits the established chain rule. Reproduction passes 36 Python tests retaining all 24 earlier tests and seven substantive Node/Chrome tests (eight runner entries including the existing fixture helper). Twenty-eight full witnesses cover 112 variants, 21,724 parameter positions and 286,603 cross-runtime numeric fields. Actual Chrome checks all 84 control-width layouts, 545,388 displayed numeric table cells, 65,172 coordinate selections and six completed downloads, with three archives replayed natively. Fresh installed wheels check both new entrypoints and all original training/step reproduction unchanged. It decomposes the exact coupled KL loss and every student-parameter gradient, then compares four teacher-tail distributions with the same target/rest mass at the selected temperature. The browser uses independent scalar reverse-mode automatic differentiation, not saved native gradients. The original trainer, six trained student arms, data, reports and one-step inspector remain unchanged.

[Equations, bounded input/report/replay schemas and limits](TAIL.md) explain the established Zhao et al. (CVPR 2022) decomposition, teacher-only CE/KL entropy offset, true excluded-class normalization, all-parameter checks and why this is neither DKD training nor a causal explanation of the original gain. Public-source reproduction, feature/PR/main CI and actual deployed bytes, behavior and pixels are separate release gates; local checks alone are not a public deployment claim.

# Target/rest KL and conditional-tail gradients

DistillLab 0.3 is a separate arithmetic workbench. It does not change the original trainer, six student arms, frozen models, earlier tests or the original one-step inspector. It implements an established KL chain rule, not a new distillation method or a DKD training run.

[Browser workbench](https://midya8.github.io/distilllab/tail.html)

## Exact identity and derivatives

For supplied training label `y`, temperature `T`, fixed teacher `q = softmax(teacher_logits / T)` and student `p = softmax(student_logits / T)`, define `qm = sum(q[j], j != y)`, `pm = sum(p[j], j != y)`, `r[j] = q[j]/qm`, `s[j] = p[j]/pm`. Then

```
KL(q || p) = KL([q[y], qm] || [p[y], pm]) + qm * KL(r || s)
                binary target/rest             conditional tail
```

This is the decomposition in Zhao et al. (CVPR 2022), equations 5-6 and the supplemental proof. It is also a special case of the KL chain rule. The target is the supplied label, which need not equal the teacher's argmax. In particular, a wrong teacher can have low target mass while being confident in a different class.

The audited objective is `(1-alpha) * hard_CE(T=1) + alpha*T*T*KL(q||p)`. The earlier trainer and step inspector use soft **cross entropy**, not KL. For a fixed teacher these have the same student derivative, but their values differ by `alpha*T*T*H(q)`. The report shows both objectives and the teacher-only offset. Interventions can change teacher entropy, so this distinction is not hidden in comparisons.

Before multiplying by alpha, the derivatives with respect to the **unscaled student logits** are:

```
d(T² * binary_KL)/dz[y] = T * (p[y] - q[y])
d(T² * binary_KL)/dz[j] = T * (q[y] - p[y]) * s[j]    j != y

d(T² * conditional_KL)/dz[y] = 0
d(T² * conditional_KL)/dz[j] = T * (s[j] - r[j])       j != y
```

Multiplying the conditional derivative by fixed teacher mass `qm`, then adding the binary derivative, recovers `T*(p-q)` in exact arithmetic. The hard derivative at T=1 is separately weighted by `(1-alpha)`. Native code uses product differences for the binary/full-class derivatives to reduce cancellation under saturation. All parameters update simultaneously from the same original model. Three separately backpropagated parameter components are compared with a direct full-class KL derivative. The original teacher is held fixed throughout each derivative and finite step.

The browser is independent scalar reverse-mode automatic differentiation, including tanh, logits, subset logsumexp, each loss component and the direct full-class objective. It does not replay native derivatives or use an ML framework. Native Python uses analytic MLP backprop, with separate centered finite-difference and 80-digit Decimal controls. Both use binary64 transcendental functions; this is numerical agreement, not an exact-real certificate.

T² scaling does not make the exact gradient temperature-invariant. Both teacher and student distributions change with T. No temperature or component weight is selected from test scores.

## Four visible variants

Each starts from the same student model and training row:

- `original`: the supplied teacher at the selected T.
- `uniform_tail`: replace the conditional non-target distribution with uniform probabilities, keeping target/rest mass.
- `rotated_tail`: left-rotate non-target probabilities by one class, in ascending class order excluding y. This preserves both masses and teacher entropy in exact arithmetic. It need not preserve class semantics.
- `configured`: left-rotate by `tail_shift`, then mix with uniform using `tail_mix` in [0,1]. Both endpoints and the intermediate distribution are valid. All four variants remain visible, including worse outcomes.

Only target/rest mass **at the selected temperature** is held fixed. These are constructed probability distributions, not claims about a single temperature-independent teacher's logits. The input's original teacher logits remain unchanged. For two classes the conditional tail has one class, all interventions are inert and its conditional KL/gradient is zero.

A finite step can overshoot despite a negative first-order directional derivative. The report shows own-teacher KL change and change against the original teacher, each with its correct before baseline. Changing teacher tails also changes the objective. Neither this arithmetic identity nor a one-row intervention identifies the cause of the earlier multi-epoch accuracy gain. No new training, generalization, calibration, optimality or production-scale result is claimed.

## Stable subsets and bounds

Target/rest normalization is evaluated with a true excluded-class logsumexp. Conditional tails are normalized directly from their selected logits, not by dividing underflowed full-class probabilities, subtracting a rounded target probability from 1, or suppressing the target with a finite minus-1000 mask. The official upstream DKD implementation uses such a finite mask; this workbench is an independent bounded implementation and does not claim byte-for-byte upstream equivalence.

At extreme logits, displayed target probability can round to 1 and rest mass can be a small positive number or underflow to zero. Conditional tail probabilities still come from their own logits. Separate masses and raw residuals are exposed; small negative KL values from rounding are not clamped into an apparent proof.

Bounds: 1-64 inputs, 1-16 hidden units, 2-16 classes; weights/biases in [-100,100], input features in [-20,20], teacher logits in [-10,000,10,000], T in [0.25,20], alpha/mix in [0,1], step in [0,2], integer shift in [0,C-2]. Unknown fields, nonfinite values, duplicate JSON keys and malformed shapes are refused. JSON inputs are capped at 1.5 million characters; replay archives at 20 million. All operations stay local; there are no API calls, secrets or uploads.

## Input, report and replay schemas

Input schema: `distilllab.tail.input.v1`, exactly these fields:

```json
{
  "schema": "distilllab.tail.input.v1",
  "model": {"W1": [[0.2]], "b1": [0.1], "W2": [[0.3], [-0.1]], "b2": [0, 0]},
  "row": {"x": [0.5], "label": 0, "teacher_logits": [2, -2]},
  "temperature": 4,
  "alpha": 0.95,
  "step": 0.02,
  "tail_mix": 0.25,
  "tail_shift": 0
}
```

The model is the original one-hidden-layer tanh MLP format. Report `distilllab.tail.report.v1` includes all class probabilities/logs, both binary masses, conditional distributions in ascending non-target class order, loss components and all parameter gradients/updates for four variants. Each variant has before, after and original-teacher-after values. Parameter order is W1 row-major, b1, W2 row-major, b2. Browser tables expose every coordinate and raw residual. The target's conditional probability is undefined and displayed as a dash; its conditional **derivative** is zero.

The download archive is exactly `{"schema":"distilllab.tail.archive.v1","input":...,"report":...}`. Native replay recomputes the full expected report from the embedded bounded input and checks every field, shape, key and numeric value, not only summaries. Numeric comparison uses `2e-9 + 2e-9*max(abs(actual),abs(expected))`, allowing cross-runtime roundoff. Replay is not cryptographic authenticity or an independent proof of method quality.

```bash
python -m distilllab.tail input.json > report.json
distilllab-tail input.json > report.json
python -m distilllab.tail downloaded-archive.json --verify
python -m distilllab.tail_fixtures regenerated
node --test tests/test_tail.cjs tests/test_tail_browser.cjs
bash examples/verify_tail_wheel.sh
```

No dependencies beyond Python 3.10+ for the native audit. Node 22+ and Chrome are needed for web tests. New installed-wheel checks work outside the source checkout and exercise both entrypoints. Earlier wheel/demo/regeneration checks remain separate and unchanged.

## Evidence scope

Twenty-eight complete native witnesses contain 112 variants and 21,724 parameter positions. Twenty-one inputs are three training rows from each of seven original archived states (six trained arms and initialization); seven inputs are labeled arithmetic controls. Independent native checks cover all seven original states and all 80 training rows, 142,240 parameter coordinates, plus 2,368 centered finite-difference component/coordinate checks, 40 high-precision logit controls and 336 separately altered report leaves. The original archived states and test splits are not refitted or overwritten.

Local actual Chrome checks all 28 inputs at 1280/390/320, every variant and parameter coordinate: 24,012 class cells, 521,376 parameter cells and 65,172 coordinate selections. Six real input/archive downloads complete; three downloaded archives are natively replayed. Three extra rightmost-scroll visual states check narrow exact-table access. Source/public/installed/CI/deployed checks are separate release gates. Local counts and checks are not evidence of public deployment. Exact release status belongs in the README and release record after those gates complete.

## Attribution and reading

- Zhao, Cui, Song, Qiu and Liang, [Decoupled Knowledge Distillation](https://arxiv.org/pdf/2203.08679), CVPR 2022. The exact coupled KL decomposition here is theirs/standard KL algebra. Their separately weighted DKD objective is different from the coupled objective audited here; this page does not implement their training schedule or reproduce their benchmark results.
- [Official supplemental derivation](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_supplemental.pdf).
- [Official DKD implementation](https://github.com/megvii-research/mdistiller/blob/master/mdistiller/distillers/DKD.py).
- [PyTorch KLDivLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.md): teacher/student orientation, log-space conventions and sample reduction.
- [Independent implementation discussion](https://distil.md/distillation-loss-pytorch): CE versus KL teacher-entropy offset and common scaling/reduction mistakes. Its empirical claims are not transferred to this project.

The original [STEP.md](STEP.md) and original distillation references remain applicable. The workbench is an educational, bounded audit, not new research attribution.

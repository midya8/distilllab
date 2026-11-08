# One simultaneous distillation step

This is an additive inspector, not a replacement for the original experiment. Standard knowledge distillation and tanh backpropagation are implemented separately in `step.py` and browser `step.js`. No model framework, service or paid API is used.

## Objective and chain rule

For a selected fixed input x and supplied label y, teacher logits v remain fixed with respect to student parameters theta. Let z(theta) be the student logits, p = softmax(z), q = softmax(z/T), and t = softmax(v/T).

- Hard CE = -log p_y.
- Soft CE = -sum_k t_k log q_k.
- Teacher entropy = -sum_k t_k log t_k.
- KL(t || q) = sum_k t_k (log t_k - log q_k), so CE = entropy + KL.
- L = (1-alpha) hard CE + alpha T^2 soft CE.
- dL/dz_k = (1-alpha)(p_k - 1[k=y]) + alpha T(q_k - t_k).

The remaining factor T is from the chain rule. T-squared scaling does not make every finite-temperature gradient identical. Comparing raw CE objectives at different T is not a general improvement ranking: target entropy and T-squared both change. Teacher entropy is constant only with respect to student parameters at fixed T.

For h_i = tanh(sum_j W1_ij x_j + b1_i), z_k = sum_i W2_ki h_i + b2_k:

- dL/dW2_ki = (dL/dz_k) h_i; dL/db2_k = dL/dz_k.
- dhpre_i = sum_k (dL/dz_k) W2_ki (1-h_i^2).
- dL/dW1_ij = dhpre_i x_j; dL/db1_i = dhpre_i.

Every derivative uses the original W2 and original activations. All parameters then update simultaneously: theta' = theta - eta grad L. Directional derivative along -grad L is -||grad L||²; the first-order predicted change is -eta ||grad L||². Neither guarantees a finite-step decrease. The supplied two-hidden-unit control at T=1, alpha=.5, eta=2 increases the selected-row loss from 1.3465855722521822 to 3.689675343344966 despite a negative first-order change. It is an arithmetic counterexample, not an original trained model.

## Stable objective versus original helpers

Log probabilities use shifted log-sum-exp. Per-class CE and KL terms stay finite even when a probability underflows to zero. Teacher logits are finite, so their log probabilities are computed before exponentiation and no `log(0)` is taken. Zero weighted terms use finite log values.

The original `distill.py` loss helpers clamp probabilities at 1e-12; their gradients use the ordinary, unclipped softmax formula. Below that floor those loss helpers are not the function whose derivative that formula computes. Original source, training and results are unchanged. The new inspector explicitly uses an unclipped logit-space objective and audits that derivative instead. It does not claim the original clipping convention has changed. Native original gradients and simultaneous updates match the new implementation on the original model states within documented numerical tolerance.

Binary64 exp/tanh can underflow/saturate; compensated sums reduce rounding, not eliminate it. Raw-logit first maximum defines prediction. No exact-real guarantee near ties or saturation is claimed.

## Input and report

`audit(payload, temperature=4, alpha=.95, step=.02)` accepts exactly:

```json
{"model":{"W1":[[0]],"b1":[0],"W2":[[0],[0]],"b2":[0,0]},"row":{"x":[0],"label":0,"teacher_logits":[0,0]},"evaluation":{"example":[{"x":[0],"label":0,"teacher_logits":[0,0]}]}}
```

Bounds: 1..64 inputs, 1..16 hidden units, 2..16 classes, finite model values ±100, inputs ±20, teacher logits ±10000, T .25..20, alpha 0..1, eta 0..2. Up to four nonempty named evaluation splits and 512 total evaluation rows. Labels are integer class positions, not bools. CLI and browser reject duplicate JSON keys; invalid browser input clears results and export state. JSON text limit is 1.5 million characters. Python object API accepts already-parsed dictionaries.

The report contains controls/dimensions, selected-row before/after losses and class terms, hidden/logit states, every named parameter/gradient/delta/result, updated full model, norms, local derivative/first-order change and all evaluation positions. Each evaluation row reports its supplied label, before/after predicted class, stable NLL and true-label probability at T=1. Split accuracy/NLL/changed counts use all supplied rows. Teacher logits on evaluation rows are retained in the input for provenance but do not enter the gradient. Evaluation labels can change metrics, never model updates.

CLI:

```bash
python -m distilllab step-audit input.json --temperature 4 --alpha .95 --step .02
python -m distilllab step-demo regenerated
```

The browser download wraps `input` and `report`. Save its `input` as input.json and pass the report's controls to step-audit. Copy input does exactly that. Input stays in the page.

## Verification and reproduction

24 Python tests include the original 12 unchanged tests. Four Node/actual-Chrome tests compare 36 complete native witnesses at 1e-12 absolute/relative tolerance, with exact keys/types/counts/discrete values (signed floating zero is numerically equivalent). Witnesses include 8,900 parameter records and 14,001 evaluation positions. Independent 70-digit Decimal checks cover 160 logit objectives/derivatives; 80 integer/uniform controls check exact chain-rule structure. Centered finite differences inspect every parameter of a 254-parameter original initialization and the 10-parameter overshoot network. Native original backprop/apply agrees for 21 original states.

Browser tests visit all 80 original training rows, all 400 split rows, every witness, invalid inputs, duplicate keys, clipboard failure/success, downloaded blob, keyboard increments, 320/390/1280 widths, 44px controls and actual rendered chart font size. Installed wheel reproduction imports seven modules and runs all four commands outside checkout; original teacher/six student fits, full data/logits/report and all 36 witnesses regenerate, without using README-rounded values.

## Scope

The frozen original experiment remains attached with all six student arms and its original scores. No original data, models, optimization settings, control arms, tests or result values are changed. The new intervention resets to a chosen original model for every entered state. It is not an epoch, repeated training, hyperparameter selection or a test-set optimization loop. Original synthetic test data are descriptive, not fresh external evaluation. A lower local loss or test NLL does not establish calibration, causal attribution, a unique distillation mechanism, robustness or deployment benefit. One smoothing setting and soft-versus-argmax comparisons do not isolate components.

## Sources

- Hinton, Vinyals and Dean (2015), equations 1-4 and section 2: https://arxiv.org/html/1503.02531
- PyTorch CrossEntropyLoss, logits and probability-target conventions: http://docs.pytorch.org/docs/main/generated/torch.nn.CrossEntropyLoss.html
- PyTorch KLDivLoss, log-space input and CE/entropy/KL distinction: https://docs.pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.md
- PyTorch functional KL, reduction caveats: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.md
- Stanford CS231n, chain rule/backpropagation: https://cs231n.github.io/optimization-2/
- Stanford CS231n, centered gradient checks and numerical caveats: https://cs231n.github.io/neural-networks-3/

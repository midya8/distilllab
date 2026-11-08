# Frozen original experiment and step witnesses

`frozen.json` contains the original teacher and six trained student models, the original initialization, all 80 student-training and 320 test rows with teacher logits, all original report values and one explicitly supplied two-hidden-unit overshoot control. Teacher fitting uses its original 800-row split, separately regenerated from the original seeds.

`witnesses.json.gz` is deterministic gzip (mtime zero) of a JSON array: 35 original-model interventions (seven models times five entered configurations) and one overshoot control. Each has `input` selectors and a complete native `report`, including every parameter and evaluation row. It is compressed only for transfer size; `gzip -dc` exposes ordinary JSON. No executable payload is embedded.

Regenerate from unchanged original training, not rounded README metrics:

```bash
python -m distilllab step-demo regenerated
python -m unittest discover -s tests
node --test tests/test_step.cjs tests/test_step_browser.cjs
bash examples/verify_wheel.sh
```

The stored fixtures are numerical regression references, not independent held-out evidence or evidence of a causal mechanism. Original test rows are used for descriptive post-update evaluation only. The update is supplied by a student-training row, never a test row.

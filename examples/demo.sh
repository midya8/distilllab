#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== evaluation ==="
python -m distilllab evaluate

echo
echo "=== gates (achievable set: must pass) ==="
python -m distilllab gates --min-teacher-acc 0.80 --min-distill-gain 0.04 --max-control-acc 0.20

echo
echo "=== gates (impossible distillation gain: must fail) ==="
if python -m distilllab gates --min-distill-gain 0.5; then
    echo "ERROR: impossible gate set passed"
    exit 1
else
    echo "impossible gate set failed as expected"
fi

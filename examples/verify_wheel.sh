#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
python -m pip wheel --no-deps "$root" --wheel-dir "$tmp/wheels" >/dev/null
python -m venv "$tmp/venv"
"$tmp/venv/bin/python" -m pip install --no-index "$tmp"/wheels/*.whl >/dev/null
cd "$tmp"
"$tmp/venv/bin/python" -m distilllab step-demo regenerated > demo.json 2>demo.err & demo_pid=$!
"$tmp/venv/bin/python" -m distilllab evaluate --format json > evaluate.json 2>evaluate.err & evaluate_pid=$!
"$tmp/venv/bin/python" -m distilllab gates --min-teacher-acc .8 --min-distill-gain .04 > gates.txt 2>gates.err & gates_pid=$!
wait "$demo_pid"
wait "$evaluate_pid"
wait "$gates_pid"
"$tmp/venv/bin/python" - <<'PY'
import gzip,importlib,json,pathlib,math
for name in ['cli','data','distill','eval','mlp','step','step_fixtures']:importlib.import_module('distilllab.'+name)
import distilllab
from distilllab.step import audit
root=pathlib.Path(distilllab.__file__).parent/'step_data'
def equal(a,b):
 if type(a) in (int,float):
  assert type(b) in (int,float) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:equal(a[k],b[k])
 elif isinstance(a,list):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b
f=json.loads((root/'frozen.json').read_text());r=json.loads(pathlib.Path('regenerated/frozen.json').read_text());equal(f,r)
w=json.loads(gzip.decompress((root/'witnesses.json.gz').read_bytes()));v=json.loads(gzip.decompress(pathlib.Path('regenerated/witnesses.json.gz').read_bytes()));equal(w,v);assert len(w)==36
for case in w:
 m=case['input'];p=f['controls'][m['control']] if 'control'in m else {'model':f['models'][m['model']],'row':f['splits']['student_train'][m['row']],'evaluation':f['splits']}
 equal(audit(p,m['temperature'],m['alpha'],m['step']),case['report'])
pathlib.Path('input.json').write_text(json.dumps(f['controls']['overshoot']))
print('7 modules import; original teacher and six student fits, all logits/data/report and 36 witnesses reproduce outside checkout')
PY
"$tmp/venv/bin/python" -m distilllab step-audit input.json --temperature 1 --alpha .5 --step 2 > audit.json
"$tmp/venv/bin/python" - <<'PY'
import json
assert json.load(open('audit.json'))['actual_loss_delta']>2
r=json.load(open('evaluate.json'));d=json.load(open('demo.json'));assert r==d['original_report'];assert d['witnesses']==36
assert 'all gates passed' in open('gates.txt').read()
print('All four CLI commands execute; original report and entered overshoot checked')
PY

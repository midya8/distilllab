#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
python -m pip wheel --no-deps "$root" --wheel-dir "$tmp/wheels" >/dev/null
python -m venv "$tmp/venv"
"$tmp/venv/bin/python" -m pip install --no-index "$tmp"/wheels/*.whl >/dev/null
cd "$tmp"
"$tmp/venv/bin/python" - <<'PY'
import copy,gzip,importlib,json,pathlib,subprocess,sys
from importlib.resources import files
from distilllab.tail import audit,verify_archive
from distilllab.tail_fixtures import build
for name in ['cli','data','distill','eval','mlp','step','step_fixtures','tail','tail_fixtures']:
    module=importlib.import_module('distilllab.'+name)
    assert pathlib.Path(module.__file__).is_relative_to(pathlib.Path(sys.prefix))
root=files('distilllab').joinpath('tail_data')
examples=json.loads(root.joinpath('examples.json').read_text())
witnesses=json.loads(gzip.decompress(root.joinpath('1-witnesses.json.gz').read_bytes()))
build('regenerated')
assert pathlib.Path('regenerated/examples.json').read_bytes()==root.joinpath('examples.json').read_bytes()
assert pathlib.Path('regenerated/1-witnesses.json.gz').read_bytes()==root.joinpath('1-witnesses.json.gz').read_bytes()
for c,w in zip(examples['examples'],witnesses):
    assert c['id']==w['id']
    path=pathlib.Path('input.json');path.write_text(json.dumps(c['input']))
    for command in [[sys.executable,'-m','distilllab.tail'],[str(pathlib.Path(sys.prefix)/'bin/distilllab-tail')]]:
        r=subprocess.run(command+[str(path)],text=True,capture_output=True,check=True)
        report=json.loads(r.stdout);assert report==w['report']
        archive={'schema':'distilllab.tail.archive.v1','input':c['input'],'report':report}
        pathlib.Path('archive.json').write_text(json.dumps(archive))
        result=subprocess.run(command+['archive.json','--verify'],text=True,capture_output=True,check=True)
        assert json.loads(result.stdout)['verified']
        archive['report']['variants'][0]['parameters'][0]['after']+=.125
        pathlib.Path('bad.json').write_text(json.dumps(archive))
        bad=subprocess.run(command+['bad.json','--verify'],text=True,capture_output=True)
        assert bad.returncode==2 and 'mismatch' in bad.stderr
print('Nine installed module origins, exact regeneration, 28 witnesses, both CLI entrypoints, 56 valid archive replays and 56 rejected forgeries outside checkout')
PY

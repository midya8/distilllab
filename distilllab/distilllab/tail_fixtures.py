"""Inputs copied from original packaged states, plus explicit arithmetic controls."""
import copy
import gzip
import json
from importlib.resources import files
from pathlib import Path
from .tail import audit,INPUT_SCHEMA


def examples():
    frozen=json.loads(files('distilllab').joinpath('step_data/frozen.json').read_text())
    cases=[]
    for name,model in frozen['models'].items():
        for index,t,alpha,mix,shift in [(0,4,.95,.25,1),(39,1,.5,0,3),(79,8,1,1,0)]:
            cases.append({'id':f'{name}-{index:02d}','title':f'Original {name}, training row {index}',
                          'input':{'schema':INPUT_SCHEMA,'model':copy.deepcopy(model),
                                   'row':copy.deepcopy(frozen['splits']['student_train'][index]),
                                   'temperature':t,'alpha':alpha,'step':.02,'tail_mix':mix,'tail_shift':shift}})
    base={'schema':INPUT_SCHEMA,'model':{'W1':[[.3,-.2],[-.4,.5]],'b1':[.1,-.2],
          'W2':[[.2,-.6],[.4,.1],[-.5,.8]],'b2':[.2,-.3,.1]},
          'row':{'x':[.7,-.4],'label':0,'teacher_logits':[2,1,-2]},
          'temperature':1,'alpha':.95,'step':.1,'tail_mix':.35,'tail_shift':1}
    def add(name,p):cases.append({'id':name,'title':name.replace('-',' ').capitalize(),'input':p})
    add('confidence-matched-tail',copy.deepcopy(base))
    q=copy.deepcopy(base);q['row']['teacher_logits']=[-2,3,1];add('teacher-wrong-about-label',q)
    q=copy.deepcopy(base);q['row']['teacher_logits']=[10000,-10000,-9999];q['temperature']=.25;add('true-excluded-tail-saturation',q)
    q=copy.deepcopy(base);q['alpha']=0;add('hard-only-interventions-inert',q)
    q=copy.deepcopy(base);q['step']=0;add('zero-step-is-not-training',q)
    q=copy.deepcopy(base);q['model']['W2']=q['model']['W2'][:2];q['model']['b2']=q['model']['b2'][:2];q['row']['teacher_logits']=[2,-2];q['tail_shift']=0;add('two-classes-singleton-tail',q)
    p=copy.deepcopy(frozen['controls']['overshoot']);p.pop('evaluation')
    p.update(schema=INPUT_SCHEMA,temperature=1,alpha=.5,step=2,tail_mix=.2,tail_shift=0)
    add('finite-step-can-overshoot',p)
    return {'schema':'distilllab.tail.examples.v1','examples':cases,
            'scope':'21 original archived model/training-row controls plus seven explicitly synthetic arithmetic controls. No new trained arm, test-label gradient or refit.'}


def build(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    data=examples();reports=[{'id':c['id'],'report':audit(c['input'])} for c in data['examples']]
    (root/'examples.json').write_text(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n')
    (root/'witnesses.json.gz').write_bytes(gzip.compress((json.dumps(reports,separators=(',',':'),allow_nan=False)+'\n').encode(),mtime=0))
    print(json.dumps({'examples':len(reports),'variants':sum(len(r['report']['variants']) for r in reports),
                      'parameter_positions':sum(len(v['parameters']) for r in reports for v in r['report']['variants'])}))
    return data


if __name__=='__main__':
    import sys
    build(sys.argv[1])

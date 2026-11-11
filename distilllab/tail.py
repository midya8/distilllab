"""Exact target/rest KL chain rule and matched-confidence teacher interventions.

A separate audit, not a change to the historical trainer or a DKD training run.
All teacher quantities are fixed when differentiating student parameters.
"""
from __future__ import annotations
import copy
import json
import math
from .step import validate as _validate_model, forward, distribution

INPUT_SCHEMA = 'distilllab.tail.input.v1'
REPORT_SCHEMA = 'distilllab.tail.report.v1'
KEYS = ('W1', 'b1', 'W2', 'b2')


def _finite(x, name, lo, hi):
    if type(x) not in (int, float) or not lo <= x <= hi or not math.isfinite(x):
        raise ValueError(name + ' outside finite bounds')
    return float(x)


def validate(p):
    required = {'schema', 'model', 'row', 'temperature', 'alpha', 'step', 'tail_mix', 'tail_shift'}
    if type(p) is not dict or set(p) != required or p['schema'] != INPUT_SCHEMA:
        raise ValueError('expected exact ' + INPUT_SCHEMA + ' fields')
    try:
        d, h, k = _validate_model({'model': p['model'], 'row': p['row'], 'evaluation': {'row': [p['row']]}})
    except OverflowError as exc:
        raise ValueError('model or row number outside finite bounds') from exc
    _finite(p['temperature'], 'temperature', .25, 20)
    _finite(p['alpha'], 'alpha', 0, 1)
    _finite(p['step'], 'step', 0, 2)
    _finite(p['tail_mix'], 'tail_mix', 0, 1)
    if type(p['tail_shift']) is not int or not 0 <= p['tail_shift'] < k-1:
        raise ValueError('tail_shift must be an integer in [0, classes-2]')
    return d, h, k


def _lse(z):
    m = max(z)
    return m + math.log(math.fsum(math.exp(v-m) for v in z))


def _partition(z, y, t):
    # True subset normalization. No 1 - rounded_target and no -1000 mask.
    a = [v/t for v in z]
    indices = [i for i in range(len(a)) if i != y]
    tail_lse = _lse([a[i] for i in indices])
    top = max(a[y], tail_lse)
    norm = math.log(math.exp(a[y]-top) + math.exp(tail_lse-top))
    binary_log = [(a[y]-top)-norm, (tail_lse-top)-norm]
    tail_log = [a[i]-tail_lse for i in indices]
    return {'indices': indices, 'binary_log': binary_log,
            'binary': [math.exp(v) for v in binary_log],
            'tail_log': tail_log, 'tail': [math.exp(v) for v in tail_log]}


def _kl(q, lq, lp):
    return math.fsum(x*(a-b) for x, a, b in zip(q, lq, lp))


def _teacher(part, y, k, mix, shift):
    n = k-1
    rotated = part['tail'][shift:] + part['tail'][:shift]
    rotated_logs = part['tail_log'][shift:] + part['tail_log'][:shift]
    # Log mixture retains small positive tails even when binary masses round.
    if n == 1:
        r, lr = [1.0], [0.0]
    elif mix == 0:
        r, lr = rotated, rotated_logs
    elif mix == 1:
        r, lr = [1/n]*n, [-math.log(n)]*n
    else:
        lr = [_lse([math.log1p(-mix)+v, math.log(mix)-math.log(n)]) for v in rotated_logs]
        r = [math.exp(v) for v in lr]
    qt, qm = part['binary']
    lqt, lqm = part['binary_log']
    probs, logs = [0.0]*k, [0.0]*k
    probs[y], logs[y] = qt, lqt
    for i, v, lv in zip(part['indices'], r, lr):
        probs[i], logs[i] = qm*v, lqm+lv
    return {'probabilities': probs, 'log_probabilities': logs, 'binary': [qt,qm],
            'binary_log': [lqt,lqm], 'tail': r, 'tail_log': lr}


def terms(z, teacher, label, temperature, alpha):
    """Return loss and derivative components for one fixed teacher distribution."""
    p = _partition(z, label, temperature)
    p1, lp1 = distribution(z)
    pt, pm = p['binary']; qt, qm = teacher['binary']
    k = len(z)
    probabilities, logs = [0.0]*k, [0.0]*k
    probabilities[label], logs[label] = pt, p['binary_log'][0]
    for i, v, lv in zip(p['indices'], p['tail'], p['tail_log']):
        probabilities[i], logs[i] = pm*v, p['binary_log'][1]+lv
    binary_kl = _kl(teacher['binary'], teacher['binary_log'], p['binary_log'])
    tail_kl = _kl(teacher['tail'], teacher['tail_log'], p['tail_log'])
    direct_kl = _kl(teacher['probabilities'], teacher['log_probabilities'], logs)
    entropy = -math.fsum(v*lv for v,lv in zip(teacher['probabilities'], teacher['log_probabilities']))
    ce = -math.fsum(v*lv for v,lv in zip(teacher['probabilities'], logs))
    factor = alpha*temperature
    # Product-difference binary derivative avoids rounded (pt-qt)=0 when
    # both displayed target probabilities are 1 but tails differ.
    btarget = factor*(pt*qm-qt*pm)
    hard = [(1-alpha)*(v if i != label else -math.fsum(p1[j] for j in range(k) if j != label)) for i,v in enumerate(p1)]
    binary, conditional, tail = [0.0]*k, [0.0]*k, [0.0]*k
    binary[label] = btarget
    for i, s, r in zip(p['indices'], p['tail'], teacher['tail']):
        binary[i] = -btarget*s
        conditional[i] = factor*(s-r)
        tail[i] = qm*conditional[i]
    total = [math.fsum(v) for v in zip(hard, binary, tail)]
    # Independent full-class product differences check the decomposition.
    direct = [hard[i] + factor*math.fsum(probabilities[i]*teacher['probabilities'][j]-teacher['probabilities'][i]*probabilities[j] for j in range(k) if j != i) for i in range(k)]
    classes = [{'class':i, 'target':i==label, 'student':probabilities[i], 'student_log':logs[i],
                'teacher':teacher['probabilities'][i], 'teacher_log':teacher['log_probabilities'][i],
                'student_1':p1[i], 'student_tail':0.0 if i==label else p['tail'][p['indices'].index(i)],
                'teacher_tail':0.0 if i==label else teacher['tail'][p['indices'].index(i)],
                'hard_gradient':hard[i], 'binary_gradient':binary[i],
                'conditional_gradient':conditional[i], 'coupled_tail_gradient':tail[i],
                'total_gradient':total[i], 'direct_gradient':direct[i],
                'gradient_residual':total[i]-direct[i]} for i in range(k)]
    scale = alpha*temperature**2
    hard_ce = -lp1[label]
    split_kl = binary_kl+qm*tail_kl
    return {'target_mass':qt, 'non_target_mass':qm, 'student_target_mass':pt,
            'student_non_target_mass':pm, 'teacher_mass_residual':math.fsum(teacher['probabilities'])-1,
            'binary_kl':binary_kl, 'conditional_kl':tail_kl, 'coupled_tail_kl':qm*tail_kl,
            'direct_kl':direct_kl, 'split_kl':split_kl, 'kl_residual':split_kl-direct_kl,
            'hard_ce':hard_ce, 'soft_ce':ce, 'teacher_entropy':entropy,
            'ce_entropy_residual':ce-entropy-direct_kl,
            'weighted_hard':(1-alpha)*hard_ce, 'weighted_binary':scale*binary_kl,
            'weighted_coupled_tail':scale*qm*tail_kl,
            'kl_objective':(1-alpha)*hard_ce+scale*split_kl,
            'original_ce_objective':(1-alpha)*hard_ce+scale*ce,
            'teacher_only_offset':scale*entropy,
            'prediction':z.index(max(z)), 'classes':classes}


def _backprop(model, x, hidden, dl):
    hidden_g = [math.fsum(dl[j]*model['W2'][j][i] for j in range(len(dl)))*(1-hidden[i]**2) for i in range(len(hidden))]
    return {'W1':[[v*a for a in x] for v in hidden_g], 'b1':hidden_g,
            'W2':[[v*a for a in hidden] for v in dl], 'b2':list(dl)}


def _positions(m):
    for key in KEYS:
        for i,v in enumerate(m[key]):
            for j,a in enumerate(v if key in ('W1','W2') else [v]):
                yield key,i,j,a,f'{key}[{i}]'+(f'[{j}]' if key in ('W1','W2') else '')


def _get(m,key,i,j):
    return m[key][i][j] if key in ('W1','W2') else m[key][i]


def _set(m,key,i,j,value):
    if key in ('W1','W2'):m[key][i][j]=value
    else:m[key][i]=value


def audit(payload):
    d,h,k = validate(payload)
    m,row = payload['model'],payload['row']
    t,a,eta = map(float,(payload['temperature'],payload['alpha'],payload['step']))
    hidden,z = forward(m,row['x'])
    y = row['label']
    partition = _partition(row['teacher_logits'],y,t)
    configs = [('original',0.0,0),('uniform_tail',1.0,0),('rotated_tail',0.0,1 if k>2 else 0),
               ('configured',float(payload['tail_mix']),payload['tail_shift'])]
    reports=[]
    for name,mix,shift in configs:
        teacher = _teacher(partition,y,k,mix,shift)
        before = terms(z,teacher,y,t,a)
        fields = {'hard':'hard_gradient','binary':'binary_gradient','coupled_tail':'coupled_tail_gradient','total':'total_gradient','direct':'direct_gradient'}
        grads = {key:_backprop(m,row['x'],hidden,[v[field] for v in before['classes']]) for key,field in fields.items()}
        updated=copy.deepcopy(m); rows=[]
        for key,i,j,value,pname in _positions(m):
            g={f:_get(v,key,i,j) for f,v in grads.items()}
            nxt=value-eta*g['total'];_set(updated,key,i,j,nxt)
            rows.append({'name':pname,'before':value,'hard_gradient':g['hard'],
                         'binary_gradient':g['binary'],'coupled_tail_gradient':g['coupled_tail'],
                         'gradient':g['total'],'direct_gradient':g['direct'],
                         'component_residual':math.fsum([g['hard'],g['binary'],g['coupled_tail']])-g['total'],
                         'direct_residual':g['total']-g['direct'],'delta':-eta*g['total'],'after':nxt})
        ha,za=forward(updated,row['x'])
        after=terms(za,teacher,y,t,a)
        original_teacher=_teacher(partition,y,k,0.0,0)
        original_after=terms(za,original_teacher,y,t,a)
        norm2=math.fsum(r['gradient']**2 for r in rows)
        reports.append({'name':name,'tail_mix':mix,'tail_shift':shift,
                        'teacher':teacher,'before':before,'after':after,
                        'original_teacher_after':original_after,
                        'logits_after':za,'hidden_after':ha,'parameters':rows,'updated_model':updated,
                        'gradient_norm':math.sqrt(norm2),'first_order_kl_delta':-eta*norm2,
                        'actual_kl_delta':after['kl_objective']-before['kl_objective'],
                        'actual_original_kl_delta':original_after['kl_objective']-terms(z,original_teacher,y,t,a)['kl_objective'],
                        'max_parameter_direct_residual':max(abs(r['direct_residual']) for r in rows),
                        'max_parameter_component_residual':max(abs(r['component_residual']) for r in rows)})
    baseline=reports[0]
    for r in reports:
        r['comparison']={'binary_loss_delta':r['before']['binary_kl']-baseline['before']['binary_kl'],
                         'entropy_delta':r['before']['teacher_entropy']-baseline['before']['teacher_entropy'],
                         'gradient_delta_norm':math.sqrt(math.fsum((p['gradient']-q['gradient'])**2 for p,q in zip(r['parameters'],baseline['parameters']))),
                         'target_mass_delta':r['before']['target_mass']-baseline['before']['target_mass'],
                         'non_target_mass_delta':r['before']['non_target_mass']-baseline['before']['non_target_mass']}
    result={'schema':REPORT_SCHEMA,'controls':{key:payload[key] for key in ('temperature','alpha','step','tail_mix','tail_shift')},
            'dimensions':{'inputs':d,'hidden':h,'classes':k,'parameters':len(baseline['parameters'])},
            'label':y,'hidden_before':hidden,'logits_before':z,'variants':reports,
            'scope':'One fixed-teacher training-row gradient and one simultaneous step per variant. KL chain rule, not a new method, DKD training run or causal explanation of original gain. Tail interventions preserve target/rest mass at the selected temperature, not generalization, label semantics or T=1 confidence.'}
    json.dumps(result,allow_nan=False)
    return result


def load_input(path, limit=1500000):
    from pathlib import Path
    text=Path(path).read_text()
    if len(text)>limit:raise ValueError('JSON exceeds bounded size')
    def pairs(items):
        out={}
        for k,v in items:
            if k in out:raise ValueError('duplicate JSON key: '+k)
            out[k]=v
        return out
    return json.loads(text,object_pairs_hook=pairs,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def verify_archive(p):
    """Replay every field of a downloaded archive, not just its summary."""
    if type(p) is not dict or set(p) != {'schema','input','report'} or p['schema'] != 'distilllab.tail.archive.v1':
        raise ValueError('invalid tail archive')
    expected=audit(p['input'])
    def check(actual,expected,path):
        if type(expected) in (int,float):
            if type(actual) not in (int,float) or not math.isfinite(actual) or abs(actual-expected)>2e-9+2e-9*max(abs(actual),abs(expected)):
                raise ValueError('numeric mismatch at '+path)
        elif type(actual) != type(expected):raise ValueError('type mismatch at '+path)
        elif type(expected) is dict:
            if set(actual)!=set(expected):raise ValueError('field mismatch at '+path)
            for k in expected:check(actual[k],expected[k],path+'.'+k)
        elif type(expected) is list:
            if len(actual)!=len(expected):raise ValueError('length mismatch at '+path)
            for i,(a,b) in enumerate(zip(actual,expected)):check(a,b,path+'['+str(i)+']')
        elif actual!=expected:raise ValueError('value mismatch at '+path)
    check(p['report'],expected,'report')
    return {'verified':True,'schema':REPORT_SCHEMA,'variants':len(expected['variants']),
            'parameter_positions':sum(len(v['parameters']) for v in expected['variants'])}


def main(argv=None):
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input')
    ap.add_argument('--verify',action='store_true',help='replay all fields of a browser archive')
    args=ap.parse_args(argv)
    try:
        payload=load_input(args.input,20000000 if args.verify else 1500000)
        print(json.dumps(verify_archive(payload) if args.verify else audit(payload),allow_nan=False,indent=2))
    except (ValueError,TypeError,KeyError,IndexError,OSError) as exc:ap.error(str(exc))


if __name__=='__main__':main()

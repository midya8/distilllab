import copy
import json
import math
import random
import tempfile
import unittest
from decimal import Decimal,localcontext
from pathlib import Path
from distilllab.tail import audit,terms,validate,INPUT_SCHEMA,_positions,_set
from distilllab.step import audit as old_audit
from distilllab.mlp import MLP


def fixture(seed=1,k=4):
    rng=random.Random(seed)
    d,h=3,2
    m=MLP(d,h,k,rng)
    return {'schema':INPUT_SCHEMA,'model':{f:getattr(m,f) for f in ('W1','b1','W2','b2')},
            'row':{'x':[rng.uniform(-1,1) for _ in range(d)],'label':seed%k,'teacher_logits':[rng.uniform(-3,3) for _ in range(k)]},
            'temperature':2.0,'alpha':.7,'step':.1,'tail_mix':.2,'tail_shift':1 if k>2 else 0}


def reference(p,variant,component):
    """Independent direct scalar objective, no audit/partition/terms/backprop."""
    m,row=p['model'],p['row'];T=p['temperature'];a=p['alpha'];y=row['label']
    h=[math.tanh(sum(w*x for w,x in zip(ws,row['x']))+b) for ws,b in zip(m['W1'],m['b1'])]
    z=[sum(w*x for w,x in zip(ws,h))+b for ws,b in zip(m['W2'],m['b2'])]
    def logs(v):
        mm=max(v);s=sum(math.exp(x-mm) for x in v);return [x-mm-math.log(s) for x in v]
    lp1=logs(z);lp=logs([v/T for v in z]);lq=logs([v/T for v in row['teacher_logits']]);q=list(map(math.exp,lq));pr=list(map(math.exp,lp));ids=[i for i in range(len(z)) if i!=y]
    # Independent reference uses sums, deliberately not the production binary LSE.
    qm=sum(q[i] for i in ids);pm=sum(pr[i] for i in ids)
    tail=[q[i]/qm for i in ids]
    mix,shift={'original':(0,0),'uniform_tail':(1,0),'rotated_tail':(0,1 if len(z)>2 else 0),'configured':(p['tail_mix'],p['tail_shift'])}[variant]
    tail=tail[shift:]+tail[:shift];tail=[(1-mix)*v+mix/len(ids) for v in tail]
    binary=q[y]*math.log(q[y]/pr[y])+qm*math.log(qm/pm)
    cond=sum(v*math.log(v/(pr[i]/pm)) for v,i in zip(tail,ids))
    hard=(1-a)*(-lp1[y]);b=a*T*T*binary;c=a*T*T*qm*cond
    return {'hard':hard,'binary':b,'coupled_tail':c,'total':hard+b+c}[component]


class TailAuditTests(unittest.TestCase):
    def test_all_parameter_component_gradients_independent_finite_differences(self):
        checks=0
        for seed in range(8):
            p=fixture(seed,2+seed%4);p['temperature']=[.5,1,4,8][seed%4];p['alpha']=[0,.3,.95,1][seed%4]
            report=audit(p)
            for variant in report['variants']:
                for comp in ('hard','binary','coupled_tail','total'):
                    for (key,i,j,value,name),actual in zip(_positions(p['model']),variant['parameters']):
                        eps=1e-5
                        q=copy.deepcopy(p);_set(q['model'],key,i,j,value+eps);plus=reference(q,variant['name'],comp)
                        _set(q['model'],key,i,j,value-eps);minus=reference(q,variant['name'],comp)
                        field='gradient' if comp=='total' else comp+'_gradient'
                        self.assertAlmostEqual((plus-minus)/(2*eps),actual[field],delta=2e-8,msg=str((seed,variant['name'],comp,name)))
                        checks+=1
        self.assertEqual(checks,2368)

    def test_decimal_logits_chain_rule_and_derivatives(self):
        with localcontext() as ctx:
            ctx.prec=80
            for seed in range(40):
                p=fixture(seed,2+seed%5);p['temperature']=[.25,1,4,20][seed%4]
                r=audit(p);z=r['logits_before'];y=p['row']['label'];t=Decimal(str(p['temperature']));a=Decimal(str(p['alpha']))
                def probs(z,temp):
                    e=[(Decimal(str(v))/temp).exp() for v in z];s=sum(e);return [v/s for v in e]
                pr=probs(z,t);q=probs(p['row']['teacher_logits'],t);p1=probs(z,Decimal(1));qm=sum(q[i] for i in range(len(q)) if i!=y);pm=sum(pr[i] for i in range(len(q)) if i!=y)
                binary=q[y]*(q[y]/pr[y]).ln()+qm*(qm/pm).ln()
                tail=sum(q[i]/qm*((q[i]/qm)/(pr[i]/pm)).ln() for i in range(len(q)) if i!=y)
                direct=sum(v*(v/w).ln() for v,w in zip(q,pr))
                b=r['variants'][0]['before']
                for key,expected in [('binary_kl',binary),('conditional_kl',tail),('direct_kl',direct)]:self.assertAlmostEqual(b[key],float(expected),delta=2e-11)
                for i,c in enumerate(b['classes']):
                    expected=(1-a)*(p1[i]-(1 if i==y else 0))+a*t*(pr[i]-q[i])
                    self.assertAlmostEqual(c['total_gradient'],float(expected),delta=1e-12)

    def test_old_audit_every_gradient_same_input(self):
        for seed in range(30):
            p=fixture(seed);r=audit(p)['variants'][0]
            old=old_audit({'model':p['model'],'row':p['row'],'evaluation':{'one':[p['row']]}},p['temperature'],p['alpha'],p['step'])
            for a,b in zip(r['parameters'],old['parameters']):self.assertAlmostEqual(a['gradient'],b['gradient'],delta=1e-13)
            self.assertAlmostEqual(r['before']['original_ce_objective'],old['before']['loss'],delta=1e-12)
            self.assertAlmostEqual(r['after']['original_ce_objective'],old['after']['loss'],delta=1e-12)

    def test_invariants_all_variants(self):
        for seed in range(60):
            p=fixture(seed,2+seed%7);r=audit(p)
            for v in r['variants']:
                self.assertEqual(v['comparison']['target_mass_delta'],0)
                self.assertEqual(v['comparison']['non_target_mass_delta'],0)
                self.assertEqual(v['comparison']['binary_loss_delta'],0)
                for state in ('before','after','original_teacher_after'):
                    b=v[state];self.assertLess(abs(b['kl_residual']),1e-11);self.assertLess(abs(b['ce_entropy_residual']),1e-11)
                    self.assertLess(abs(sum(c['total_gradient'] for c in b['classes'])),1e-12)
                self.assertLess(v['max_parameter_direct_residual'],1e-12)
                self.assertLess(v['max_parameter_component_residual'],1e-12)
                self.assertLess(abs(v['before']['original_ce_objective']-v['before']['kl_objective']-v['before']['teacher_only_offset']),1e-11)
            self.assertAlmostEqual(r['variants'][2]['comparison']['entropy_delta'],0,delta=1e-13)
            if len(p['model']['b2'])==2:
                self.assertEqual(r['variants'][0]['before']['conditional_kl'],0)
                self.assertEqual(r['variants'][2]['comparison']['gradient_delta_norm'],0)

    def test_saturation_true_subset_not_subtract_1000(self):
        p=fixture(0,3);p['model']['W2']=[[0,0]]*3;p['model']['b2']=[100,-100,-99];p['row']['label']=0;p['row']['teacher_logits']=[10000,-10000,-9999];p['temperature']=.25
        r=audit(p);b=r['variants'][0]['before']
        self.assertEqual(b['target_mass'],1)
        self.assertEqual(b['non_target_mass'],0)
        self.assertTrue(math.isfinite(b['conditional_kl']))
        # Excluded target is truly absent from normalization despite being 80k
        # temperature-scaled units above the tail, unlike a finite -1000 mask.
        q=r['variants'][0]['teacher']['tail']
        self.assertAlmostEqual(q[1],1/(1+math.exp(-4)),delta=1e-11)
        json.dumps(r,allow_nan=False)
        p['row']['teacher_logits']=[100,-100,-99];p['temperature']=1
        b=audit(p)['variants'][0]['before']
        self.assertEqual(b['target_mass'],1)
        self.assertGreater(b['non_target_mass'],0)

    def test_matched_confidence_rotated_tail_changes_gradient(self):
        p=fixture(3,4);r=audit(p)
        self.assertGreater(r['variants'][2]['comparison']['gradient_delta_norm'],.01)
        self.assertAlmostEqual(r['variants'][2]['comparison']['entropy_delta'],0,delta=1e-13)
        self.assertEqual(r['variants'][2]['comparison']['binary_loss_delta'],0)

    def test_alpha_zero_step_zero_and_input_unmodified(self):
        p=fixture();before=copy.deepcopy(p);audit(p);self.assertEqual(p,before)
        p['alpha']=0;p['step']=0;r=audit(p)
        for v in r['variants']:
            self.assertEqual(v['updated_model'],p['model']);self.assertEqual(v['comparison']['gradient_delta_norm'],0)
            self.assertEqual(v['actual_kl_delta'],0)

    def test_validation_and_json(self):
        p=fixture()
        bad=[]
        for key,value in [('schema','bad'),('temperature',0),('alpha',1.1),('step',-1),('tail_mix',float('nan')),('tail_shift',True),('tail_shift',3),('temperature',10**400)]:
            q=copy.deepcopy(p);q[key]=value;bad.append(q)
        for key in p:
            q=copy.deepcopy(p);del q[key];bad.append(q)
        q=copy.deepcopy(p);q['extra']=0;bad.append(q)
        q=copy.deepcopy(p);q['row']['teacher_logits'][0]=float('inf');bad.append(q)
        q=copy.deepcopy(p);q['model']['W1'][0][0]=10**400;bad.append(q)
        for q in bad:
            with self.assertRaises(ValueError):audit(q)
        from distilllab.tail import load_input
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'a.json'
            for s in ['{"x":1,"x":2}', '{"x":NaN}', ' '*1500001]:
                path.write_text(s)
                with self.assertRaises(ValueError):load_input(path)


class TailArchiveTests(unittest.TestCase):
    def test_original_all_seven_models_all_80_training_rows(self):
        from importlib.resources import files
        from distilllab.mlp import softmax
        frozen=json.loads(files('distilllab').joinpath('step_data/frozen.json').read_text())
        positions=0
        for name,model in frozen['models'].items():
            net=object.__new__(MLP)
            for key,value in model.items():setattr(net,key,value)
            net.dim,net.hidden,net.n_classes=32,6,8
            for row in frozen['splits']['student_train']:
                p={'schema':INPUT_SCHEMA,'model':model,'row':row,'temperature':4,'alpha':.95,'step':.02,'tail_mix':.3,'tail_shift':1}
                actual=audit(p)['variants'][0]
                z=net.logits(row['x']);s=softmax(z,4);q=softmax(row['teacher_logits'],4);hard=softmax(z)
                dl=[.95*4*(a-b)+.05*(c-(1 if i==row['label'] else 0)) for i,(a,b,c) in enumerate(zip(s,q,hard))]
                gs=net.backprop(row['x'],dl);flat=[v for r in gs[0] for v in r]+gs[1]+[v for r in gs[2] for v in r]+gs[3]
                for r,v in zip(actual['parameters'],flat):self.assertAlmostEqual(r['gradient'],v,delta=2e-12);positions+=1
        self.assertEqual(positions,142240)

    def test_archive_replay_all_witnesses_and_single_leaf_forgeries(self):
        from distilllab.tail import verify_archive
        from distilllab.tail_fixtures import examples,build
        from importlib.resources import files
        with tempfile.TemporaryDirectory() as directory:
            build(directory)
            packed=files('distilllab').joinpath('tail_data')
            generated=Path(directory,'1-witnesses.json.gz').read_bytes()
            self.assertEqual(generated[:10],bytes([31,139,8,0,0,0,0,0,2,255]))
            self.assertEqual(generated,packed.joinpath('1-witnesses.json.gz').read_bytes())
            self.assertEqual(Path(directory,'examples.json').read_bytes(),packed.joinpath('examples.json').read_bytes())
        checked=0
        for c in examples()['examples']:
            p={'schema':'distilllab.tail.archive.v1','input':c['input'],'report':audit(c['input'])}
            self.assertTrue(verify_archive(p)['verified'])
            def leaves(x,path=()):
                if type(x) is dict:
                    for k,v in x.items():yield from leaves(v,path+(k,))
                elif type(x) is list:
                    for i,v in enumerate(x):yield from leaves(v,path+(i,))
                elif type(x) in (int,float):yield path
            paths=list(leaves(p['report']))
            for path in paths[::max(1,len(paths)//12)][:12]:
                q=copy.deepcopy(p);obj=q['report']
                for k in path[:-1]:obj=obj[k]
                obj[path[-1]]+=.125
                with self.assertRaises(ValueError):verify_archive(q)
                checked+=1
        self.assertEqual(checked,336)

    def test_original_teacher_comparison_uses_original_baseline(self):
        p=fixture(3);r=audit(p);base=r['variants'][0]['before']['kl_objective']
        for v in r['variants']:
            self.assertAlmostEqual(v['actual_original_kl_delta'],v['original_teacher_after']['kl_objective']-base,delta=1e-12)

    def test_maximum_shape_archive_replay(self):
        from distilllab.tail import verify_archive
        p={'schema':INPUT_SCHEMA,'model':{'W1':[[.01]*64 for _ in range(16)],'b1':[.1]*16,'W2':[[.01]*16 for _ in range(16)],'b2':[.1]*16},'row':{'x':[.1]*64,'label':15,'teacher_logits':list(range(16))},'temperature':20,'alpha':.95,'step':2,'tail_mix':.3,'tail_shift':14}
        report=audit(p);self.assertEqual(report['dimensions']['parameters'],1312)
        envelope={'schema':'distilllab.tail.archive.v1','input':p,'report':report};self.assertTrue(verify_archive(envelope)['verified'])

if __name__=='__main__':unittest.main()

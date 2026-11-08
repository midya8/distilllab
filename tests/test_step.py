import gzip
import copy
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from distilllab.step import audit, forward, loss_terms, validate
from distilllab.step_fixtures import OVERSHOOT
from distilllab.mlp import MLP
from distilllab.distill import distill_grads,soft_targets

ROOT=Path(__file__).resolve().parents[1]

class TestStep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads((ROOT/'distilllab/step_data/frozen.json').read_text())
        cls.manifest=json.loads(gzip.decompress((ROOT/'distilllab/step_data/witnesses.json.gz').read_bytes()))
    def payload(self,name='distilled',row=0):
        return {'model':copy.deepcopy(self.data['models'][name]),'row':copy.deepcopy(self.data['splits']['student_train'][row]),'evaluation':copy.deepcopy(self.data['splits'])}
    def assert_close(self,a,b,path=''):
        if type(a) in (int,float):
            self.assertEqual(type(a),type(b),path) if type(a) is int else self.assertIsInstance(b,(int,float),path)
            self.assertTrue(math.isclose(a,b,abs_tol=1e-12,rel_tol=1e-12),(path,a,b))
        elif isinstance(a,dict):
            self.assertEqual(set(a),set(b),path)
            for k in a:self.assert_close(a[k],b[k],path+'.'+k)
        elif isinstance(a,list):
            self.assertEqual(len(a),len(b),path)
            for i,(x,y) in enumerate(zip(a,b)):self.assert_close(x,y,path+'.'+str(i))
        else:self.assertEqual(a,b,path)
    def test_all_archived_witnesses(self):
        for w in self.manifest:
            m=w['input']
            p=self.data['controls'][m['control']] if 'control'in m else self.payload(m['model'],m['row'])
            self.assert_close(w['report'],audit(p,m['temperature'],m['alpha'],m['step']))
    def test_original_gradients_and_simultaneous_updates(self):
        for name in self.data['models']:
            for index in (0,17,79):
                p=self.payload(name,index);r=audit(p,4,.95,.02);m=p['model'];d,h,k=validate(p)
                net=MLP(d,h,k,random.Random(0))
                for key,value in m.items():setattr(net,key,copy.deepcopy(value))
                t,_=__import__('distilllab.step',fromlist=['distribution']).distribution(p['row']['teacher_logits'],4)
                g=distill_grads(net,p['row']['x'],t,p['row']['label'],4,.95)
                net.apply_grads(g,.02)
                for key in m:
                    self.assert_close(r['updated_model'][key],getattr(net,key))
    def test_all_parameter_finite_differences(self):
        for p,t,a in [(self.payload('initial',3),4,.95),(copy.deepcopy(OVERSHOOT),1,.5)]:
            r=audit(p,t,a,.02);flat=r['parameters'];eps=1e-5
            for param in flat:
                import re
                key=param['name'].split('[')[0];indices=list(map(int,re.findall(r'\[(\d+)\]',param['name'])))
                def change(delta):
                    m=copy.deepcopy(p['model']);arr=m[key]
                    if len(indices)==2:arr[indices[0]][indices[1]]+=delta
                    else:arr[indices[0]]+=delta
                    _,z=forward(m,p['row']['x'])
                    return loss_terms(z,p['row']['teacher_logits'],p['row']['label'],t,a)['loss']
                numeric=(change(eps)-change(-eps))/(2*eps)
                self.assertTrue(math.isclose(numeric,param['gradient'],abs_tol=2e-8,rel_tol=2e-6),(param['name'],numeric,param['gradient']))
    def test_independent_decimal_logit_objectives(self):
        rng=random.Random(131)
        with localcontext() as ctx:
            ctx.prec=70
            for _ in range(160):
                k=rng.randrange(2,9);z=[rng.randrange(-20,21)/4 for _ in range(k)];tz=[rng.randrange(-20,21)/4 for _ in range(k)];y=rng.randrange(k);t=rng.choice([.25,1,4,20]);a=rng.choice([0,.25,.5,.95,1]);r=loss_terms(z,tz,y,t,a)
                dz=list(map(lambda v:Decimal(str(v)),z));dtz=list(map(lambda v:Decimal(str(v)),tz));T=Decimal(str(t));A=Decimal(str(a))
                def dist(vals,temp):
                    ex=[(v/temp).exp() for v in vals];s=sum(ex);return [v/s for v in ex]
                p=dist(dz,Decimal(1));q=dist(dz,T);target=dist(dtz,T)
                hard=-p[y].ln();ce=-sum(v*s.ln() for v,s in zip(target,q));ent=-sum(v*v.ln() for v in target);kl=sum(v*(v.ln()-s.ln()) for v,s in zip(target,q))
                for actual,expected in [(r['loss'],(1-A)*hard+A*T*T*ce),(r['soft_ce'],ce),(r['teacher_entropy'],ent),(r['kl'],kl)]:self.assertAlmostEqual(actual,float(expected),delta=1e-10)
                for i,c in enumerate(r['classes']):self.assertAlmostEqual(c['gradient'],float(A*T*(q[i]-target[i])+(1-A)*(p[i]-(1 if i==y else 0))),delta=1e-12)
    def test_zero_step_and_input_immutability(self):
        p=self.payload();saved=copy.deepcopy(p);r=audit(p,4,.95,0)
        self.assertEqual(p,saved);self.assertEqual(r['updated_model'],p['model']);self.assertEqual(r['actual_loss_delta'],0)
        self.assertTrue(all(v['delta']==0 for v in r['parameters']))
    def test_evaluation_labels_cannot_change_gradient(self):
        p=self.payload();a=audit(p);p['evaluation']['original_test'][0]['label']^=1;b=audit(p)
        self.assertEqual(a['parameters'],b['parameters']);self.assertEqual(a['before'],b['before']);self.assertNotEqual(a['evaluation'],b['evaluation'])
    def test_teacher_entropy_identity_and_zero_soft_gradient(self):
        p=self.payload();_,z=forward(p['model'],p['row']['x']);p['row']['teacher_logits']=z
        r=audit(p,4,1,.02)
        self.assertAlmostEqual(r['before']['soft_ce'],r['before']['teacher_entropy'],places=13)
        self.assertEqual(r['before']['kl'],0);self.assertTrue(all(c['gradient']==0 for c in r['before']['classes']));self.assertGreater(r['before']['loss'],0)
    def test_overshoot_is_visible(self):
        r=audit(OVERSHOOT,1,.5,2)
        self.assertLess(r['directional_derivative'],0);self.assertLess(r['first_order_loss_delta'],0);self.assertGreater(r['actual_loss_delta'],2)
    def test_unclipped_underflow_loss(self):
        r=loss_terms([1000,-1000],[-1000,1000],1,1,1)
        self.assertEqual(r['loss'],2000);self.assertEqual(r['classes'][0]['gradient'],1);self.assertEqual(r['classes'][1]['gradient'],-1)
        self.assertTrue(all(math.isfinite(r[k]) for k in ['loss','teacher_entropy','kl']))
    def test_translation_invariance_and_sum_gradient(self):
        p=self.payload();_,z=forward(p['model'],p['row']['x']);r=loss_terms(z,p['row']['teacher_logits'],1,4,.95);s=loss_terms([v+23 for v in z],[v-71 for v in p['row']['teacher_logits']],1,4,.95)
        self.assertAlmostEqual(r['loss'],s['loss'],places=11);self.assertAlmostEqual(sum(c['gradient'] for c in r['classes']),0,places=13)
    def test_input_validation(self):
        p=self.payload()
        for key,value in [('temperature',True),('temperature',0),('alpha',1.1),('step',float('nan'))]:
            with self.assertRaises(ValueError):audit(p,**{key:value})
        for mut in [lambda p:p['row'].__setitem__('label',True),lambda p:p['model']['W1'][0].pop(),lambda p:p['row']['x'].__setitem__(0,float('inf')),lambda p:p['evaluation'].__setitem__('empty',[])]:
            q=copy.deepcopy(p);mut(q)
            with self.assertRaises(ValueError):audit(q)
    def test_cli_duplicate_invalid_and_valid(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'in.json';path.write_text('{"model":1,"model":2}')
            r=subprocess.run([sys.executable,'-m','distilllab','step-audit',str(path)],capture_output=True,text=True);self.assertNotEqual(r.returncode,0);self.assertIn('duplicate',r.stderr);self.assertEqual(r.stdout,'')
            path.write_text(json.dumps(self.payload()));r=subprocess.run([sys.executable,'-m','distilllab','step-audit',str(path)],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout),audit(self.payload()))

if __name__=='__main__':unittest.main()

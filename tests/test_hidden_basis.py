"""Signed hidden-unit permutations preserve a tanh network, not its labels.

This is a finite equivariance regression for the unchanged step/tail engines,
not a new training result or an independent backpropagation reference.
"""
import copy
import json
import math
import random
import re
import unittest
from importlib.resources import files
from distilllab.step import audit as step_audit, forward
from distilllab.tail import audit as tail_audit, INPUT_SCHEMA
from tests.test_tail import fixture


def basis(model, order, signs):
    out = copy.deepcopy(model)
    out['W1'] = [[signs[j]*v for v in model['W1'][i]] for j,i in enumerate(order)]
    out['b1'] = [signs[j]*model['b1'][i] for j,i in enumerate(order)]
    out['W2'] = [[signs[j]*row[i] for j,i in enumerate(order)] for row in model['W2']]
    return out


def gradient_model(model, rows, field):
    out = copy.deepcopy(model)
    for row in rows:
        key,i,j = re.fullmatch(r'(W1|b1|W2|b2)\[(\d+)\](?:\[(\d+)\])?', row['name']).groups()
        if j is None: out[key][int(i)] = row[field]
        else: out[key][int(i)][int(j)] = row[field]
    return out


class HiddenBasisTests(unittest.TestCase):
    def close(self, a, b):
        if isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for key in a: self.close(a[key],b[key])
        elif isinstance(a,list):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b): self.close(x,y)
        elif type(a) in (float,int):
            self.assertAlmostEqual(a,b,delta=3e-11+3e-12*max(abs(a),abs(b)))
        else: self.assertEqual(a,b)

    def check_basis(self, payload, order, signs):
        original=copy.deepcopy(payload)
        changed=copy.deepcopy(payload);changed['model']=basis(payload['model'],order,signs)
        a=tail_audit(payload);b=tail_audit(changed)
        self.close(a['logits_before'],b['logits_before'])
        self.close([signs[j]*a['hidden_before'][i] for j,i in enumerate(order)],b['hidden_before'])
        for left,right in zip(a['variants'],b['variants']):
            for key in ('teacher','before','after','original_teacher_after','logits_after','gradient_norm',
                        'first_order_kl_delta','actual_kl_delta','actual_original_kl_delta','comparison'):
                self.close(left[key],right[key])
            self.close(basis(left['updated_model'],order,signs),right['updated_model'])
            for field in ('hard_gradient','binary_gradient','coupled_tail_gradient','gradient','direct_gradient'):
                x=gradient_model(payload['model'],left['parameters'],field)
                y=gradient_model(changed['model'],right['parameters'],field)
                self.close(basis(x,order,signs),y)
            self.close([signs[j]*left['hidden_after'][i] for j,i in enumerate(order)],right['hidden_after'])
        request=dict(model=payload['model'],row=payload['row'],evaluation={'one':[payload['row']]})
        request2=dict(request,model=changed['model'])
        args=payload['temperature'],payload['alpha'],payload['step']
        a=step_audit(request,*args);b=step_audit(request2,*args)
        for key in ('before','after','logits_before','logits_after','evaluation','gradient_norm',
                    'update_norm','directional_derivative','actual_loss_delta'):
            self.close(a[key],b[key])
        self.close(basis(a['updated_model'],order,signs),b['updated_model'])
        self.close(basis(gradient_model(payload['model'],a['parameters'],'gradient'),order,signs),
                   gradient_model(changed['model'],b['parameters'],'gradient'))
        self.assertEqual(payload,original)

    def test_all_seven_saved_models_two_signed_bases(self):
        data=json.loads(files('distilllab').joinpath('step_data/frozen.json').read_text())
        for model in data['models'].values():
            for row_index in (0,39,79):
                p=dict(schema=INPUT_SCHEMA,model=model,row=data['splits']['student_train'][row_index],
                       temperature=4,alpha=.95,step=.02,tail_mix=.3,tail_shift=1)
                self.check_basis(p,list(range(5,-1,-1)),[1,-1,1,-1,1,-1])
                self.check_basis(p,[2,3,4,5,0,1],[-1]*6)

    def test_24_generated_models_temperature_alpha_and_step_endpoints(self):
        rng=random.Random(70193)
        for seed in range(24):
            p=fixture(seed,2+seed%7)
            p['temperature']=(.25,1,4,20)[seed%4]
            p['alpha']=(0,.3,.95,1)[seed%4]
            p['step']=(0,.02,.1,2)[seed%4]
            p['tail_mix']=(0,.2,1)[seed%3]
            order=[0,1];rng.shuffle(order);signs=[rng.choice([-1,1]) for _ in order]
            self.check_basis(p,order,signs)

    def test_single_hidden_unit_sign_and_saturated_tanh(self):
        for value in (.2,100):
            p=dict(schema=INPUT_SCHEMA,model=dict(W1=[[value,-value]],b1=[.1],W2=[[.5],[-.3],[.2]],b2=[.1,0,-.1]),
                   row=dict(x=[1,-1],label=1,teacher_logits=[2,-1,0]),temperature=1,alpha=.7,step=.2,tail_mix=.4,tail_shift=1)
            self.check_basis(p,[0],[-1])

    def test_flipping_only_one_side_is_not_a_network_symmetry(self):
        p=fixture(3,4);a=p['model'];b=basis(a,[0,1],[-1,1])
        # Undo the outgoing sign: the incoming sign alone changes the function.
        b['W2']=copy.deepcopy(a['W2'])
        za=forward(a,p['row']['x'])[1];zb=forward(b,p['row']['x'])[1]
        self.assertGreater(max(abs(x-y) for x,y in zip(za,zb)),.01)

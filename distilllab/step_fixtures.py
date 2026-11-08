"""Regenerate frozen original models, rows and independent-step witnesses."""
import json
import gzip
import random
from pathlib import Path
from .data import make_split,TEACHER_SPLIT,STUDENT_SPLIT,TEST_SPLIT,DIM,N_CLASSES
from .eval import train_teacher,train_student,TEMPERATURE,ALPHA,CONTROL_TEMPERATURE
from .mlp import MLP
from .step import audit


def model_dict(net):
    return {k:getattr(net,k) for k in ("W1","b1","W2","b2")}


OVERSHOOT = {'model': {'W1': [[0.20057454986283352], [-0.6956246915162576]], 'b1': [0.823806546494195, 1.9164579788584892], 'W2': [[0.11143685545315307, 2.6742460801481887], [-1.7745320793771404, -2.0714145514450206]], 'b2': [-1.4421976274197097, 0.16631100048775682]}, 'row': {'x': [1.9037996784677356], 'label': 1, 'teacher_logits': [2, -2]}, 'evaluation': {'control': [{'x': [1.9037996784677356], 'label': 1, 'teacher_logits': [2, -2]}]}}


def generate():
    teacher=train_teacher(make_split(*TEACHER_SPLIT))
    small=make_split(*STUDENT_SPLIT); test=make_split(*TEST_SPLIT)
    perm=list(range(len(small)));random.Random(99).shuffle(perm)
    models={"initial":model_dict(MLP(DIM,6,N_CLASSES,random.Random(5)))}
    nets={}
    for name,mode,t in [("direct","direct",TEMPERATURE),("distilled","distill",TEMPERATURE),("pseudo","pseudo",TEMPERATURE),("smoothing","smoothing",TEMPERATURE),("mismatched","control",CONTROL_TEMPERATURE),("T1","distill",1.0)]:
        net=train_student(small,mode,teacher,temperature=t,alpha=ALPHA,perm=perm)
        nets[name]=net;models[name]=model_dict(net)
    def rows(data):return [{"x":x,"label":y,"teacher_logits":teacher.logits(x)} for x,y in data]
    splits={"student_train":rows(small),"original_test":rows(test)}
    from .distill import mean_kl
    direct,distilled=nets["direct"],nets["distilled"]
    r={"teacher_acc":teacher.accuracy(test),"teacher_params":teacher.param_count(),"student_params":direct.param_count(),"direct_acc":direct.accuracy(test),"direct_train_acc":direct.accuracy(small),"distilled_acc":distilled.accuracy(test),"distilled_train_acc":distilled.accuracy(small),"control_acc":nets["mismatched"].accuracy(test),"pseudo_acc":nets["pseudo"].accuracy(test),"smoothing_acc":nets["smoothing"].accuracy(test),"ablation_t1_acc":nets["T1"].accuracy(test),"kl_direct":mean_kl(direct,teacher,test),"kl_distilled":mean_kl(distilled,teacher,test)}
    for target,a,b in [("distill_gain","distilled_acc","direct_acc"),("pseudo_gain_over_direct","pseudo_acc","direct_acc"),("distill_gain_over_pseudo","distilled_acc","pseudo_acc"),("overfit_gap_direct","direct_train_acc","direct_acc"),("overfit_gap_distilled","distilled_train_acc","distilled_acc")]:r[target]=r[a]-r[b]
    return {"schema":"distilllab-frozen-v1","teacher_model":model_dict(teacher),"models":models,"splits":splits,"original_report":r,"controls":{"overshoot":OVERSHOOT},"scope":"Original synthetic splits and unchanged original training. Student row supplies the one-step update; test labels are read only for descriptive evaluation. No selection or refit."}


def write_archive(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    fixture=generate()
    (root/"frozen.json").write_text(json.dumps(fixture,separators=(",",":"))+"\n")
    configs=[(0,4,.95,.02),(17,1,0,.1),(39,8,1,.5),(79,.25,.5,2),(5,20,.95,0)]
    manifest=[]
    witnesses=[]
    for name,model in fixture["models"].items():
        for index,t,a,eta in configs:
            payload={"model":model,"row":fixture["splits"]["student_train"][index],"evaluation":fixture["splits"]}
            witness=audit(payload,t,a,eta)
            filename=f"{name}-{index:02d}.json"
            witnesses.append({"input":{"model":name,"row":index,"temperature":t,"alpha":a,"step":eta},"report":witness})
            manifest.append({"file":filename,"model":name,"row":index,"temperature":t,"alpha":a,"step":eta})
    payload = fixture["controls"]["overshoot"]
    witnesses.append({"input":{"control":"overshoot","temperature":1,"alpha":.5,"step":2},"report":audit(payload,1,.5,2)})
    manifest.append({"file":"overshoot.json","control":"overshoot","temperature":1,"alpha":.5,"step":2})
    (root/"witnesses.json.gz").write_bytes(gzip.compress((json.dumps(witnesses,separators=(",",":"))+"\n").encode(),mtime=0))
    return fixture,manifest

if __name__=="__main__":
    import sys
    fixture,manifest=write_archive(sys.argv[1] if len(sys.argv)>1 else "step_data")
    print(json.dumps({"original_report":fixture["original_report"],"witnesses":len(manifest)},indent=2))

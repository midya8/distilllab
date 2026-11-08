"""Stable one-row student update audit. Separate from the original clipped loss."""
from __future__ import annotations
import copy
import math


def _num(v, name, lo, hi):
    if type(v) not in (int, float) or not math.isfinite(v) or not lo <= v <= hi:
        raise ValueError(name + " outside finite bounds")
    return float(v)


def validate(payload):
    if type(payload) is not dict or set(payload) != {"model", "row", "evaluation"}:
        raise ValueError("expected model, row and evaluation")
    m = payload["model"]
    if type(m) is not dict or set(m) != {"W1", "b1", "W2", "b2"}:
        raise ValueError("invalid model fields")
    h = len(m["b1"]) if type(m["b1"]) is list else 0
    k = len(m["b2"]) if type(m["b2"]) is list else 0
    d = len(m["W1"][0]) if type(m["W1"]) is list and m["W1"] and type(m["W1"][0]) is list else 0
    if not 1 <= h <= 16 or not 2 <= k <= 16 or not 1 <= d <= 64:
        raise ValueError("model dimensions exceed 64 inputs, 16 hidden, 16 classes")
    for key, rows, cols in [("W1", h, d), ("W2", k, h)]:
        if type(m[key]) is not list or len(m[key]) != rows:
            raise ValueError("invalid " + key)
        for row in m[key]:
            if type(row) is not list or len(row) != cols: raise ValueError("invalid " + key)
            for v in row: _num(v, key, -100, 100)
    for key in ("b1", "b2"):
        for v in m[key]: _num(v, key, -100, 100)
    def check_row(r):
        if type(r) is not dict or set(r) != {"x", "label", "teacher_logits"}:
            raise ValueError("invalid row")
        if type(r["x"]) is not list or len(r["x"]) != d: raise ValueError("invalid x")
        for v in r["x"]: _num(v, "x", -20, 20)
        if type(r["label"]) is not int or not 0 <= r["label"] < k: raise ValueError("invalid label")
        if type(r["teacher_logits"]) is not list or len(r["teacher_logits"]) != k: raise ValueError("invalid teacher logits")
        for v in r["teacher_logits"]: _num(v, "teacher logit", -10000, 10000)
    check_row(payload["row"])
    ev = payload["evaluation"]
    if type(ev) is not dict or not ev or len(ev) > 4: raise ValueError("invalid splits")
    total = 0
    for name, rows in ev.items():
        if type(name) is not str or not name or len(name) > 40 or type(rows) is not list or not rows: raise ValueError("invalid split")
        total += len(rows)
        for r in rows: check_row(r)
    if total > 512: raise ValueError("at most 512 evaluation rows")
    return d, h, k


def distribution(z, temperature=1.0):
    scaled = [v / temperature for v in z]
    top = max(scaled)
    exps = [math.exp(v-top) for v in scaled]
    lognorm = math.log(math.fsum(exps))
    logs = [(v-top)-lognorm for v in scaled]
    return [math.exp(v) for v in logs], logs


def forward(model, x):
    hidden = [math.tanh(math.fsum(w*v for w, v in zip(row, x))+b) for row, b in zip(model["W1"], model["b1"])]
    logits = [math.fsum(w*v for w, v in zip(row, hidden))+b for row, b in zip(model["W2"], model["b2"])]
    return hidden, logits


def loss_terms(z, teacher_logits, label, temperature, alpha):
    p, lp = distribution(z)
    q, lq = distribution(z, temperature)
    t, lt = distribution(teacher_logits, temperature)
    hard = -lp[label]
    ce_terms = [-v*l for v,l in zip(t,lq)]
    entropy_terms = [-v*l for v,l in zip(t,lt)]
    kl_terms = [v*(l-r) for v,l,r in zip(t,lt,lq)]
    ce, entropy, kl = map(math.fsum, (ce_terms, entropy_terms, kl_terms))
    soft_g = [alpha*temperature*(s-v) for s,v in zip(q,t)]
    hard_g = [(1-alpha)*(v-(1 if i==label else 0)) for i,v in enumerate(p)]
    classes = [{"class":i,"teacher":t[i],"student_T":q[i],"student_1":p[i],"soft_ce_term":ce_terms[i],"teacher_entropy_term":entropy_terms[i],"kl_term":kl_terms[i],"soft_gradient":soft_g[i],"hard_gradient":hard_g[i],"gradient":soft_g[i]+hard_g[i]} for i in range(len(z))]
    return {"hard_ce":hard,"soft_ce":ce,"teacher_entropy":entropy,"kl":kl,"weighted_hard":(1-alpha)*hard,"weighted_soft":alpha*temperature**2*ce,"weighted_kl":alpha*temperature**2*kl,"loss":(1-alpha)*hard+alpha*temperature**2*ce,"prediction":z.index(max(z)),"classes":classes}


def _norm(values):
    return math.sqrt(math.fsum(v*v for v in values))


def _flatten(m):
    return [v for r in m["W1"] for v in r]+m["b1"]+[v for r in m["W2"] for v in r]+m["b2"]


def audit(payload, temperature=4.0, alpha=0.95, step=0.02):
    d,h,k = validate(payload)
    temperature = _num(temperature,"temperature",0.25,20)
    alpha = _num(alpha,"alpha",0,1)
    step = _num(step,"step",0,2)
    m, row = payload["model"],payload["row"]
    hidden,z = forward(m,row["x"])
    before = loss_terms(z,row["teacher_logits"],row["label"],temperature,alpha)
    dl = [c["gradient"] for c in before["classes"]]
    hidden_grad = [math.fsum(dl[j]*m["W2"][j][i] for j in range(k))*(1-hidden[i]**2) for i in range(h)]
    g = {"W1":[[v*x for x in row["x"]] for v in hidden_grad],"b1":hidden_grad,"W2":[[v*a for a in hidden] for v in dl],"b2":dl}
    updated = copy.deepcopy(m)
    parameter_rows = []
    for key in ("W1","b1","W2","b2"):
        matrix = key in ("W1","W2")
        for i, v in enumerate(m[key]):
            for j, value in enumerate(v if matrix else [v]):
                grad = g[key][i][j] if matrix else g[key][i]
                nxt = value-step*grad
                if matrix: updated[key][i][j]=nxt
                else: updated[key][i]=nxt
                parameter_rows.append({"name":f"{key}[{i}]"+(f"[{j}]" if matrix else ""),"before":value,"gradient":grad,"delta":-step*grad,"after":nxt})
    hidden2,z2 = forward(updated,row["x"])
    after = loss_terms(z2,row["teacher_logits"],row["label"],temperature,alpha)
    evaluation = {}
    for name, rows in payload["evaluation"].items():
        records=[]
        for i,r in enumerate(rows):
            _, a=forward(m,r["x"]); _, b=forward(updated,r["x"])
            pa,la=distribution(a); pb,lb=distribution(b)
            pred_a,pred_b=a.index(max(a)),b.index(max(b))
            records.append({"position":i,"label":r["label"],"before_prediction":pred_a,"after_prediction":pred_b,"before_nll":-la[r["label"]],"after_nll":-lb[r["label"]],"before_label_probability":pa[r["label"]],"after_label_probability":pb[r["label"]]})
        n=len(records)
        evaluation[name]={"count":n,"before_accuracy":sum(r["before_prediction"]==r["label"] for r in records)/n,"after_accuracy":sum(r["after_prediction"]==r["label"] for r in records)/n,"before_nll":math.fsum(r["before_nll"] for r in records)/n,"after_nll":math.fsum(r["after_nll"] for r in records)/n,"changed_predictions":sum(r["before_prediction"]!=r["after_prediction"] for r in records),"rows":records}
    flat=_flatten(g)
    return {"schema":"distilllab-step-v1","controls":{"temperature":temperature,"alpha":alpha,"step":step},"dimensions":{"inputs":d,"hidden":h,"classes":k,"parameters":len(flat)},"hidden_before":hidden,"hidden_after":hidden2,"logits_before":z,"logits_after":z2,"before":before,"after":after,"gradient_norm":_norm(flat),"update_norm":step*_norm(flat),"directional_derivative":-math.fsum(v*v for v in flat),"first_order_loss_delta":-step*math.fsum(v*v for v in flat),"actual_loss_delta":after["loss"]-before["loss"],"parameters":parameter_rows,"updated_model":updated,"evaluation":evaluation}

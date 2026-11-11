/* Independent scalar reverse-mode evaluator for the target/tail audit.
 * No replayed native derivatives, dependencies, remote requests or model fitting.
 */
(function(root,factory){if(typeof module==='object'&&module.exports)module.exports=factory();else root.DistillTail=factory();})(globalThis,function(){
'use strict';
const INPUT='distilllab.tail.input.v1',REPORT='distilllab.tail.report.v1';
const SCOPE='One fixed-teacher training-row gradient and one simultaneous step per variant. KL chain rule, not a new method, DKD training run or causal explanation of original gain. Tail interventions preserve target/rest mass at the selected temperature, not generalization, label semantics or T=1 confidence.';
const sum=xs=>{let s=0,c=0;for(const x of xs){const t=s+x;c+=Math.abs(s)>=Math.abs(x)?(s-t)+x:(x-t)+s;s=t;}return s+c;};
const keys=(x,ks)=>x&&typeof x==='object'&&!Array.isArray(x)&&Object.keys(x).sort().join('|')===[...ks].sort().join('|');
const num=(x,name,lo,hi)=>{if(typeof x!=='number'||!Number.isFinite(x)||x<lo||x>hi)throw Error(name+' outside finite bounds');return x;};
function validate(p){
 if(!keys(p,['schema','model','row','temperature','alpha','step','tail_mix','tail_shift'])||p.schema!==INPUT)throw Error('expected exact '+INPUT+' fields');
 const m=p.model;if(!keys(m,['W1','b1','W2','b2']))throw Error('invalid model');
 const h=Array.isArray(m.b1)?m.b1.length:0,k=Array.isArray(m.b2)?m.b2.length:0,d=Array.isArray(m.W1)&&Array.isArray(m.W1[0])?m.W1[0].length:0;
 if(d<1||d>64||h<1||h>16||k<2||k>16)throw Error('invalid model dimensions');
 for(const [key,rows,cols] of [['W1',h,d],['W2',k,h]]){if(!Array.isArray(m[key])||m[key].length!==rows)throw Error('invalid '+key);for(const row of m[key]){if(!Array.isArray(row)||row.length!==cols)throw Error('invalid '+key);for(const x of row)num(x,key,-100,100);}}
 for(const key of ['b1','b2'])for(const x of m[key])num(x,key,-100,100);
 const r=p.row;if(!keys(r,['x','label','teacher_logits'])||!Array.isArray(r.x)||r.x.length!==d||!Number.isInteger(r.label)||r.label<0||r.label>=k||!Array.isArray(r.teacher_logits)||r.teacher_logits.length!==k)throw Error('invalid row');
 for(const x of r.x)num(x,'x',-20,20);for(const x of r.teacher_logits)num(x,'teacher logit',-10000,10000);
 num(p.temperature,'temperature',.25,20);num(p.alpha,'alpha',0,1);num(p.step,'step',0,2);num(p.tail_mix,'tail_mix',0,1);
 if(!Number.isInteger(p.tail_shift)||p.tail_shift<0||p.tail_shift>=k-1)throw Error('invalid tail_shift');
 return{inputs:d,hidden:h,classes:k,parameters:h*d+h+k*h+k};
}
const lse=xs=>{const m=Math.max(...xs);return m+Math.log(sum(xs.map(v=>Math.exp(v-m))));};
function partition(z,y,t){const a=z.map(v=>v/t),ids=a.map((_,i)=>i).filter(i=>i!==y),tail=lse(ids.map(i=>a[i])),top=Math.max(a[y],tail),norm=Math.log(Math.exp(a[y]-top)+Math.exp(tail-top)),bl=[a[y]-top-norm,tail-top-norm],tl=ids.map(i=>a[i]-tail);return{ids,binary_log:bl,binary:bl.map(Math.exp),tail_log:tl,tail:tl.map(Math.exp)};}
function teacherOf(p,y,k,mix,shift){
 const n=k-1,rot=[...p.tail.slice(shift),...p.tail.slice(0,shift)],rl=[...p.tail_log.slice(shift),...p.tail_log.slice(0,shift)];
 let r,lr;if(n===1){r=[1];lr=[0];}else if(mix===0){r=rot;lr=rl;}else if(mix===1){r=Array(n).fill(1/n);lr=Array(n).fill(-Math.log(n));}else{lr=rl.map(v=>lse([Math.log1p(-mix)+v,Math.log(mix)-Math.log(n)]));r=lr.map(Math.exp);}
 const [qt,qm]=p.binary,[lqt,lqm]=p.binary_log,probs=Array(k).fill(0),logs=Array(k).fill(0);probs[y]=qt;logs[y]=lqt;p.ids.forEach((i,j)=>{probs[i]=qm*r[j];logs[i]=lqm+lr[j];});
 return{probabilities:probs,log_probabilities:logs,binary:[qt,qm],binary_log:[lqt,lqm],tail:r,tail_log:lr};
}
function tape(){
 const nodes=[];
 function v(x,parents=[]){const n={v:x,parents,g:0};nodes.push(n);return n;}
 const add=(a,b)=>v(a.v+b.v,[[a,1],[b,1]]),scale=(a,b)=>v(a.v*b,[[a,b]]),sub=(a,b)=>add(a,scale(b,-1));
 const total=xs=>v(sum(xs.map(x=>x.v)),xs.map(x=>[x,1]));
 const exp=a=>{const e=Math.exp(a.v);return v(e,[[a,e]]);},log=a=>v(Math.log(a.v),[[a,1/a.v]]);
 const tanh=a=>{const h=Math.tanh(a.v);return v(h,[[a,1-h*h]]);};
 const logsum=xs=>{const top=Math.max(...xs.map(x=>x.v));return add(v(top),log(total(xs.map(x=>exp(add(x,v(-top)))))));};
 function backward(loss){nodes.forEach(n=>{n.g=0;});loss.g=1;for(let i=nodes.length-1;i>=0;i--){const n=nodes[i];for(const [p,d] of n.parents)p.g+=n.g*d;}}
 return{v,add,scale,sub,total,exp,log,tanh,logsum,backward};
}
function evaluate(m,row,teacher,T,alpha){
 const A=tape(),{v,add,scale,sub,total,tanh,logsum}=A,model={},leaves=[],y=row.label,k=m.b2.length;
 for(const key of ['W1','b1','W2','b2']){if(key==='W1'||key==='W2')model[key]=m[key].map((r,i)=>r.map((x,j)=>{const n=v(x);leaves.push({key,i,j,value:x,name:key+'['+i+']['+j+']',node:n});return n;}));else model[key]=m[key].map((x,i)=>{const n=v(x);leaves.push({key,i,j:0,value:x,name:key+'['+i+']',node:n});return n;});}
 const hidden=model.W1.map((r,i)=>tanh(add(total(r.map((w,j)=>scale(w,row.x[j]))),model.b1[i])));
 const logits=model.W2.map((r,i)=>add(total(r.map((w,j)=>{const h=hidden[j];return A.v(w.v*h.v,[[w,h.v],[h,w.v]]);})),model.b2[i]));
 const z=logits.map(n=>n.v),ids=z.map((_,i)=>i).filter(i=>i!==y),zs=logits.map(n=>scale(n,1/T)),tailNorm=logsum(ids.map(i=>zs[i])),norm=logsum([zs[y],tailNorm]),lpTarget=sub(zs[y],norm),lpRest=sub(tailNorm,norm),lp=zs.map(n=>sub(n,norm)),lt=ids.map(i=>sub(zs[i],tailNorm));
 const lp1=logits.map(n=>sub(n,logsum(logits)));
 function kl(q,lq,ls){return total(q.map((w,i)=>scale(sub(v(lq[i]),ls[i]),w)));}
 const binaryKL=kl(teacher.binary,teacher.binary_log,[lpTarget,lpRest]),tailKL=kl(teacher.tail,teacher.tail_log,lt),directKL=kl(teacher.probabilities,teacher.log_probabilities,lp),qm=teacher.binary[1],sc=alpha*T*T;
 const hard=scale(lp1[y],-(1-alpha)),binary=scale(binaryKL,sc),conditional=scale(tailKL,sc),coupled=scale(conditional,qm),full=total([hard,binary,coupled]),direct=add(hard,scale(directKL,sc));
 const roots={hard,binary,conditional,coupled_tail:coupled,total:full,direct},gs={},cs={};
 for(const [name,n] of Object.entries(roots)){A.backward(n);gs[name]=leaves.map(x=>x.node.g);cs[name]=logits.map(x=>x.g);}
 // Numeric report distributions use the same binary/conditional factorization,
 // but every derivative above comes from actual independent reverse-mode AD.
 const p=partition(z,y,T),prob=Array(k).fill(0),logs=Array(k).fill(0);prob[y]=p.binary[0];logs[y]=p.binary_log[0];ids.forEach((i,j)=>{prob[i]=p.binary[1]*p.tail[j];logs[i]=p.binary_log[1]+p.tail_log[j];});
 const full1=(()=>{const top=Math.max(...z),ln=Math.log(sum(z.map(x=>Math.exp(x-top))));return z.map(x=>x-top-ln);})(),p1=full1.map(Math.exp),entropy=-sum(teacher.probabilities.map((x,i)=>x*teacher.log_probabilities[i])),ce=-sum(teacher.probabilities.map((x,i)=>x*logs[i]));
 const classes=z.map((_,i)=>({class:i,target:i===y,student:prob[i],student_log:logs[i],teacher:teacher.probabilities[i],teacher_log:teacher.log_probabilities[i],student_1:p1[i],student_tail:i===y?0:p.tail[ids.indexOf(i)],teacher_tail:i===y?0:teacher.tail[ids.indexOf(i)],hard_gradient:cs.hard[i],binary_gradient:cs.binary[i],conditional_gradient:cs.conditional[i],coupled_tail_gradient:cs.coupled_tail[i],total_gradient:cs.total[i],direct_gradient:cs.direct[i],gradient_residual:cs.total[i]-cs.direct[i]}));
 const split=binaryKL.v+qm*tailKL.v;
 const before={target_mass:teacher.binary[0],non_target_mass:qm,student_target_mass:p.binary[0],student_non_target_mass:p.binary[1],teacher_mass_residual:sum(teacher.probabilities)-1,binary_kl:binaryKL.v,conditional_kl:tailKL.v,coupled_tail_kl:qm*tailKL.v,direct_kl:directKL.v,split_kl:split,kl_residual:split-directKL.v,hard_ce:-full1[y],soft_ce:ce,teacher_entropy:entropy,ce_entropy_residual:ce-entropy-directKL.v,weighted_hard:hard.v,weighted_binary:binary.v,weighted_coupled_tail:coupled.v,kl_objective:full.v,original_ce_objective:hard.v+sc*ce,teacher_only_offset:sc*entropy,prediction:z.indexOf(Math.max(...z)),classes};
 return{before,grads:gs,leaves,hidden:hidden.map(n=>n.v),logits:z};
}
function audit(p){
 const dims=validate(p),m=p.model,row=p.row,T=p.temperature,alpha=p.alpha,eta=p.step,k=dims.classes,y=row.label,part=partition(row.teacher_logits,y,T),original=teacherOf(part,y,k,0,0);
 const baseline=evaluate(m,row,original,T,alpha),reports=[];
 const configs=[['original',0,0],['uniform_tail',1,0],['rotated_tail',0,k>2?1:0],['configured',p.tail_mix,p.tail_shift]];
 for(const [name,mix,shift] of configs){
  const teacher=teacherOf(part,y,k,mix,shift),e=name==='original'?baseline:evaluate(m,row,teacher,T,alpha),b=e.before,g=e.grads,updated=JSON.parse(JSON.stringify(m)),rows=[];
  e.leaves.forEach((leaf,index)=>{const {key,i,j,value,name}=leaf,nxt=value-eta*g.total[index];if(key==='W1'||key==='W2')updated[key][i][j]=nxt;else updated[key][i]=nxt;rows.push({name,before:value,hard_gradient:g.hard[index],binary_gradient:g.binary[index],coupled_tail_gradient:g.coupled_tail[index],gradient:g.total[index],direct_gradient:g.direct[index],component_residual:sum([g.hard[index],g.binary[index],g.coupled_tail[index]])-g.total[index],direct_residual:g.total[index]-g.direct[index],delta:-eta*g.total[index],after:nxt});});
  const eafter=evaluate(updated,row,teacher,T,alpha),oafter=evaluate(updated,row,original,T,alpha),after=eafter.before,originalAfter=oafter.before,norm2=sum(g.total.map(x=>x*x));
  reports.push({name,tail_mix:mix,tail_shift:shift,teacher,before:b,after,original_teacher_after:originalAfter,logits_after:eafter.logits,hidden_after:eafter.hidden,parameters:rows,updated_model:updated,gradient_norm:Math.sqrt(norm2),first_order_kl_delta:-eta*norm2,actual_kl_delta:after.kl_objective-b.kl_objective,actual_original_kl_delta:originalAfter.kl_objective-baseline.before.kl_objective,max_parameter_direct_residual:Math.max(...rows.map(r=>Math.abs(r.direct_residual))),max_parameter_component_residual:Math.max(...rows.map(r=>Math.abs(r.component_residual)))});
 }
 const base=reports[0];for(const r of reports)r.comparison={binary_loss_delta:r.before.binary_kl-base.before.binary_kl,entropy_delta:r.before.teacher_entropy-base.before.teacher_entropy,gradient_delta_norm:Math.sqrt(sum(r.parameters.map((x,i)=>(x.gradient-base.parameters[i].gradient)**2))),target_mass_delta:r.before.target_mass-base.before.target_mass,non_target_mass_delta:r.before.non_target_mass-base.before.non_target_mass};
 return{schema:REPORT,controls:{temperature:T,alpha,step:eta,tail_mix:p.tail_mix,tail_shift:p.tail_shift},dimensions:dims,label:y,hidden_before:baseline.hidden,logits_before:baseline.logits,variants:reports,scope:SCOPE};
}
function strictJSON(text){if(text.length>1500000)throw Error('JSON exceeds 1.5 MB');let pos=0;function ws(){while(/\s/.test(text[pos]||'')&&pos<text.length)pos++;}function val(){ws();const ch=text[pos];if(ch==='{'||ch==='['){pos++;const object=ch==='{',close=object?'}':']',seen=new Set();ws();if(text[pos]===close){pos++;return;}while(true){if(object){ws();if(text[pos]!=='"')throw Error('invalid JSON key');const key=string();if(seen.has(key))throw Error('duplicate JSON key: '+key);seen.add(key);ws();if(text[pos++]!==':')throw Error('invalid colon');}val();ws();if(text[pos]===close){pos++;return;}if(text[pos++]!==',')throw Error('invalid separator');}}else if(ch==='"'){string();}else{const m=text.slice(pos).match(/^(?:-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)/);if(!m)throw Error('invalid JSON value');pos+=m[0].length;}}function string(){const start=pos++;while(pos<text.length){if(text[pos]==='\\'){pos+=2;continue;}if(text[pos++]==='"')return JSON.parse(text.slice(start,pos));}throw Error('unterminated string');}val();ws();if(pos!==text.length)throw Error('trailing JSON');const p=JSON.parse(text);return p;}
return{audit,validate,strictJSON};
});

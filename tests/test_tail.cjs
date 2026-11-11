'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),zlib=require('node:zlib');
const engine=require('../docs/tail-engine.js');
const data=JSON.parse(fs.readFileSync(path.join(__dirname,'../distilllab/tail_data/examples.json')));
const witnesses=JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(__dirname,'../distilllab/tail_data/1-witnesses.json.gz'))));
let fields=0;
function match(a,b,p='root'){
 assert.equal(typeof a,typeof b,p);if(typeof a==='number'){fields++;assert.ok(Number.isFinite(a)&&Number.isFinite(b),p);assert.ok(Math.abs(a-b)<=2e-9+2e-9*Math.max(Math.abs(a),Math.abs(b)),p+': '+a+' vs '+b);}
 else if(a===null||typeof a!=='object')assert.equal(a,b,p);else{assert.deepEqual(Object.keys(a).sort(),Object.keys(b).sort(),p);for(const k of Object.keys(a))match(a[k],b[k],p+'.'+k);}
}
test('independent reverse-mode AD matches every native report field and parameter component',()=>{for(const c of data.examples){const witness=witnesses.find(w=>w.id===c.id);assert.ok(witness);match(engine.audit(c.input),witness.report,c.id);}console.log(JSON.stringify({examples:data.examples.length,variants:data.examples.length*4,numericFields:fields,parameterPositions:witnesses.reduce((s,w)=>s+w.report.variants.reduce((a,v)=>a+v.parameters.length,0),0)}));});
test('strict JSON, bounds, no input mutation and binary singleton',()=>{
 for(const text of ['{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}','{}bad','[1,]'])assert.throws(()=>engine.strictJSON(text));
 const p=JSON.parse(JSON.stringify(data.examples[0].input)),before=JSON.stringify(p);engine.audit(p);assert.equal(JSON.stringify(p),before);
 for(const [k,v] of [['temperature',0],['alpha',2],['step',-1],['tail_mix',Infinity],['tail_shift',true],['schema','no']])assert.throws(()=>engine.audit({...p,[k]:v}));
 const binary=engine.audit(data.examples.find(x=>x.id==='two-classes-singleton-tail').input);for(const v of binary.variants){assert.equal(v.before.conditional_kl,0);assert.equal(v.comparison.gradient_delta_norm,0);}
});
module.exports={match,data};

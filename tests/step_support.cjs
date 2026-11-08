'use strict';
const fs=require('node:fs'),path=require('node:path'),zlib=require('node:zlib'),assert=require('node:assert/strict');
const data=JSON.parse(fs.readFileSync(path.join(__dirname,'../distilllab/step_data/frozen.json'))),witnesses=JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(__dirname,'../distilllab/step_data/witnesses.json.gz'))));
function equal(a,b,p='root'){if(typeof a==='number'){assert.equal(typeof b,'number',p);assert.ok(Number.isFinite(b),p);if(a===0&&b===0){}else if(Number.isInteger(a)&&Number.isInteger(b))assert.equal(a,b,p);else assert.ok(Math.abs(a-b)<=1e-12+1e-12*Math.max(Math.abs(a),Math.abs(b)),p+': '+a+' != '+b);}else if(Array.isArray(a)){assert.ok(Array.isArray(b),p);assert.equal(a.length,b.length,p);a.forEach((x,i)=>equal(x,b[i],p+'.'+i));}else if(a!==null&&typeof a==='object'){assert.deepEqual(Object.keys(a).sort(),Object.keys(b).sort(),p);for(const k in a)equal(a[k],b[k],p+'.'+k);}else assert.equal(a,b,p);}
const payload=m=>m.control?data.controls[m.control]:{model:data.models[m.model],row:data.splits.student_train[m.row],evaluation:data.splits};
module.exports={data,witnesses,equal,payload};

'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),zlib=require('node:zlib'),{spawn,spawnSync}=require('node:child_process'),{waitForDebugger}=require('./chrome_support.js');
const data=JSON.parse(fs.readFileSync(path.join(__dirname,'../distilllab/tail_data/examples.json'))),witnesses=JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(__dirname,'../distilllab/tail_data/witnesses.json.gz'))));
const delay=ms=>new Promise(r=>setTimeout(r,ms));
function equal(a,b,p='root'){assert.equal(typeof a,typeof b,p);if(typeof a==='number')assert.ok(Number.isFinite(b)&&Math.abs(a-b)<=2e-9+2e-9*Math.max(Math.abs(a),Math.abs(b)),p+': '+a+' '+b);else if(a===null||typeof a!=='object')assert.equal(a,b,p);else{assert.deepEqual(Object.keys(a).sort(),Object.keys(b).sort(),p);for(const k of Object.keys(a))equal(a[k],b[k],p+'.'+k);}}
test('actual Chrome: all target-tail reports, every table value/coordinate, trusted downloads and stale-state refusal',{timeout:240000},async()=>{
 const profile=fs.mkdtempSync(path.join(os.tmpdir(),'distill-tail-chrome-')),downloads=fs.mkdtempSync(path.join(os.tmpdir(),'distill-tail-download-'));let ws,browser,error,diagnostics='';
 try{
 browser=spawn(process.env.CHROME_BIN||'google-chrome',['--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],{stdio:['ignore','ignore','pipe'],detached:true});browser.on('error',e=>error=e);browser.stderr.on('data',d=>diagnostics=(diagnostics+d).slice(-5000));
 for(let i=0;i<1200&&!error&&browser.exitCode===null&&!fs.existsSync(path.join(profile,'DevToolsActivePort'));i++)await delay(25);
 assert.ok(fs.existsSync(path.join(profile,'DevToolsActivePort')),String(error||diagnostics));let port='';for(let i=0;i<1200&&browser.exitCode===null;i++){port=fs.readFileSync(path.join(profile,'DevToolsActivePort'),'utf8').split('\n')[0];if(/^\d+$/.test(port)&&Number(port)>0)break;await delay(25);}assert.ok(/^\d+$/.test(port)&&Number(port)>0,'complete debugger port: '+JSON.stringify(port)+' '+diagnostics);let tab;try{tab=await waitForDebugger('http://127.0.0.1:'+port+'/json/list',{alive:()=>browser.exitCode===null});}catch(e){throw Error(e.message+' port='+port+' exit='+browser.exitCode+' stderr='+diagnostics);}
 ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});let seq=0;const pending=new Map(),events=[];ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const p=pending.get(m.id);if(p){pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}}else events.push(m);};
 const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});
 const js=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 await call('Page.enable');await call('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:downloads,eventsEnabled:true});await call('Page.navigate',{url:process.env.DISTILL_TAIL_URL||'file://'+path.resolve(__dirname,'../docs/tail.html')});
 for(let i=0;i<500;i++){if(await js('Boolean(globalThis.DISTILL_TAIL_REPORT)'))break;await delay(20);}assert.equal(await js('DISTILL_TAIL_REPORT.schema'),'distilllab.tail.report.v1');
 const screenshots=process.env.DISTILL_TAIL_SCREEN_DIR;if(screenshots)fs.mkdirSync(screenshots,{recursive:true});let classCells=0,paramCells=0,coordinates=0,layouts=0;
 for(const width of [1280,390,320]){
  await call('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:width<700});
  for(let n=0;n<data.examples.length;n++){
   await js(`document.getElementById('example').value=${n};document.getElementById('example').dispatchEvent(new Event('change'))`);
   equal(witnesses[n].report,await js('DISTILL_TAIL_REPORT'),data.examples[n].id);
   const counts=await js(`(()=>{let pc=0,cc=0,coords=0;const r=DISTILL_TAIL_REPORT,must=(v,m)=>{if(!v)throw Error(m);};
    for(let vi=0;vi<r.variants.length;vi++){
     const s=document.getElementById('variant');s.value=vi;s.dispatchEvent(new Event('change'));const v=r.variants[vi],cr=[...document.querySelectorAll('#classes tr')],pr=[...document.querySelectorAll('#parameters tr')];
     must(cr.length===v.before.classes.length,'class count');must(pr.length===v.parameters.length,'parameter count');
     const classFields=['teacher','student','teacher_tail','student_tail','hard_gradient','binary_gradient','conditional_gradient','coupled_tail_gradient','total_gradient','direct_gradient','gradient_residual'];
     v.before.classes.forEach((c,i)=>{classFields.forEach((key,j)=>{const cell=cr[i].cells[j+1];if(c.target&&(key==='teacher_tail'||key==='student_tail'))must(cell.textContent==='-','undefined target tail');else{must(Number(cell.dataset.value)===c[key],'class '+i+' '+key);cc++;}});});
     v.parameters.forEach((p,i)=>{must(pr[i].cells[0].textContent===p.name,'parameter name');['before','hard_gradient','binary_gradient','coupled_tail_gradient','gradient','direct_gradient','after','direct_residual'].forEach((key,j)=>{must(Number(pr[i].cells[j+1].dataset.value)===p[key],'parameter '+i+' '+key);pc++;});const c=document.getElementById('coordinate');c.value=i;c.dispatchEvent(new Event('change'));must(document.getElementById('coordinate-detail').textContent.includes(p.name),'coordinate detail');must(pr[i].classList.contains('selected'),'selected coordinate');coords++;});
    }
    const s=document.getElementById('variant');s.value='0';s.dispatchEvent(new Event('change'));return{pc,cc,coords};})()`);
   classCells+=counts.cc;paramCells+=counts.pc;coordinates+=counts.coords;layouts++;
   assert.ok(await js('document.documentElement.scrollWidth<=innerWidth'),'overflow '+n+' '+width);
   assert.ok(await js(`(()=>{for(const e of document.querySelectorAll('.table-wrap')){e.scrollLeft=e.scrollWidth;const last=e.querySelector('tbody tr')?.lastElementChild;if(last&&last.getBoundingClientRect().right>e.getBoundingClientRect().right+2)return false;e.scrollLeft=0;}return true;})()`),'rightmost cells '+width);
   assert.ok(await js('[...document.querySelectorAll("button,input,select,summary")].filter(e=>e.getClientRects().length).every(e=>e.getBoundingClientRect().height>=44)'),'touch '+width);
   if(screenshots){await js('document.activeElement.blur();window.scrollTo(0,0)');const m=await call('Page.getLayoutMetrics'),image=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip:{x:0,y:0,width:m.cssContentSize.width,height:m.cssContentSize.height,scale:1}});fs.writeFileSync(path.join(screenshots,data.examples[n].id+'-'+width+'.png'),Buffer.from(image.data,'base64'));}
  }
  // A separate rightmost-scroll visual state uses a short control so the last
  // columns are actually visible in the captured pixels at every width.
  if(screenshots){await js(`document.getElementById('example').value=${data.examples.findIndex(c=>c.id==='confidence-matched-tail')};document.getElementById('example').dispatchEvent(new Event('change'));for(const e of document.querySelectorAll('.table-wrap'))e.scrollLeft=e.scrollWidth;document.activeElement.blur();window.scrollTo(0,0)`);const m=await call('Page.getLayoutMetrics'),image=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip:{x:0,y:0,width:m.cssContentSize.width,height:m.cssContentSize.height,scale:1}});fs.writeFileSync(path.join(screenshots,'rightmost-'+width+'.png'),Buffer.from(image.data,'base64'));}
 }
 // Trusted input must invalidate the previous successful result and exports.
 await js('document.getElementById("temperature").focus();document.getElementById("temperature").select()');await call('Input.insertText',{text:'2'});
 assert.equal(await js('DISTILL_TAIL_REPORT'),null);assert.equal(await js('document.getElementById("results").hidden'),true);assert.equal(await js('document.getElementById("download-report").disabled'),true);
 await js('document.getElementById("run").click()');assert.equal(await js('DISTILL_TAIL_REPORT.controls.temperature'),2);
 for(const text of ['{"model":1,"model":2}','{"x":NaN}',JSON.stringify({...data.examples[0].input,tail_shift:999}),JSON.stringify({...data.examples[0].input,alpha:2})]){await js(`document.getElementById('input').value=${JSON.stringify(text)};document.getElementById('input').dispatchEvent(new Event('input'));document.getElementById('run').click()`);assert.equal(await js('DISTILL_TAIL_REPORT'),null);assert.match(await js('document.getElementById("status").textContent'),/Refused/);}
 const trustedClick=async id=>{const rect=await js(`(()=>{const e=document.getElementById(${JSON.stringify(id)});e.scrollIntoView({block:'center'});const r=e.getBoundingClientRect();return{x:r.x+r.width/2,y:r.y+r.height/2};})()`);await call('Input.dispatchMouseEvent',{type:'mouseMoved',...rect});await call('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',clickCount:1,...rect});await call('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',clickCount:1,...rect});};
 let completed=0;for(const index of [0,data.examples.findIndex(c=>c.id==='true-excluded-tail-saturation'),data.examples.length-1]){
  await js(`document.getElementById('example').value=${index};document.getElementById('example').dispatchEvent(new Event('change'))`);const expected=await js('DISTILL_TAIL_REPORT');
  for(const id of ['download-input','download-report']){
   for(const f of fs.readdirSync(downloads))fs.unlinkSync(path.join(downloads,f));events.length=0;await trustedClick(id);
   for(let i=0;i<500&&!events.some(e=>e.method==='Browser.downloadProgress'&&e.params.state==='completed');i++)await delay(20);
   assert.ok(events.some(e=>e.method==='Browser.downloadProgress'&&e.params.state==='completed'),'actual download complete');const files=fs.readdirSync(downloads).filter(f=>!f.endsWith('.crdownload'));assert.equal(files.length,1);const file=path.join(downloads,files[0]),value=JSON.parse(fs.readFileSync(file));
   if(id==='download-input')equal(value,data.examples[index].input);else{equal(value.report,expected);const r=spawnSync(process.env.PYTHON||'python3',['-m','distilllab.tail',file,'--verify'],{cwd:path.resolve(__dirname,'..'),encoding:'utf8'});assert.equal(r.status,0,r.stderr);assert.equal(JSON.parse(r.stdout).verified,true);}completed++;
  }
 }
 console.log(JSON.stringify({examples:data.examples.length,variants:data.examples.length*4,layouts,classCells,parameterCells:paramCells,coordinateSelections:coordinates,actualCompletedDownloads:completed,nativeReplayedArchives:3,widths:[1280,390,320]}));
 }finally{if(ws)ws.close();if(browser?.pid){try{process.kill(-browser.pid,'SIGTERM');}catch(e){if(e.code!=='ESRCH')throw e;}if(browser.exitCode===null)await new Promise(r=>{browser.once('exit',r);setTimeout(r,3000);});}for(const d of [profile,downloads])await fs.promises.rm(d,{recursive:true,force:true,maxRetries:20,retryDelay:100});}
});

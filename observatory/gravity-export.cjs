'use strict';
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const context=vm.createContext({window:{}});context.window=context;
for(const file of ['data.js','gravity.js'])vm.runInContext(fs.readFileSync(path.join(__dirname,file),'utf8'),context);
const G=vm.runInContext('Gravity',context);
const a=JSON.parse(fs.readFileSync(0,'utf8'));
if(!['binary','triple'].includes(a.preparation)||!Number.isInteger(a.substeps)||a.substeps<1||a.substeps>3600)throw Error('Invalid preparation or substeps');
const s={...G.defaults,item:a.preparation};
for(const [key,v] of Object.entries(a.settings)){
 if(!Object.hasOwn(G.defaults,key))throw Error('Unknown gravity option '+key);
 if(key.startsWith('body')){if(!Object.hasOwn(G.composition,v))throw Error('Unknown composition');}
 else{const range=key.startsWith('radius')?[200,3000]:key.startsWith('density')?[.5,6]:key==='speed'?[.2,1.7]:[8,22];if(!Number.isFinite(v)||v<range[0]||v>range[1])throw Error('Parameter out of range: '+key);}
 s[key]=v;
}
process.stdout.write(JSON.stringify(G.integrate(s,a.substeps)));

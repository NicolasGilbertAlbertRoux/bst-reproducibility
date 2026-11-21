'use strict';
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),{spawnSync}=require('node:child_process');
const {createCanvas,GlobalFonts}=require('@napi-rs/canvas');
for(const weight of ['Regular','Bold']){
  GlobalFonts.registerFromPath(path.join(__dirname,'../projections/fonts/NotoSansDisplay-'+weight+'.ttf'),'Noto Sans Display');
}
function load(atlasOnly=false){global.window=global;global.pako=require('./vendor/inflate.js');for(const f of ['data.js','scenes.js','extended.js','films.js','gif.js','film.js','crystals.js','optics.js','optical-view.js','thermal-view.js','atlas-state.js','atlas-view.js','gravity.js','comparative-views.js','projections.js','projection-options.js'].filter(f=>!atlasOnly||f!=='films.js'))vm.runInThisContext(fs.readFileSync(path.join(__dirname,f),'utf8'),{filename:f});return vm.runInThisContext('Projections');}
async function save(P,s,file){const [w,h,n]=P.size(s);let result;if(s.view==='reference')result=Buffer.from(P.film(s).raw);else{const can=createCanvas(w,h),cx=can.getContext('2d'),encoder=createGifEncoder(w,h,s.domain==='sky'?44:9,OpticalView.swatches(s));for(let i=0;i<n;i++){P.render(can,s,i,createCanvas);encoder.addFrame(cx.getImageData(0,0,w,h).data);}result=Buffer.from(await encoder.finish().arrayBuffer());}fs.mkdirSync(path.dirname(path.resolve(file)),{recursive:true});fs.writeFileSync(file,result,{flag:'wx'});return {width:w,height:h,frames:n,bytes:result.length};}
function read(P,domain,item,args){return projectionSettings(P,domain,item,args);}
async function main(){const [d,id,file,...args]=process.argv.slice(2);if(!d||!id||!file)throw Error('Use DOMAIN ITEM NEW.gif [--view VIEW] [--mode N] [--band NM] [--language en|fr]');const legacy=['nuclear','spectrum'].includes(d)||(d==='field'&&/^\d+$/.test(id))||(d==='thermal'&&(id==='Xe'||args.includes('--mode')))||(d==='sky'&&args.includes('--band'));
 if(legacy){const p=spawnSync(process.execPath,[path.join(__dirname,'comparison-export.cjs'),...process.argv.slice(2)],{stdio:'inherit'});process.exitCode=p.status??1;return;}
 if(d==='atlas'&&id==='all'){
   const dest=path.resolve(file),parent=path.dirname(dest);fs.mkdirSync(parent,{recursive:true});
   if(fs.existsSync(dest))throw Error('Choose a new atlas directory');
   const tmp=fs.mkdtempSync(path.join(parent,'.neutral-atlas-'));
   try{const D=JSON.parse(fs.readFileSync(path.join(__dirname,'data.js'),'utf8').replace(/^window.BST_DATA = /,'').replace(/;\s*$/,''));const keys=D.elements.map(x=>x[1]);
     for(let j=0;j<keys.length;j+=4){const step=spawnSync(process.execPath,[__filename,'atlas-batch',keys.slice(j,j+4).join(','),tmp,...args],{stdio:'inherit'});if(step.status!==0)throw Error('Atomic batch failed at '+keys[j]);}
     fs.renameSync(tmp,dest);return;
   }finally{if(fs.existsSync(tmp))fs.rmSync(tmp,{recursive:true,force:true});}
 }
 const P=load(d==='atlas-batch'&&!args.includes('reference'));if(d==='atlas-batch'){for(const k of id.split(','))await save(P,read(P,'atom',k,args),path.join(file,k+'.gif'));return;}
 console.log(JSON.stringify(await save(P,read(P,d,id,args),file)));}
module.exports={load,read,save};if(require.main===module)main().catch(e=>{console.error(e.message);process.exitCode=1;});

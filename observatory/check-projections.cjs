'use strict';
const assert = require('node:assert/strict');
const {createHash} = require('node:crypto');
const {createCanvas} = require('@napi-rs/canvas');
const render = require('./render.cjs');
const [task,subject,selection]=process.argv.slice(2);
if(!task){
  const fs=require('node:fs'), path=require('node:path'), {spawnSync}=require('node:child_process');
  const run=(...args)=>{
    const output=spawnSync(process.execPath,[__filename,...args],{encoding:'utf8'});
    if(output.status!==0)throw Error('Projection check failed for '+args.join(' ')+' '+(output.stderr||output.signal));
    return JSON.parse(output.stdout);
  };
  const data=fs.readFileSync(path.join(__dirname,'data.js'),'utf8');
  const atoms=JSON.parse(data.slice(data.indexOf('{'),data.lastIndexOf('}')+1)).elements.map(x=>x[1]);
  const films=['beat:pair','atom:Kr','molecule:Kr2','molecule:C2'];
  let decoded=0,cases=0;
  for(const key of films)for(const lang of ['en','fr'])decoded+=run('film',key,lang);
  for(let i=0;i<atoms.length;i+=8)cases+=run('views','atom',atoms.slice(i,i+8).join(','));
  for(const d of ['beat','substrate','molecule','thermal','field','sky','gravity'])cases+=run('views',d);
  const physics=run('numerical');
  console.log(JSON.stringify({reference_films:8,decoded_reference_frames:decoded,animated_preparations_and_readings:cases,...physics}));
  process.exit(0);
}
const atlasOnly=task==='views'&&subject==='atom'&&!selection.split(',').includes('Kr');
const P=render.load(atlasOnly);

// These digests were obtained by independently decoding all frames with Pillow.
const record = require('../evidence/projection-checks.json');
const vm = require('node:vm');
const A = vm.runInThisContext('AtlasView'), G = vm.runInThisContext('Gravity');
function near(a,b,t) { assert.ok(Math.abs(a-b)<t, a+' vs '+b); }

function digest(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

const canvases=new Map();
function frame(state, index) {
  const [width, height] = P.size(state);
  const key=width+'×'+height;
  if(!canvases.has(key))canvases.set(key,createCanvas(width,height));
  const canvas=canvases.get(key);
  P.render(canvas, state, index, createCanvas);
  return digest(canvas.getContext('2d').getImageData(0, 0, width, height).data);
}

if(task==='film'){
    const [domain,item]=subject.split(':'),name=P.clips[subject],lang=selection;
    const state = {...P.baseState, domain, item, view:'reference',lang};
    const film=P.film(state), expected=(lang==='fr'?record.reference_films:record.english_reference_films)[name];
    const hash=createHash('sha256');
    for (let i=0;i<film.frames.length;i++) hash.update(film.read(i));
    assert.equal(film.frames.length,expected.frames);
    assert.equal(hash.digest('hex'),expected.decoded_rgba_sha256,name+'/'+lang);
    assert.equal(digest(film.raw),expected.gif_sha256);
    assert.equal(frame(state,0),digest(film.read(0)));
    console.log(film.frames.length);process.exit(0);
}

let cases = 0;
assert.equal(P.options('atom').length, 118);
if(task==='views'){
  const domain=subject;
  for (const item of selection?selection.split(','):P.options(domain)) {
    for (const view of P.readings(domain, item)) {
      const state = {...P.baseState, domain, item, view};
      const later = Math.min(15, P.size(state)[2] - 1);
      assert.notEqual(frame(state, 0), frame(state, later), [domain, item, view].join('/'));
      cases++;
    }
  }
  console.log(cases);process.exit(0);
}

const carbon = {...P.baseState, domain: 'molecule', item: 'C2', view: 'molecular'};
assert.equal(new Set([473.7, 516.5, 563.5].map(band => frame({...carbon, band}, 15))).size, 3);
assert.equal(new Set(['fine', 'molecular', 'ensemble'].map(view => frame({...carbon, view}, 15))).size, 3);

const substrate = {...P.baseState, domain: 'substrate', item: 'material', view: 'profile'};
for (const quantity of ['shape', 'scalar', 'charge']) {
  const values = new Set();
  for (const rigidity of [1, 2, 4]) {
    for (const inertia of [4, 8, 16]) {
      values.add(frame({...substrate, quantity, rigidity, inertia}, 12));
    }
  }
  assert.ok(values.size > 1, quantity);
}
// Neutral counts, display phase and archive coverage.
for (const [z,symbol,period,archived,total,outer] of BST_DATA.elements) {
  const m=A.model(symbol);assert.equal(m.points.length,total);assert.equal(m.count,total);
  assert.equal(m.surface.reduce((v,n)=>v+n,0),outer);
  assert.equal(m.sourced.reduce((v,n)=>v+n,0),m.sourceCount);
  if(z>36){assert.equal(m.sourceCount,outer);assert.ok(m.points.some((p,i)=>!m.surface[i]&&Math.hypot(...p)<.60));}
  near(A.beatRadius(m,0)/A.beatRadius(m,.5),1.28/.72,1e-12);
}
assert.equal(A.model('Og').count,783);assert.equal(A.model('Kr').count,297);
assert.equal(BST_DATA.elements.filter(r=>r[3]!==null).length,86);
assert.equal(BST_DATA.isotopes.length,5842);assert.equal(BST_DATA.ions.filter(r=>r[1]>0).length,7021);
assert.ok(BST_DATA.isotopes.every(r=>r[7]===null));assert.ok(BST_DATA.ions.every(r=>r[4]===null));
for(const [k,ratio] of [['H2O',31/66],['CO2',66/76],['SF6',57/105]]){
  const g=A.geometry({...P.baseState,item:k,mode:1},.25);
  near(g.r[1]/g.r[0],ratio,1e-12);
  assert.ok(g.pos.every(p=>p.every(Number.isFinite)));
}
const sky={...P.baseState,domain:'sky',view:'map'};
const c=createCanvas(640,376);
function residual(key) {
  const label=P.render(c,{...sky,item:key},0,createCanvas);
  const pixels=c.getContext('2d').getImageData(425,116,202,112).data;
  return [Number(label.match(/RMS\(ΔT\) = ([0-9.]+)/)[1]),digest(pixels)];
}
const skies=Object.keys(BST_DATA.sky).map(residual);
assert.equal(skies[0][0],0);assert.ok(skies.slice(1).every(r=>r[0]>0));assert.ok(new Set(skies.map(r=>r[1])).size>=3);
const magnetic={...P.baseState,domain:'field',item:'SF6',view:'induction',mode:11};
const f=mode=>{P.render(c,{...magnetic,mode},8,createCanvas);return digest(c.getContext('2d').getImageData(340,80,278,225).data);};
assert.notEqual(f(11),f(1));
const initial={...P.baseState,item:'binary'};
const orbit=G.integrate(initial),refined=G.integrate(initial,120),heavy=G.integrate({...initial,body1:'SF6'}),large=G.integrate({...initial,radius1:2*initial.radius1-200});
near(large.bodies[0].mass/orbit.bodies[0].mass,((2*initial.radius1-200)/initial.radius1)**3,1e-11);
assert.ok(heavy.bodies[0].mass>orbit.bodies[0].mass);assert.notDeepEqual(heavy.records[1].xy,orbit.records[1].xy);
assert.equal(orbit.records.length,refined.records.length);
const drift=orbit.records.at(-1).drift;
assert.ok(drift<1e-3 && refined.records.at(-1).drift<drift);
let centreError=0,orbitDifference=0;
for(const [i,r] of orbit.records.entries()){
  for(let k=0;k<2;k++) {
    const cm=r.xy.reduce((v,p,j)=>v+p[k]*orbit.bodies[j].mass,0)/orbit.bodies.reduce((v,b)=>v+b.mass,0);
    centreError=Math.max(centreError,Math.abs(cm));
  }
  for(let j=0;j<2;j++)orbitDifference=Math.max(orbitDifference,Math.hypot(...r.xy[j].map((x,k)=>x-refined.records[i].xy[j][k])));
}
assert.ok(centreError<1e-4 && orbitDifference<2000);
const triple=G.integrate({...initial,item:'triple'}),last=triple.records.at(-1);
assert.ok(last.contact);
const [body1,body2]=last.contact.map(i=>i-1);
near(Math.hypot(...last.xy[body1].map((x,k)=>x-last.xy[body2][k])),triple.bodies[body1].radius+triple.bodies[body2].radius,1e-5);
console.log(JSON.stringify({neutral_counts:118,distinct_sky_residuals:new Set(skies.map(r=>r[1])).size,default_binary_max_energy_drift:drift,centre_of_mass_error_m:centreError,step_halving_max_difference_m:orbitDifference,default_triple_contact_s:last.t}));

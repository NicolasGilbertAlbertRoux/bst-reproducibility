'use strict';
const assert = require('node:assert/strict');
const {createHash} = require('node:crypto');
const {createCanvas} = require('@napi-rs/canvas');
const {runInThisContext} = require('node:vm');
const {load, read} = require('./render.cjs');

const P = load(true);
const atlas = runInThisContext('AtlasState');
const thermal = runInThisContext('ThermalView');
const optical = runInThisContext('OpticalView');
const canvas = createCanvas(640, 376);
function crop(s, frame, box) {
  const [w,h] = P.size(s);
  if(canvas.width!==w)canvas.width=w;
  if(canvas.height!==h)canvas.height=h;
  P.render(canvas, s, frame, createCanvas);
  return createHash('sha256').update(canvas.getContext('2d').getImageData(...box).data).digest('hex');
}
const atom = (...args) => read(P, 'atom', 'He', args);
const cases = [
  [atom('--isotope', '3'), 1, 2],
  [atom('--isotope', '4'), 2, 2],
  [atom('--isotope', '4', '--charge', '1'), 2, 1],
  [atom('--charge', '2'), null, 0]
];
const bars = new Set();
for (const [s, neutrons, electrons] of cases) {
  const a = atlas.resolve(s);
  assert.equal(a.n, neutrons); assert.equal(a.e, electrons); assert.equal(a.z, 2);
  assert.equal(a.count, null); assert.equal(a.reference, 13);
  assert.equal(P.size(s)[2], 1); assert.equal(s.view, 'state');
  bars.add(crop(s, 0, [140, 119, 288, 89]));
}
assert.equal(bars.size, cases.length);
assert.equal(P.size(atom())[2], 72);
assert.throws(() => atom('--isotope', '99'));
assert.throws(() => atom('--charge', '3'));
assert.throws(() => atom('--isotope', '4', '--view', 'atlas'));
assert.throws(() => atom('--nuclear-state', 'm'));
const state=(item, view, ...args) => read(P, 'thermal', item, ['--view', view, ...args]);
const box = [100, 62, 440, 218];
const structures = new Set();
for (const [item, n, degree, atoms] of [['H2O',72,4,3],['CO2',108,12,3],['SF6',54,8,7],['Kr',108,12,1]]) {
  const s = state(item, 'solid');
  const {sites, model} = thermal.sample(s, 0);
  assert.equal(model.n, n); assert.equal(model.degree, degree); assert.equal(sites.length, n * atoms);
  assert.equal(model.frames, 72); assert.ok(model.relativeEnergySpan < .005);
  const mobile = thermal.sample(s, 8).sites;
  assert.ok(sites.some((v,i) => Math.hypot(...v.point.map((p,k)=>p-mobile[i].point[k])) > .002));
  const angles=[[0,0],[67,0],[67,37],[0,90]];
  assert.equal(new Set(angles.map(([angle,elevation])=>crop({...s,angle,elevation},8,box))).size,angles.length);
  structures.add(crop(s,8,box));
  const shell={...s,view:'shell'};
  const local=thermal.sample(shell,8);
  assert.equal(local.centres.length,degree+1); assert.equal(local.sites.length,(1+degree)*atoms);
  assert.deepEqual(local.centres[0],[0,0,0]);
  assert.notEqual(crop(s,8,box),crop(shell,8,box));
  for (const view of ['solid','shell','gas','liquid','ordered-lj']) {
    const c=state(item,view);
    assert.equal(c.colour,'spectral');
    assert.equal(P.size(c)[1],527);
    const motion=state(item,view,'--colour','motion');
    assert.equal(P.size(motion)[1],376);
    assert.equal(crop(c,8,box),crop(motion,8,box));
    assert.equal(state(item,view,'--colour','composition').colour,'motion');
    const expected=item==='H2O'&&['solid','shell','liquid'].includes(view)?'transmission':
      view==='gas'?(item==='Kr'?'lines':'infrared'):'unknown';
    assert.equal(optical.reading(c).kind,expected);
  }
}
assert.equal(structures.size,4);
for(const view of ['solid','shell','liquid']){
  const readings=[];
  for(const distance of optical.paths){
    const c=state('H2O',view,'--path-length',String(distance));
    const {record}=optical.reading(c);
    crop(c,0,box);
    const rgb=Array.from(canvas.getContext('2d').getImageData(110,451,1,1).data).slice(0,3);
    assert.deepEqual(rgb,record.rgb);
    assert.deepEqual(Array.from(canvas.getContext('2d').getImageData(42,451,1,1).data).slice(0,3),[255,255,255]);
    assert.ok(record.transmission.every(x=>x>=0&&x<=1));
    readings.push(record);
  }
  assert.deepEqual(readings[0].rgb,[255,255,255]);
  assert.equal(readings[0].relative_Y,1);
  assert.ok(readings.every((r,i)=>!i||r.relative_Y<readings[i-1].relative_Y));
  assert.ok(readings[4].rgb[2]-readings[4].rgb[0]>60);
}
assert.notDeepEqual(optical.reading(state('H2O','solid')).record.rgb,optical.reading(state('H2O','liquid')).record.rgb);
for(const item of ['H2O','CO2','SF6']){
  const m=read(P,'molecule',item,['--view','molecular']);
  assert.equal(optical.reading(m).kind,'infrared');
  assert.ok(optical.reading(m).bands.every(x=>x>2&&x<18));
  assert.equal(P.size({...m,view:'fine'})[1],376);
}
assert.throws(()=>state('H2O','gas','--path-length','1'));
assert.throws(()=>state('H2O','liquid','--path-length','-1'));
assert.throws(()=>state('H2O','liquid','--colour','motion','--path-length','1'));
const iodine = read(P,'molecule','I2',['--view','molecular']);
const iodineModel = runInThisContext('AtlasView');
const maximum = iodineModel.geometry(iodine,0);
const minimum = iodineModel.geometry(iodine,.5);
assert.deepEqual(maximum.atoms,['I','I']);
assert.deepEqual(maximum.r,[1.39,1.39]);
assert.equal(maximum.freq,214.502);
assert.ok(Math.abs(maximum.pos[1][0]-minimum.pos[0][0]-2.6663)<1e-12);
let previousY = Infinity;
for(const distance of optical.paths){
  const s=read(P,'molecule','I2',['--view','molecular','--path-length',String(distance)]);
  const {record}=optical.reading(s);
  crop(s,12,box);
  assert.deepEqual(Array.from(canvas.getContext('2d').getImageData(110,451,1,1).data).slice(0,3),record.rgb);
  assert.ok(record.relative_Y<previousY);
  previousY=record.relative_Y;
  if(distance===0)assert.deepEqual(record.rgb,[255,255,255]);
  if(distance===1)assert.ok(record.rgb[1]<record.rgb[0]-70 && record.rgb[1]<record.rgb[2]-70);
}
assert.equal(P.size({...iodine,view:'fine'})[1],376);
assert.throws(()=>read(P,'molecule','I2',['--view','fine','--path-length','1']));
const norms = [1,2,3];
for(const [az,el] of [[0,0],[31,-81],[156,32],[360,90]]) {
  assert.ok(Math.abs(Math.hypot(...thermal.camera(az,el)(norms))-Math.hypot(...norms)) < 1e-12);
}
console.log('Passed isotope/ion counts and independent frame export settings, four crystal motifs, two camera axes and phase-specific spectroscopy including six water and ice path lengths.');

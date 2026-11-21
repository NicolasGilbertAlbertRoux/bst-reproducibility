'use strict';
const AtlasView = (() => {
  const loaded = new Map();
  const sprites = new Map();
  const TAU = 2*Math.PI;
  const radii = BST_DATA.projectionInputs.covalentRadiusA;
  function model(symbol) {
    if (loaded.has(symbol)) return loaded.get(symbol);
    const m = BST_EXTENDED.atlas[symbol];
    const bytes = pako.inflate(bytes64(m.raw));
    const rows = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength/4);
    const points=[], u=[], surface=[], sourced=[];
    for (let i=0; i<rows.length; i+=8) {
      points.push(rows.slice(i,i+3)); u.push(rows.slice(i+3,i+6)); surface.push(rows[i+6]); sourced.push(rows[i+7]);
    }
    const result={...m,points,u,surface,sourced};
    loaded.set(symbol,result); return result;
  }
  function beatRadius(m, phase) { return .46*m.spacing*(1+.28*Math.cos(TAU*phase)); }
  function sprite(kind, make) {
    if (sprites.has(kind)) return sprites.get(kind);
    const tile=make(56,56), c=tile.getContext('2d'), g=c.createRadialGradient(28,28,0,28,28,28);
    const halo=kind.startsWith('mantle');

    const rgb=halo?(kind==='mantle/grey'?'232,232,232':'162,206,206'):(kind==='wave/grey'?'244,244,244':'244,245,240');
    const stops=halo?[[0,.30],[.40,.27],[.73,.16],[1,0]]:[[0,.025],[.48,.075],[.68,.15],[.79,.87],[.89,.39],[1,0]];
    for (const [f,a] of stops) g.addColorStop(f,'rgba('+rgb+','+a+')');
    c.fillStyle=g; c.fillRect(0,0,56,56); sprites.set(kind,tile);return tile;
  }
  function camera(p, angle) {
    const a=angle*Math.PI/180, x=p[0]*Math.cos(a)+p[2]*Math.sin(a),z=-p[0]*Math.sin(a)+p[2]*Math.cos(a);
    const e=.38; return [x,p[1]*Math.cos(e)+z*Math.sin(e),z*Math.cos(e)-p[1]*Math.sin(e)];
  }
  function cloud(ctx, symbol, center, radius, phase, state, make, orientation=0) {
    const m=model(symbol), cs=Math.cos(TAU*phase), sn=Math.abs(Math.sin(TAU*phase));
    const length=radius/(1+.46*m.spacing), glyph=length*beatRadius(m,phase);
    const native=state.view==='mode', coarse=state.view==='ensemble', atom=state.domain==='atom';
    const ring=sprite(state.domain==='molecule'?'wave/grey':'wave',make), tint=sprite(state.domain==='molecule'?'mantle/grey':'mantle',make);
    const projected=m.points.map((p,j)=>{
      const v=m.u[j], xyz=camera([p[0]+.095*cs*v[0],p[1]+.095*cs*v[1],p[2]+.095*cs*v[2]],state.angle+orientation);
      return [xyz[0],xyz[1],xyz[2],Math.hypot(...v),m.surface[j],m.sourced[j]];
    }).sort((a,b)=>a[2]-b[2]);
    ctx.save();
    for(const [x,y,z,v,surf] of projected) {
      const px=center[0]+length*x, py=center[1]-length*y, depth=.60+.33*z;
      if(coarse || state.view==='atlas' || (!atom && state.view!=='fine')){
        const r=length*m.spacing*(coarse?1.40:1.06);
        ctx.globalAlpha=Math.min(.85,(.18+.52*sn)*depth*(surf?1:.55)*(state.view==='atlas'?.28:1));
        ctx.drawImage(tint,px-r,py-r,2*r,2*r);
      }
      if(!coarse){
        const r=native?length*.46*m.spacing:glyph;
        // A faint position marker remains readable at a turning point. Brightness from motion goes to zero there.
        ctx.globalAlpha=Math.min(.95,depth*((native?.085:.105)+(native?.85*v:.83)*sn)*(atom||state.view==='fine'?1:.49));
        ctx.drawImage(ring,px-r,py-r,2*r,2*r);
      }
    }
    ctx.restore();
  }
  function geometry(s, phase) {
    const q=Math.cos(TAU*phase);let pos, atoms, freq, unit;
    if(s.item==='SF6'){
      const a=BST_DATA.fluoride,j=s.mode-1,v=a.vectors[j],amp=.12*q/(Math.max(...v.map(Math.abs))||1);
      atoms=['S',...Array(6).fill('F')];pos=[[0,0,0]];
      for(let i=0;i<6;i++){
        const ray=a.base[i].map((x,k)=>x+a.frame[i][0][k]*(a.angles[2*i]+amp*v[2*i])+a.frame[i][1][k]*(a.angles[2*i+1]+amp*v[2*i+1]));
        pos.push(ray.map(x=>x*(a.radii[i]+amp*v[12+i])*a.scale/Math.hypot(...ray)));
      }
      freq=a.frequencies[j];unit=s.lang==='fr'?'ω (modèle)':'ω (model)';
    }else if(s.item==='I2'){
      const separation=2.6663+.06*q;
      atoms=['I','I'];pos=[[-separation/2,0,0],[separation/2,0,0]];
      freq=214.502;unit='cm⁻¹';
    }else if(s.item==='H2O'){
      let a=104.52*Math.PI/360;const d=[.9578,.9578];if(s.mode===1)a+=.095*q;
      if(s.mode===2||s.mode===3){d[0]+=.075*q;d[1]+=.075*q*(s.mode===2?1:-1);}
      atoms=['O','H','H'];pos=[[0,0,0],[-d[0]*Math.sin(a),-d[0]*Math.cos(a),0],[d[1]*Math.sin(a),-d[1]*Math.cos(a),0]];
      freq=[1595,3657,3756][s.mode-1];unit='cm⁻¹';
    }else{
      atoms=['C','O','O'];pos=s.mode===1?[[0,0,0],[-1.162,.11*q,0],[1.162,.11*q,0]]:[[0,0,0],[-1.162+.09*q,0,0],[1.162+.09*q,0,0]];
      freq=s.mode===1?667:2349;unit='cm⁻¹';
    }
    const mass=atoms.map(x=>({H:1.008,C:12.011,O:15.999,F:18.998403,S:32.06,I:126.90447})[x]);
    const total=mass.reduce((x,y)=>x+y),cm=[0,1,2].map(k=>pos.reduce((a,p,j)=>a+p[k]*mass[j],0)/total);
    return {pos:pos.map(p=>p.map((x,k)=>x-cm[k])),atoms,freq,unit,r:atoms.map(x=>radii[x])};
  }
  function bar(c,x,y,scale,label) {
    c.save();c.lineWidth=1;c.strokeStyle='#9eb9bc';c.beginPath();c.moveTo(x,y-3);c.lineTo(x,y+3);c.moveTo(x,y);c.lineTo(x+scale,y);c.lineTo(x+scale,y-3);c.lineTo(x+scale,y+3);c.stroke();
    c.fillStyle='#aec3c4';c.font='12px sans-serif';c.fillText(label,x,y-7);c.restore();
  }
  function molecule(c,s,i,make,center=[320,183],scale=68){
    const t=i/72+.14, g=geometry(s,t);
    const sites=g.pos.map((p,j)=>[...camera(p,s.angle),j]).sort((a,b)=>a[2]-b[2]);
    for(const [x,y,z,j] of sites)cloud(c,g.atoms[j],[center[0]+scale*x,center[1]-scale*y],scale*g.r[j],t,s,make,j*31);
    if(s.domain==='molecule')bar(c,27,307,scale,'1 Å');
    return g;
  }
  return {model,beatRadius,cloud,camera,geometry,molecule,bar};
})();

'use strict';
const Gravity = (() => {
  const composition={H2O:{H:2,O:1},CO2:{C:1,O:2},SF6:{S:1,F:6},Kr:{Kr:1}};
  const defaults={radius1:1600,radius2:850,radius3:420,body1:'H2O',body2:'CO2',body3:'Kr',density1:2.5,density2:2.5,density3:2.5,separation:14,speed:1};
  const source=BST_DATA.projectionInputs;
  function prepare(s){
    const n=s.item==='binary'?2:3, bodies=[];
    const x=s.separation*1e6;
    const locations=[[-.33*x,0],[.67*x,0],[.28*x,1.18*x]],v=[[0,-70*s.speed],[0,151*s.speed],[-118*s.speed,2*s.speed]];
    for(let i=0;i<n;i++){
      const species=s['body'+(i+1)],radius=s['radius'+(i+1)]*1000,numberDensity=s['density'+(i+1)]*1e28;
      const molecularMass=Object.entries(composition[species]).reduce((a,[el,k])=>a+k*source.nuclideMassU[el],0)*source.atomicMassKg;
      const count=4*Math.PI/3*radius**3*numberDensity;
      bodies.push({species,radius,count,mass:count*molecularMass,x:locations[i].slice(),v:v[i].slice()});
    }
    const mass=bodies.reduce((s,b)=>s+b.mass,0);
    for(const prop of ['x','v'])for(let k=0;k<2;k++){
      const mean=bodies.reduce((s,b)=>s+b.mass*b[prop][k],0)/mass;
      for(const b of bodies)b[prop][k]-=mean;
    }
    return bodies;
  }
  function acceleration(b){
    const result=b.map(()=>[0,0]);
    for(let i=0;i<b.length;i++)for(let j=i+1;j<b.length;j++){
      const dx=b[j].x[0]-b[i].x[0],dy=b[j].x[1]-b[i].x[1],g=source.gravityG/Math.hypot(dx,dy)**3;
      for(const [k,d] of [dx,dy].entries()){result[i][k]+=b[j].mass*g*d;result[j][k]-=b[i].mass*g*d;}
    }return result;
  }
  function energy(b){
    let e=b.reduce((s,a)=>s+.5*a.mass*(a.v[0]**2+a.v[1]**2),0);
    for(let i=0;i<b.length;i++)for(let j=i+1;j<b.length;j++)e-=source.gravityG*b[i].mass*b[j].mass/Math.hypot(b[i].x[0]-b[j].x[0],b[i].x[1]-b[j].x[1]);
    return e;
  }
  function firstContact(b, trial){
    let first=null;
    for(let i=0;i<b.length;i++)for(let j=i+1;j<b.length;j++){
      const d=b[j].x.map((v,k)=>v-b[i].x[k]),u=trial[j].map((v,k)=>v-trial[i][k]-d[k]);
      const a=u[0]**2+u[1]**2, dot=d[0]*u[0]+d[1]*u[1],c=d[0]**2+d[1]**2-(b[i].radius+b[j].radius)**2;
      const disc=dot**2-a*c;if(a===0||disc<0||dot>=0)continue;
      const t=(-dot-Math.sqrt(disc))/a;
      if(t>=0&&t<=1&&(!first||t<first.f))first={f:t,pair:[i+1,j+1]};
    }return first;
  }
  function integrate(s,substeps=60){
    const b=prepare(s), records=[],initial=energy(b),dt=3600/substeps;let t=0,err=0,hit=null,acc=acceleration(b);
    function sample(){records.push({t,xy:b.map(o=>o.x.slice()),drift:err,contact:hit?.pair??null});}
    sample();
    for(let h=0;h<96&&!hit;h++){
      for(let k=0;k<substeps&&!hit;k++){
        const trial=b.map((o,i)=>o.x.map((x,j)=>x+dt*(o.v[j]+.5*dt*acc[i][j])));
        hit=firstContact(b,trial);const f=hit?hit.f:1,step=dt*f;
        b.forEach((o,i)=>o.x=o.x.map((x,j)=>x+f*(trial[i][j]-x)));
        const next=acceleration(b);b.forEach((o,i)=>o.v=o.v.map((v,j)=>v+.5*step*(acc[i][j]+next[i][j])));
        t+=step;acc=next;err=Math.max(err,Math.abs((energy(b)-initial)/initial));
      }sample();
    }
    return {bodies:b.map(({species,mass,radius,count})=>({species,mass,radius,count})),records};
  }
  let prior='',series;
  function history(s){
    const key=JSON.stringify([s.item,...Object.keys(defaults).map(k=>s[k])]);
    if(key!==prior){series=integrate(s);prior=key;}return series;
  }
  function draw(c,s,i){
    const sim=history(s);
    i=Math.min(Math.max(0,Math.trunc(i)),sim.records.length-1);
    const record=sim.records[i],palette=['#75ddd0','#ffb775','#a2b9f0'];
    const extent=Math.max(s.separation*1.50e6,...sim.records.flatMap(r=>r.xy.map(p=>Math.max(Math.abs(p[0])*.56,Math.abs(p[1])))));
    const scale=124/extent, X=(x)=>320+x*scale,Y=(x)=>177-x*scale;
    for(let b=0;b<sim.bodies.length;b++){
      c.beginPath();for(let j=0;j<=i;j++){const p=sim.records[j].xy[b];if(j===0)c.moveTo(X(p[0]),Y(p[1]));else c.lineTo(X(p[0]),Y(p[1]));}
      c.strokeStyle=palette[b];c.lineWidth=1.25;c.stroke();
      const [x,y]=record.xy[b],xx=X(x),yy=Y(y),rad=sim.bodies[b].radius*scale;
      const g=c.createRadialGradient(xx-rad*.34,yy-rad*.34,0,xx,yy,rad);g.addColorStop(0,'#fff7e1');g.addColorStop(.36,palette[b]);g.addColorStop(1,'#334b55');
      c.fillStyle=g;c.beginPath();c.arc(xx,yy,rad,0,2*Math.PI);c.fill();
      c.fillStyle=palette[b];c.font='13px sans-serif';c.fillText((b+1)+' · '+sim.bodies[b].species,xx+rad+4,yy-3);
    }
    AtlasView.bar(c,22,313,5e6*scale,'5 000 km');
    return (record.contact?(s.lang==='fr'?'Contact des corps ':'Surface contact ')+record.contact.join('–')+'   ·   ':'')+'t = '+(record.t/3600).toFixed(2)+' h   ·   |ΔE/E₀|max = '+record.drift.toExponential(2);
  }
  return {defaults,composition,history,integrate,draw};
})();

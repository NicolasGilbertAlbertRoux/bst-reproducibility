'use strict';
const ThermalView = (() => {
  const models = window.BST_CRYSTALS;
  const buffers = new Map();
  const names = {
    H2O:['Hexagonal ice Ih', 'Glace hexagonale Ih'],
    CO2:['Dry ice I · Pa-3', 'Glace carbonique I · Pa-3'],
    SF6:['Plastic SF₆ · Im-3m', 'SF₆ plastique · Im-3m'],
    Kr:['fcc krypton · Fm-3m', 'Krypton CFC · Fm-3m']
  };
  const tr=(s,en,fr)=>s.lang==='fr'?fr:en;
  function crystalName(s) {return names[s.item][s.lang==='fr'?1:0];}
  function number(s,x,digits) {return Number(x).toLocaleString(s.lang,{maximumFractionDigits:digits,useGrouping:false});}
  function decoded(key,raw) {
    if(!buffers.has(key)){
      const bytes=pako.inflate(bytes64(raw));
      buffers.set(key,new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4));
    }
    return buffers.get(key);
  }
  function camera(azimuth,elevation) {
    const a=azimuth*Math.PI/180,b=elevation*Math.PI/180;
    const c=Math.cos(a),s=Math.sin(a),h=Math.cos(b),v=Math.sin(b);
    return ([x,y,z])=>[x*c+z*s,y*h+(z*c-x*s)*v,(z*c-x*s)*h-y*v];
  }
  function oriented(p,axes){
    return [0,1,2].map(k=>p[0]*axes[0][k]+p[1]*axes[1][k]+p[2]*axes[2][k]);
  }
  function fluidAxes(j,t){
    // Rigid silhouettes rotate for display. The LJ trajectories contain only centres.
    const a=j*2.399963+t*(.17+.10*(j%4)), b=j*1.837+.11*t;
    const ca=Math.cos(a),sa=Math.sin(a),cb=Math.cos(b),sb=Math.sin(b);
    return [[ca,0,-sa],[sa*sb,cb,ca*sb],[sa*cb,-sb,ca*cb]];
  }
  function sample(s,i){
    const shell=s.view==='shell',crystal=s.view==='solid'||shell;
    const m=crystal?models[s.item]:BST_EXTENDED.thermal[s.view==='ordered-lj'?'solid':s.view];
    const b=decoded(crystal?'crystal/'+s.item:'lj/'+s.view,crystal?m.motion:m.raw);
    const geometry=s.item==='Kr'?{atoms:['Kr'],pos:[[0,0,0]]}:AtlasView.geometry({...s,mode:1},.25);
    const radii=BST_DATA.projectionInputs.covalentRadiusA;
    const sites=[],centres=[];
    const selection=shell?m.shell:Array.from({length:m.n},(_,j)=>j);
    const origin=shell?m.centers[m.shell[0]].map((p,d)=>p+b[4*(i*m.n+m.shell[0])+d]):[0,0,0];
    for(const j of selection){
      const k=4*(i*m.n+j);
      // The archived LJ reporter used 5.1 Å per reduced length as a graphical convention.
      const centre=crystal?m.centers[j].map((v,d)=>v+b[k+d]):[0,1,2].map(d=>(b[k+d]-.5)*m.length*5.1);
      for(let d=0;d<3;d++)centre[d]-=origin[d];
      centres.push(centre);
      const axes=crystal?m.axes[j]:fluidAxes(j,m.values[i][0]);
      for(let n=0;n<geometry.atoms.length;n++){
        const symbol=geometry.atoms[n];
        const local=geometry.pos[n].map((v,d)=>v-(crystal?geometry.pos[0][d]:0));
        const offset=oriented(local,axes);
        sites.push({point:centre.map((v,d)=>v+offset[d]),symbol,radius:radii[symbol],speed:b[k+3]});
      }
    }
    return {sites,centres,model:m,crystal,shell};
  }
  function legend(ctx,s){
    ctx.save();ctx.font='12px Noto Sans Display, sans-serif';ctx.fillStyle='#d7e4e3';
    ctx.fillText(s.domain==='molecule'?tr(s,'Grey · mantle motion','Gris · mouvement du manteau'):tr(s,'Grey · centre speed','Gris · vitesse des centres'),18,326);ctx.restore();
  }
  function render(c,s,i){
    const {sites,centres,model:m,crystal,shell}=sample(s,i);
    const projection=camera(s.angle,s.elevation);
    // Equal horizontal, vertical and depth units. Zoom stays fixed while the camera rotates.
    const scale=shell?20:crystal?8.0:125/(Math.sqrt(3)*m.length*5.1/2+2.2);
    const list=sites.map(v=>({...v,p:projection(v.point)})).sort((a,b)=>a.p[2]-b.p[2]);
    c.save();c.beginPath();c.rect(0,48,640,259);c.clip();
    if(shell){
      c.strokeStyle='rgba(185,185,185,.52)';c.lineWidth=1;c.setLineDash([3,4]);
      c.beginPath();
      for(const centre of centres.slice(1)){const [x,y]=projection(centre);c.moveTo(320,177);c.lineTo(320+x*scale,177-y*scale);}
      c.stroke();c.setLineDash([]);
    }
    for(const {radius,speed,p} of list){
      const x=320+p[0]*scale,y=177-p[1]*scale,r=radius*scale;
      const motion=crystal?Math.min(1,speed/.24):Math.min(1,speed/2.5);
      const depth=Math.max(.55,Math.min(1,.79+p[2]*scale/445));
      const light=(.58+.42*motion)*depth;
      const grad=c.createRadialGradient(x,y,0,x,y,r);
      for(const [frac,alpha] of [[0,.36],[.48,.65],[.72,1],[1,0]])grad.addColorStop(frac,'rgba(232,232,232,'+alpha*light+')');
      c.fillStyle=grad;c.fillRect(x-r,y-r,2*r,2*r);
    }
    c.restore();
    c.save();c.font='13px Noto Sans Display, sans-serif';c.fillStyle='#bfd1d1';
    const phase=crystal?crystalName(s):s.view==='ordered-lj'?tr(s,'Ordered LJ comparator','Comparateur LJ ordonné'):tr(s,'LJ centre dynamics','Dynamique LJ des centres');
    c.fillText(phase,19,47,585);
    c.restore();
    AtlasView.bar(c,548,298,(shell?2:5)*scale,shell?'2 Å':crystal?'5 Å':tr(s,'5 Å · display','5 Å · rendu'));
    legend(c,s);
    if(shell)return tr(s,'One central site · first neighbours: ','Un site central · premiers voisins : ')+m.degree+' · '+tr(s,'reduced time','temps réduit')+' '+number(s,i*m.step,1);
    if(crystal)return m.n+' '+tr(s,'sites','sites')+' · '+tr(s,'neighbours','voisins')+' '+m.degree+' · '+tr(s,'reduced time','temps réduit')+' '+number(s,i*m.step,1)+' · '+tr(s,'harmonic motion','mouvement harmonique');
    const v=m.values[i];
    return 'T* = '+number(s,v[1],3)+'   ·   ρ* = '+number(s,m.density,3)+'   ·   P* = '+number(s,v[3],3)+'   ·   MSD / σ² = '+number(s,v[2],3);
  }
  return {render,crystalName,camera,sample,legend};
})();

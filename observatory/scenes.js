"use strict";
const D=window.BST_DATA, tau=2*Math.PI;
function beatHistory(stiffness, sign, mass, restoring) {
  const h=.004, radius=.8, coupling=.4;
  let d=2,b=0,p=0,u=sign*Math.sqrt(2*mass*.30);
  const rows=[];
  function forces(){const overlap=Math.max(0,2*(radius+b)-d);return [-coupling/Math.max(.05,d)**2+stiffness*overlap,-restoring*b-2*stiffness*overlap];}
  for(let i=0;i<=5000;i++){
    if(i%25===0){const overlap=Math.max(0,2*(radius+b)-d);rows.push([i*h,d,b,overlap,p*p/2+u*u/(2*mass)-coupling/d+restoring*b*b/2+stiffness*overlap*overlap/2,u/mass,p]);}
    if(i===5000)break;
    let f=forces();p+=h*f[0]/2;u+=h*f[1]/2;d+=h*p;b+=h*u/mass;
    f=forces();p+=h*f[0]/2;u+=h*f[1]/2;
  }
  return rows;
}
function project(point,angle){const a=angle*Math.PI/180, c=Math.cos(a),s=Math.sin(a),z=point[2]||0;return [point[0]*c+z*s,point[1]*.94+(-point[0]*s+z*c)*.342];}
function norm(v){return Math.hypot(...v);}
function seriesPlot(ctx, records, cursor, names, units, error=null){
  const left=66, right=683, top=85, bottom=357, all=records.flatMap(r=>r.slice(1,1+names.length));
  if(error!==null)records.forEach(r=>all.push(r[1]-r[error],r[1]+r[error]));
  let lo=Math.min(...all),hi=Math.max(...all);const pad=(hi-lo||1)*.08;lo-=pad;hi+=pad;
  const x=t=>left+(right-left)*(t-records[0][0])/(records.at(-1)[0]-records[0][0]||1);
  const y=v=>bottom-(bottom-top)*(v-lo)/(hi-lo);
  ctx.font="12px sans-serif";ctx.lineWidth=1;ctx.strokeStyle="#313c3a";ctx.fillStyle="#bdc9c5";ctx.textAlign="right";
  for(let g=0;g<5;g++){const v=lo+(hi-lo)*g/4;ctx.beginPath();ctx.moveTo(left,y(v));ctx.lineTo(right,y(v));ctx.stroke();ctx.fillText(Number(v.toPrecision(3)).toString(),left-6,y(v)+4);}
  ctx.textAlign="left";ctx.fillText(units,left,top-10);
  if(error!==null){ctx.strokeStyle="#8a9490";records.forEach(r=>{ctx.beginPath();ctx.moveTo(x(r[0]),y(r[1]-r[error]));ctx.lineTo(x(r[0]),y(r[1]+r[error]));ctx.stroke();});}
  const colors=['#ebeeed','#74d7ae','#e8b87a','#81aeea'];
  names.forEach((label,j)=>{ctx.lineWidth=j===0?1.4:2;ctx.strokeStyle=colors[j];ctx.beginPath();records.forEach((r,i)=>{if(i===0)ctx.moveTo(x(r[0]),y(r[j+1]));else ctx.lineTo(x(r[0]),y(r[j+1]));});ctx.stroke();ctx.fillStyle=colors[j];ctx.fillText(label,left+j*164,395);});
  const i=Math.min(records.length-1,Math.round(cursor*(records.length-1))),r=records[i];
  ctx.strokeStyle='#9daba4';ctx.setLineDash([3,4]);ctx.beginPath();ctx.moveTo(x(r[0]),top);ctx.lineTo(x(r[0]),bottom);ctx.stroke();ctx.setLineDash([]);
  for(let j=0;j<names.length;j++){ctx.fillStyle=colors[j];ctx.beginPath();ctx.arc(x(r[0]),y(r[j+1]),3.4,0,tau);ctx.fill();}
  ctx.fillStyle='#bdc9c5';ctx.fillText(String(+records[0][0].toPrecision(5)),left,bottom+17);ctx.textAlign="right";ctx.fillText(String(+records.at(-1)[0].toPrecision(5)),right,bottom+17);ctx.textAlign="left";return r;
}
function line(ctx,x,y,a,b,color='#89938e'){ctx.strokeStyle=color;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(a,b);ctx.stroke();}
function dot(ctx,x,y,r,a=1){ctx.fillStyle='rgba(242,245,244,'+Math.min(1,Math.max(.025,a))+')';ctx.beginPath();ctx.arc(x,y,Math.max(.6,r),0,tau);ctx.fill();}
function ring(ctx,x,y,r,alpha){
  const outer=r+7, gradient=ctx.createRadialGradient(x,y,0,x,y,outer);
  const q=Math.max(.1,Math.min(.96,r/outer));gradient.addColorStop(0,'transparent');gradient.addColorStop(Math.max(0,q-.11),'transparent');gradient.addColorStop(q,'rgba(252,252,252,'+Math.min(1,.2+.78*alpha)+')');gradient.addColorStop(1,'transparent');
  ctx.fillStyle=gradient;ctx.beginPath();ctx.arc(x,y,outer,0,tau);ctx.fill();
}
function fixed(value,digits=4){return Number(value).toLocaleString('en-US',{maximumSignificantDigits:digits,useGrouping:false});}
function drawScene(canvas,st,phase){
  const ctx=canvas.getContext('2d'); ctx.save();ctx.scale(canvas.width/720,canvas.height/440);ctx.fillStyle='#000';ctx.fillRect(0,0,720,440);
  const tr=(a,b)=>st.lang==='fr'?b:a; const title=st.title;
  ctx.fillStyle='#e9f1ec';ctx.font='bold 21px sans-serif';ctx.fillText(title,22,31,660);ctx.font='13px sans-serif';ctx.fillStyle='#adbbb4';
  const metrics=[];let footer='';
  const add=(label,v)=>metrics.push([label,v]);
  function points(pos,signals,links=[],magnification=1,labels=[]){
    const pp=pos.map(p=>project(p,st.angle));const scale=137/(magnification||1);
    const xy=pp.map(([x,y])=>[360+x*scale,218-y*scale]);
    links.forEach(([i,j])=>line(ctx,...xy[i],...xy[j],'#66756e'));
    xy.forEach(([x,y],i)=>{const a=signals[i]||0;ring(ctx,x,y,5+8*Math.min(1,Math.abs(a)),Math.min(1,Math.abs(a)));dot(ctx,x,y,1.8,.8);
      if(labels[i]){ctx.fillStyle='#c7d1cc';ctx.fillText(labels[i],x+12,y-8);}});
  }
  if(st.domain==='beat'){
    const row=st.history[Math.round(phase*(st.history.length-1))], [,d,b,o,,v]=row;
    const centers=[[-d/2,0,0],[d/2,0,0]].map(p=>project(p,st.angle));const scale=84;
    for(const p of centers){const x=360+p[0]*scale,y=210-p[1]*scale;ring(ctx,x,y,(.8+b)*scale,Math.min(1,Math.abs(v)/1.2));dot(ctx,x,y,1.7);}
    add('t',fixed(row[0]));add(tr('separation','séparation'),fixed(d));add(tr('radius','rayon'),fixed(.8+b));add(tr('overlap','recouvrement'),fixed(o));add('E',fixed(row[4],6));
    footer=(v>=0?tr('Dilation','Dilatation'):tr('Contraction','Contraction'))+'  ·  t = '+fixed(row[0])+'  ·  '+tr('model units','unités du modèle');
  }else if(st.domain==='atom'){
    const a=D.atoms[st.item],m=D.modalScenes[st.item];
    if(a){const max=Math.max(...a.mode.map(Math.abs)),q=Math.cos(tau*phase),local=a.mode.map(e=>e/max);
      const pos=a.points.map((p,i)=>p.map(k=>k*(1+.12*local[i]*q)));
      points(pos,local.map(e=>e*Math.sin(tau*phase)),[],Math.max(...a.points.map(norm))*1.13);
      add('λ',fixed(a.eigenvalue,6));add(tr('shell sites','sites de surface'),a.points.length);add(tr('multiplicity','multiplicité'),a.multiplicity);
      footer=tr('Radial display of a scalar surface-contact eigenmode · phase in cycles','Vue radiale d’un mode scalaire de contact · phase en cycles');
    }else if(m?.mode){const max=Math.max(...m.mode.map(norm)),q=Math.cos(tau*phase),radius=Math.max(...m.points.map(norm));
      const pos=m.points.map((p,i)=>p.map((x,j)=>x+.1*radius*m.mode[i][j]/max*q));points(pos,m.mode.map(v=>norm(v)*Math.abs(Math.sin(tau*phase))/max),[],1.14*radius);
      add('ω',fixed(m.omega));footer=tr('Archived neutral mode · displacement enlarged','Mode neutre archivé · déplacement amplifié');
    }else{ctx.fillStyle='#c6d1ca';ctx.font='20px sans-serif';ctx.fillText(tr('See the complete animated atlas','Voir l’atlas animé complet'),190,220);footer=tr('No archived mode for this measurement','Aucun mode archivé pour cette mesure');}
  }else if(st.domain==='molecule'&&st.item==='SF6'){
    const a=D.fluoride,j=st.mode-1, u=a.vectors[j], v=Math.max(...u.map(Math.abs)); const amp=.12*Math.cos(tau*phase)/v;
    const pos=[[0,0,0]];const signals=[0];for(let i=0;i<6;i++){
      const dir=a.base[i].map((x,k)=>x+a.frame[i][0][k]*(a.angles[2*i]+amp*u[2*i])+a.frame[i][1][k]*(a.angles[2*i+1]+amp*u[2*i+1]));
      const r=(a.radii[i]+amp*u[12+i])*a.scale;pos.push(dir.map(q=>r*q/norm(dir)));signals.push(Math.abs(Math.sin(tau*phase))*Math.hypot(u[2*i],u[2*i+1],u[12+i])/v);
    }
    points(pos,signals,[[0,1],[0,2],[0,3],[0,4],[0,5],[0,6]],1.65,['S','F','F','F','F','F','F']);
    add(tr('mode','mode'),st.mode+'/66');add('ω',fixed(a.frequencies[j]));add(tr('phase','phase'),fixed(phase));footer=tr('Normal mode about the recorded equilibrium · display amplitude enlarged','Mode normal autour de l’équilibre archivé · amplitude visuelle amplifiée');
  }else if(st.domain==='molecule'){
    const a=D.modalScenes[st.item],j=st.mode-1,v=a.currents[j],max=Math.max(...v.map(norm))||1, q=Math.sin(tau*phase);
    const r=Math.max(...a.points.map(norm));
    points(a.points,v.map(p=>norm(p)*Math.abs(q)/max),[],r||1);
    a.points.forEach((p,i)=>{let [x,y]=project(p,st.angle),[vx,vy]=project(v[i],st.angle);x=360+x*137/r;y=218-y*137/r;line(ctx,x,y,x+vx*66*q/max,y-vy*66*q/max,'#d8e0dc');});
    add(tr('current mode','mode de courant'),j+1);add('ω / ω₀',fixed(a.frequencies[j]));footer=tr('Archived site currents · fixed site positions · grayscale measures current magnitude','Courants de site archivés · positions fixes · blanc selon la norme du courant');
  }else if(st.domain==='field'){
    const a=D.field[Number(st.item)-1],j=st.mode-1, records=a.t.map((t,i)=>[t,...a.powers[i].map(p=>p[j])]);const r=seriesPlot(ctx,records,phase,['n₁','n₂','n₃','n₄'],tr('projected power · native units','puissance projetée · unités natives'));add('t',fixed(r[0]));add('k',D.wavenumbers[j]);footer=tr('Time at four observer directions · transverse readout','Temps dans quatre directions de visée · lecture transverse');
  }else if(st.domain==='sky'){
    const rows=D.sky[st.item][st.band].map(([ell,obs,sigma,model,ref])=>[ell,obs,model,ref,sigma]);const r=seriesPlot(ctx,rows,phase,['Planck','BST',tr('Planck best fit','Ajustement Planck')],'Dℓ (µK²)',4);
    add('ℓ',fixed(r[0]));add('BST',fixed(r[2]));add('Δ / σ',fixed((r[2]-r[1])/r[4]));footer=tr('Multipole scan · diagonal band errors','Balayage en multipôle · écarts-types diagonaux par bande');
  }else if(st.domain==='thermal'){
    if(['H2O','CO2'].includes(st.item)){
      const w=st.item==='H2O',[lo,hi,A,B,C]=w?[274,303,5.40221,1838.675,-31.737]:[154.26,195.89,6.81228,1301.679,-3.494];
      const rows=Array.from({length:101},(_,i)=>{const t=lo+(hi-lo)*i/100;return [t,10**(A-B/(t+C))];});const r=seriesPlot(ctx,rows,phase,['NIST'], 'P (bar)');add('T (K)',fixed(r[0]));add('P (bar)',fixed(r[1]));
      footer=w?tr('NIST · liquid and vapour coexistence','NIST · coexistence liquide–vapeur'):tr('NIST · solid and vapour coexistence','NIST · coexistence solide–vapeur');
    }else{const k=st.mode-1, rows=D.thermal['0'].filter(r=>r[0]===st.item).sort((a,b)=>a[1]-b[1]).map(r=>[r[1],r[2][k],r[3][k]]);const u=['P (Pa)','ρᵥ (mol/m³)','ρₗ (mol/m³)','Δh (J/mol)'][k];const r=seriesPlot(ctx,rows,phase,['BST','HEOS'],u);add('T (K)',fixed(r[0]));add('BST',fixed(r[1]));add('HEOS',fixed(r[2]));footer=tr('Coexistence along temperature · branch 0','Coexistence selon la température · branche 0');}
  }else if(st.domain==='nuclear'){
    const a=D.beta.find(r=>r[0]===st.item), rows=Array.from({length:101},(_,i)=>[i/25,2**(-i/25)]);
    const r=seriesPlot(ctx,rows,phase,['N / N₀'], 'N / N₀');add('t / t½',fixed(r[0]));add('t½ (s)',fixed(a[2],8));add(tr('daughter','noyau fils'),a[1]);footer=tr('Population law using the evaluated half-life','Population calculée à partir de la demi-vie évaluée');
  }else if(st.domain==='spectrum'){
    if(st.item==='SF6'){const a=D.infrared[st.band];const rows=a.map(p=>[p[0],p[1]*1e20]);const r=seriesPlot(ctx,rows,phase,['PTB'], 'σ (10⁻²⁰ cm² / molecule)');add('ν̃ (cm⁻¹)',fixed(r[0],7));add('λ (µm)',fixed(10000/r[0]));footer=tr('Infrared absorption · no visible hue','Absorption infrarouge · aucune teinte visible');}
    else if(st.item==='H'){
      const rows=D.hydrogen.map(r=>[299792458e9/r[1],r[1],r[2]]).sort((a,b)=>a[0]-b[0]);
      const selected=rows[Math.min(rows.length-1,Math.floor(phase*rows.length))];
      const spacing=rows[1][0]-rows[0][0],lo=rows[0][0]-spacing*.45,hi=rows[1][0]+spacing*.45;
      rows.forEach(([w])=>{const x=75+(w-lo)/(hi-lo)*575;ctx.fillStyle=wavelengthDisplay(w);ctx.fillRect(x-3,120,6,185);ctx.fillStyle='#cfdbd4';ctx.fillText(w.toFixed(6)+' nm',x-43,335);});
      const cursor=75+(selected[0]-lo)/(hi-lo)*575;line(ctx,cursor-11,104,cursor+11,104,'#f7fbf9');
      add('λ (nm)',selected[0].toFixed(7));add('ν (Hz)',String(selected[1]));add('u(ν) (Hz)',selected[2]);
      footer=tr('H 2S–4P · equal marker heights · approximate display hue','H 2S–4P · hauteurs conventionnelles · teinte d’affichage approchée');
    }
    else{const data=D.optical.filter(r=>r.observation.startsWith(st.item+' ')),i=Math.min(data.length-1,Math.floor(phase*data.length));const max=Math.max(...data.map(r=>r.mesure_nW));data.forEach((r,j)=>{const x=120+j*175,h=r.mesure_nW/max*150;ctx.fillStyle=j===i?'#eaf2ee':'#758a7f';ctx.fillRect(x,300-h,75,h);ctx.fillStyle='#d6ded9';ctx.fillText(r.observation,x,326);});add('nW',fixed(data[i].mesure_nW));footer=tr('Measured band-integrated power · wavelength distribution inside each band is unresolved','Puissance mesurée par bande · distribution interne non résolue');}
  }
  ctx.fillStyle='#becbc5';ctx.font='14px sans-serif';ctx.fillText(footer,22,426,680);ctx.restore();return metrics;
}

function wavelengthDisplay(w){
  // Piecewise Gaussian fits to the CIE 1931 matching functions (Wyman et al., 2013).
  const g=(mid,left,right)=>Math.exp(-.5*((w-mid)*(w<mid?left:right))**2);
  const x=1.056*g(599.8,.0264,.0323)+.362*g(442,.0624,.0374)-.065*g(501.1,.049,.0382);
  const y=.821*g(568.8,.0213,.0247)+.286*g(530.9,.0613,.0322);
  const z=1.217*g(437,.0845,.0278)+.681*g(459,.0385,.0725);
  const rgb=[3.2406*x-1.5372*y-.4986*z,-.9689*x+1.8758*y+.0415*z,.0557*x-.204*y+1.057*z].map(v=>Math.max(0,v));
  const peak=Math.max(...rgb);return 'rgb('+rgb.map(v=>{v/=peak;return Math.round(255*(v<=.0031308?12.92*v:1.055*v**(1/2.4)-.055));}).join(',')+')';
}

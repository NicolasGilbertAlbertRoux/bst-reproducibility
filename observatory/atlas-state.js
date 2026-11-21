'use strict';
const AtlasState=(()=>{
  const tr=(s,en,fr)=>s.lang==='fr'?fr:en;
  const bySymbol=new Map();
  function rows(symbol){
    if(!bySymbol.has(symbol)){
      const element=BST_DATA.elements.find(r=>r[1]===symbol);
      if(!element)throw Error('Unknown element');
      const z=element[0];
      bySymbol.set(symbol,{element,nuclei:BST_DATA.isotopes.filter(r=>r[0]===z),charges:BST_DATA.ions.filter(r=>r[0]===z)});
    }
    return bySymbol.get(symbol);
  }
  function defaultNucleus(symbol){
    const {nuclei}=rows(symbol);
    return nuclei.find(r=>r[3]&&r[2]==='gs')||nuclei.find(r=>r[2]==='gs')||nuclei[0];
  }
  function resolve(s){
    const {element,nuclei,charges}=rows(s.item),z=element[0];
    const nucleus=s.isotope?nuclei.find(r=>r[1]===s.isotope&&r[2]===s.nuclearState):null;
    if(s.isotope&&!nucleus)throw Error('Unknown nuclide or nuclear state for '+s.item);
    const ion=charges.find(r=>r[1]===s.charge);
    if(!ion)throw Error('Unknown charge for '+s.item);
    const count=nucleus&&s.charge?null:nucleus?nucleus[7]:ion[4];
    const id=(nucleus?s.item+'-'+nucleus[1]:' '+s.item).trim()+(s.charge?'  +'+s.charge:'')+(nucleus&&nucleus[2]!=='gs'?'  ('+nucleus[2]+')':'');
    return {id,z,n:nucleus?nucleus[1]-z:null,e:ion[2],nucleus,ion,count,
            reference:element[3]??element[4],conditional:element[3]===null,
            scale:Math.max(z,...nuclei.map(a=>a[1]-z))};
  }
  const format=(s,n)=>Number(n).toLocaleString(s.lang,{useGrouping:false,maximumSignificantDigits:9});
  function details(s,a){
    const n=a.nucleus, out=[];
    if(n){
      out.push(n[2]==='gs'?tr(s,'Nuclear ground state','État nucléaire fondamental'):tr(s,'Nuclear state ','État nucléaire ')+n[2]);
      if(n[8]!=null)out.push(tr(s,'neutral-atom mass','masse de l’atome neutre')+' '+format(s,n[8])+' u'+(n[6]?tr(s,' (estimated)',' (estimée)'):''));
      if(n[9])out.push('Jπ '+n[9]);
      if(n[10]!=null)out.push('E* '+format(s,n[10])+' keV');
      out.push(n[3]?tr(s,'stable','stable'):n[4]!=null?tr(s,'half-life ','demi-vie ')+format(s,n[4])+' s'+(n[5]?tr(s,' (estimated)',' (estimée)'):''):tr(s,'half-life unavailable','demi-vie indisponible'));
    }
    if(s.charge)out.push(a.ion[3]!=null?tr(s,'cumulative ionization ','ionisation cumulée ')+format(s,a.ion[3])+' eV':tr(s,'cumulative ionization energy unavailable','énergie cumulée indisponible'));
    return out.join(' · ');
  }
  function draw(c,s){
    const a=resolve(s);
    c.save();c.fillStyle='#e4eeee';c.font='bold 24px Noto Sans Display, sans-serif';c.fillText(a.id,32,80);
    c.font='15px Noto Sans Display, sans-serif';c.fillStyle='#bed0d0';c.fillText(tr(s,'Composition of the selected state','Composition de l’état sélectionné'),32,109);
    const bars=[[tr(s,'Protons','Protons'),a.z,'#efa967'],[tr(s,'Neutrons','Neutrons'),a.n,'#68bec0'],[tr(s,'Electrons','Électrons'),a.e,'#9eabeb']];
    for(let j=0;j<bars.length;j++){
      const [label,num,colour]=bars[j], y=140+j*29;
      c.fillStyle='#dce6e6';c.fillText(label,32,y+4);c.fillStyle='#203536';c.fillRect(140,y-6,288,8);
      if(num!==null){c.fillStyle=colour;c.fillRect(140,y-6,288*num/a.scale,8);}
      c.fillStyle='#e7eded';c.fillText(num===null?tr(s,'unspecified','non précisé'):String(num),444,y+4);
    }
    c.font='bold 18px Noto Sans Display, sans-serif';c.fillStyle='#e7ecec';
    c.fillText(tr(s,'BST waves in this state','Ondes BST de cet état')+' · '+(a.count??tr(s,'undetermined','indéterminé')),32,252);
    c.font='13px Noto Sans Display, sans-serif';c.fillStyle='#aabbbb';
    c.fillText(tr(s,'Separate neutral element reference','Repère élémentaire neutre distinct')+' · '+a.reference+' '+tr(s,'sites','sites')+(a.conditional?tr(s,' (conditional)',' (conditionnel)'):''),32,280);
    c.fillText(details(s,a),32,313,570);c.restore();
    return a.id+' · Z = '+a.z+(a.nucleus?' · A = '+a.nucleus[1]:'')+' · q = '+(s.charge?'+'+s.charge:'0');
  }
  return {rows,defaultNucleus,resolve,details,draw};
})();

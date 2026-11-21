'use strict';
const OpticalView=(()=>{
  const tr=(s,en,fr)=>s.lang==='fr'?fr:en;
  const number=(s,value,digits=2)=>value.toLocaleString(s.lang,{useGrouping:false,maximumFractionDigits:digits});
  const infrared={H2O:[1594.59,3656.65,3755.79],CO2:[667.38,2349.16],SF6:[615.5,947.5]};
  const krLines=[557.02894,587.09160];
  const paths=BST_OPTICS.water.map(r=>r.path_m);

  function supported(s){
    return s.domain==='thermal'||(s.domain==='molecule'&&(s.item==='I2'||Object.hasOwn(infrared,s.item))&&s.view!=='fine');
  }
  function cell(s){
    if(s.domain==='molecule'&&s.item==='I2'&&s.view!=='fine')return 'iodine';
    if(s.domain!=='thermal'||s.item!=='H2O')return null;
    if(s.view==='liquid')return 'water';
    if(s.view==='solid'||s.view==='shell')return 'ice';
    return null;
  }
  function reading(s){
    const medium=cell(s);
    if(medium){
      const record=BST_OPTICS[medium].find(x=>x.path_m===s.pathM);
      if(!record)throw Error('Unsupported optical path');
      if(medium==='iodine'){
        const density=number(s,BST_OPTICS.iodine_number_density_cm3/1e15)+' × 10¹⁵ cm⁻³';
        return {kind:'transmission',record,title:tr(s,'Daylight through iodine vapour','Lumière du jour à travers la vapeur de diiode'),
          subtitle:'I₂ · 295 K · '+density+' · '+pathName(s,s.pathM)+' · D65 · 380–700 nm',
          description:tr(s,'Measured gas absorption. The colour is integrated over the spectrum of the entire cell at fixed exposure. The grey pair above illustrates vibration.','Absorption mesurée en phase gazeuse. La couleur est intégrée sur le spectre de toute la cuve à exposition fixe. La paire grise illustre la vibration.')};
      }
      const name=medium==='water'?tr(s,'pure liquid water, 22 °C','eau liquide pure, 22 °C'):tr(s,'clear ice Ih, 266 K','glace Ih limpide, 266 K');
      return {kind:'transmission',record,title:tr(s,'Transmitted daylight','Lumière du jour transmise'),
        subtitle:name+' · D65 · '+number(s,s.pathM)+' m · 380–700 nm',
        description:tr(s,'Bulk sample under daylight D65. The optical path is independent of the simulation box. Absorption only, without scattering or boundary reflections.','Échantillon éclairé par la lumière du jour D65. Le trajet optique est indépendant de la taille de la boîte simulée. Seule l’absorption est prise en compte, hors diffusion et reflets de surface.')};
    }
    const gas=s.domain==='molecule'||s.view==='gas';
    if(gas&&s.item==='Kr'){
      return {kind:'lines',title:tr(s,'Emission of excited gaseous Kr I','Émission de Kr I gazeux excité'),
        subtitle:tr(s,'Two measured lines in air · separate excitation reference','Deux raies mesurées dans l’air · excitation de référence distincte'),
        description:tr(s,'Each filter shows the hue of one measured atomic line in excited gas. Peak display intensities are normalised separately. The temperature or motion of the model does not set the excitation.','Chaque filtre donne la teinte d’une raie atomique mesurée dans le gaz excité. Les éclats sont normalisés séparément pour l’écran. L’excitation est une condition distincte du mouvement et de la température simulés.')};
    }
    if(gas&&Object.hasOwn(infrared,s.item)){
      const bands=infrared[s.item].map(v=>10000/v).sort((a,b)=>a-b);
      return {kind:'infrared',bands,title:tr(s,'Gas-phase infrared absorption','Absorption infrarouge en phase gazeuse'),
        subtitle:tr(s,'Measured band positions','Positions de bandes mesurées')+' · '+bands.map(v=>number(s,v)).join(' / ')+' µm',
        description:tr(s,'These bands lie outside vision. The marks locate measured wavelengths with no assumption about their relative strengths. A visible emission spectrum for this preparation is unavailable.','Ces bandes sont hors du visible. Les traits situent les longueurs d’onde mesurées sans présumer leurs intensités relatives. Le spectre d’émission visible de cette préparation est indisponible.')};
    }
    const dry=s.item==='CO2'&&['solid','shell'].includes(s.view);
    return {kind:'unknown',title:tr(s,'Visible spectrum','Spectre visible'),subtitle:dry?tr(s,'Clear CO₂ ice · absorption below the detection limit','Glace de CO₂ limpide · absorption sous le seuil de détection'):tr(s,'No quantitative visible spectrum for this preparation','Aucun spectre visible quantitatif pour cette préparation'),
      description:dry?tr(s,'In the transmission experiment on clear CO₂ ice, visible absorption was below the detection limit. These data do not resolve a hue.','Dans l’expérience de transmission sur glace de CO₂ limpide, l’absorption visible est sous le seuil de détection. Ces données ne distinguent aucune teinte.'):tr(s,'The phase and excitation must match the optical data to infer an appearance. The neutral silhouettes show the calculated motion.','La phase et l’excitation doivent correspondre aux données optiques pour en tirer un aspect visible. Les silhouettes grises montrent le mouvement calculé.')};
  }
  function pathName(s,value){
    return value>0&&value<1?number(s,100*value)+' cm':number(s,value)+' m';
  }
  function swatch(c,x,y,width,rgb){
    c.fillStyle='rgb('+rgb.join(',')+')';c.fillRect(x,y,width,27);
    c.strokeStyle='#9ba3a6';c.lineWidth=.7;c.strokeRect(x,y,width,27);
  }
  function text(c,x,y,label,size=14,colour='#dce2e2'){
    c.font=size+'px Noto Sans Display, sans-serif';c.fillStyle=colour;c.fillText(label,x,y,604);
  }
  function draw(c,s){
    const a=reading(s);c.save();c.fillStyle='#10191c';c.fillRect(0,376,640,151);
    c.fillStyle='#344347';c.fillRect(16,376,608,1);
    text(c,18,402,a.title,19);text(c,18,425,a.subtitle,14,'#c8d2d3');
    if(a.kind==='transmission'){
      const r=a.record; swatch(c,18,442,55,[255,255,255]);swatch(c,73,442,98,r.rgb);
      text(c,18,488,'D65',12);text(c,77,488,tr(s,'Through sample','Transmis'),12);
      const x=252,y=444,w=353,h=40;
      c.strokeStyle='#54636a';c.strokeRect(x,y,w,h);c.beginPath();
      r.transmission.forEach((v,i)=>{const xx=x+w*i/(r.transmission.length-1),yy=y+h*(1-v);if(i)c.lineTo(xx,yy);else c.moveTo(xx,yy);});
      c.strokeStyle='#f3f5f5';c.lineWidth=1.8;c.stroke();
      text(c,x-44,y+4,'100 %',12);text(c,x-31,y+h,'0 %',12);
      text(c,x,502,'380 nm',12);text(c,x+w-45,502,'700 nm',12);
      text(c,18,515,tr(s,'Luminance','Luminance')+' '+number(s,100*r.relative_Y,1)+' %',13);
    }else if(a.kind==='lines'){
      krLines.forEach((nm,i)=>{const x=18+188*i;swatch(c,x,441,150,BST_OPTICS.lines[String(nm)]);text(c,x,489,number(s,nm)+' nm');});
      text(c,18,515,tr(s,'Unexcited gas appears colourless · separate emission filters','Gaz non excité incolore · filtres d’émission distincts'),13);
    }else if(a.kind==='infrared'){
      const left=25,width=570,y=462;
      c.strokeStyle='#839196';c.beginPath();c.moveTo(left,y);c.lineTo(left+width,y);c.stroke();
      c.strokeStyle='#f0f1f2';c.lineWidth=2;c.beginPath();
      a.bands.forEach(v=>{const x=left+(v-2)*width/16;c.moveTo(x,y-17);c.lineTo(x,y+3);});c.stroke();
      text(c,left,486,'2 µm',12);text(c,left+width-35,486,'18 µm',12);
      text(c,18,513,tr(s,'Ordinary gas appears colourless. These bands are infrared.','Le gaz ordinaire paraît incolore. Ces bandes sont infrarouges.'),14);
    }else{
      text(c,18,465,tr(s,'The available data do not determine a visible hue.','Les données disponibles ne déterminent pas de teinte visible.'),16);
      text(c,18,497,tr(s,'The image above shows motion in greyscale.','La figure au-dessus montre le mouvement en gris.'),14);
    }
    c.restore();
  }
  function summary(s){
    const a=reading(s);
    const luminance=a.kind==='transmission'?tr(s,' Relative luminance ',' Luminance relative ')+number(s,100*a.record.relative_Y,2)+' %.':'';
    const wavelengths=a.kind==='lines'?' '+krLines.map(w=>number(s,w,5)+' nm').join(' / ')+'.':'';
    const appearance=['infrared','lines'].includes(a.kind)?tr(s,' The unexcited gas appears colourless under ordinary white light.',' Sous un éclairage blanc ordinaire, le gaz non excité paraît incolore.'):'';
    return a.title+' · '+a.subtitle+'.'+luminance+wavelengths+appearance+' '+a.description;
  }
  function swatches(s){
    if(s.domain==='molecule'&&s.item==='C2'&&s.view!=='fine')return [BST_OPTICS.lines[String(s.band)]];
    if(s.colour!=='spectral'||!supported(s))return [];
    const a=reading(s);
    if(a.kind==='transmission')return [a.record.rgb];
    if(a.kind==='lines')return krLines.map(nm=>BST_OPTICS.lines[String(nm)]);
    return [];
  }
  return {supported,cell,reading,draw,pathName,paths,swatches,summary};
})();

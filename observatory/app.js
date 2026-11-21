"use strict";
const el=id=>document.getElementById(id);
const state={lang:'en',domain:'beat',item:'pair',mode:1,angle:24,band:'TT',phase:0,k:3,mass:.3,restoring:2,sign:1};
const titles={beat:['Primordial beat','Battement primordial'],atom:['Atomic atlas','Atlas atomique'],molecule:['Molecular motion','Mouvement moléculaire'],spectrum:['Spectral observations','Observations spectrales'],field:['Current and polarization','Courant et polarisation'],thermal:['Thermal states','États thermiques'],sky:['Cosmological comparison','Comparaison cosmologique'],nuclear:['Beta decay','Désintégration bêta']};
const notes={
beat:['Two compressible mantles interact in the reduced Hamiltonian model. The settings integrate a new trajectory from the same initial separation.','Deux manteaux déformables interagissent dans le modèle hamiltonien réduit. Les réglages intègrent une nouvelle trajectoire à partir de la même séparation.','A white rim locates each mantle. Its brightness tracks the magnitude of radial speed in both directions. Time, radius and energy are expressed in native model units.','Un liseré blanc situe chaque manteau. Sa luminosité suit la valeur absolue de la vitesse radiale dans les deux sens. Temps, rayon et énergie sont en unités du modèle.'],
atom:['This benchmark tab shows the archived operators. Open Simulations for all 118 neutral animations.','Cet onglet présente les opérateurs du banc. Les 118 animations neutres sont dans Simulations.','For Z = 37–118, the lowest positive mode of the archived scalar contact operator is drawn radially on the surface directions. Its scale and radial embedding are display choices. Argon uses an archived vector mode. Counts attached to a neutral reference do not identify an isotope-specific wave.','Pour Z = 37–118, le plus bas mode positif de l’opérateur de contact scalaire est représenté radialement sur les directions de surface. L’échelle et la représentation radiale sont des choix graphiques. L’argon utilise un mode vectoriel archivé. Le décompte associé à une référence neutre ne donne pas celui d’un isotope.'],
molecule:['Rotate the preparation and select a single normal mode or current mode.','Faites pivoter la préparation et sélectionnez un mode normal ou un mode de courant.','SF₆ uses all six directions, the equilibrium radii and the 66-dimensional mass and stiffness operators from the reviewed benchmark. H₂O and CO₂ replay site currents retained as earlier illustrations. Their arrows are currents at three fixed sites. A white halo encodes motion. Visible emission has no spatial calibration here.','SF₆ utilise les six directions, les rayons à l’équilibre et les opérateurs de masse et de raideur à 66 dimensions du banc relu. H₂O et CO₂ reprennent les courants archivés comme illustrations antérieures. Leurs flèches indiquent le courant de trois sites fixes. Le halo blanc code le mouvement. L’émission visible ne dispose ici d’aucun étalonnage spatial.'],
spectrum:['Select a spectral observation and move the cursor across its recorded band.','Sélectionnez une observation et déplacez le curseur dans la bande enregistrée.','SF₆ displays infrared absorption at 296.1 K and nominal 225.5 Torr. The graph retains the local extrema in each display bin. H shows the measured hydrogen 2S–4P frequency doublet in approximate display color. The gas panels show photometric band integrals. Neither data product fixes an RGB emission field at the molecule.','SF₆ présente l’absorption infrarouge à 296,1 K et à une pression nominale de 225,5 Torr. Le graphique conserve les extrema locaux de chaque classe graphique. H présente le doublet 2S–4P de l’hydrogène mesuré, en couleur d’affichage approchée. Les panneaux des gaz présentent des intégrales photométriques par bande. Aucune de ces mesures ne détermine un champ d’émission RVB à l’échelle moléculaire.'],
field:['Compare four transverse projections from a single preparation.','Comparez quatre projections transverses issues d’une même préparation.','The four directions and the four native wave numbers index recorded complex amplitudes. The plotted observable is their squared norm at nine sampled times. These traces do not locate magnetic field lines.','Les quatre directions et les quatre nombres d’onde natifs indexent les amplitudes complexes enregistrées. La grandeur tracée est leur norme au carré à neuf instants. Ces lectures ne localisent pas des lignes de champ magnétique.'],
thermal:['Read the equilibrium state as temperature changes.','Suivez l’état d’équilibre quand la température change.','Kr and Xe compare a BST Helmholtz branch with the HEOS reference at coexistence. H₂O and CO₂ are external NIST Antoine reference curves, respectively for liquid–vapour and solid–vapour equilibrium. The cursor changes temperature, independently of elapsed time.','Kr et Xe comparent une branche de Helmholtz BST à la référence HEOS à coexistence. H₂O et CO₂ sont des courbes externes d’Antoine du NIST, respectivement pour les équilibres liquide–vapeur et solide–vapeur. Le curseur parcourt les températures, indépendamment du temps écoulé.'],
sky:['Select a cosmological family and one of the TT, TE and EE band spectra.','Choisissez une famille cosmologique et un spectre en bandes TT, TE ou EE.','White marks the Planck band value, green the computed family, amber the binned official best fit. The bars are diagonal band uncertainties. The cursor scans angular multipoles. The model parameters are fixed for each family.','Le blanc donne Planck, le vert la famille calculée et l’ocre l’ajustement officiel en bandes. Les barres figurent les incertitudes diagonales. Le curseur parcourt les multipôles angulaires. Les paramètres sont fixés pour chaque famille.'],
nuclear:['Read the survival fraction of a population over four half-lives.','Lisez la fraction survivante d’une population sur quatre demi-vies.','The exponential curve uses the evaluated neutral-parent half-life stored in the catalogue. State-conditioned BST amplitude intervals and measured lifetimes are distinguished in the article. Individual events and intermediate shape changes are unresolved.','La courbe exponentielle utilise la demi-vie évaluée du parent neutre inscrite au catalogue. L’article distingue les intervalles d’amplitude BST par état et les durées mesurées. Les événements individuels et les déformations intermédiaires ne sont pas résolus.']};
const L=(a,b)=>state.lang==='fr'?b:a;
function pick(x){return x[state.lang==='fr'?1:0];}
function opts(select,arr,saved){select.replaceChildren();for(const entry of arr){const [v,label]=Array.isArray(entry)?entry:[entry,entry];select.add(new Option(label,v));}if(arr.some(x=>String(Array.isArray(x)?x[0]:x)===String(saved)))select.value=saved;}
function addSelect(parent,title,options,selected,fn){
  const label=document.createElement('label'),text=document.createElement('span'),s=document.createElement('select');text.textContent=title;label.append(text,s);parent.append(label);opts(s,options,selected);s.onchange=()=>fn(s.value);return s;
}
function slider(node,label,key,min,max,step){
  const wrap=document.createElement('label'),text=document.createElement('span'),output=document.createElement('output'),range=document.createElement('input');wrap.append(text,range);text.textContent=label;text.append(output);range.type='range';Object.assign(range,{min,max,step,value:state[key]});output.value=state[key];range.oninput=()=>{state[key]=Number(range.value);output.value=range.value;if(key!=='angle')recalculate();display();};node.append(wrap);
}
function recalculate(){state.history=beatHistory(state.k,state.sign,state.mass,state.restoring);}
function items(domain){switch(domain){
 case 'beat':return [['pair',L('Mantle pair','Paire de manteaux')]];
 case 'atom':return D.elements.map(e=>[e[1],e[0]+' · '+e[1]]);
 case 'molecule':return ['SF6','H2O','CO2'];
 case 'field':return D.field.map((_,i)=>[String(i+1),L('Preparation ','Préparation ')+(i+1)]);
 case 'thermal':return ['Kr','Xe','H2O','CO2'];case 'sky':return Object.keys(D.sky);
 case 'nuclear':return D.beta.map(r=>r[0]);case 'spectrum':return ['SF6','H',...new Set(D.optical.map(r=>r.observation.split(' ')[0]))];}}
function renderControls(){
  document.documentElement.lang=state.lang;document.querySelectorAll('[data-en]').forEach(n=>n.textContent=n.dataset[state.lang]);
  opts(el('domain'),Object.entries(titles).map(([v,a])=>[v,pick(a)]),state.domain);opts(el('item'),items(state.domain),state.item);state.item=el('item').value;
  el('item-label').textContent=L('Preparation or dataset','Préparation ou données');el('options').replaceChildren();el('catalogue').replaceChildren();el('view').replaceChildren();el('physical').replaceChildren();
  const domain=state.domain;document.getElementById('beat').hidden=domain!=='beat';
  if(domain==='beat'){
    slider(el('physical'),L('Contact stiffness','Raideur de contact'),'k',1,5,.05);
    slider(el('physical'),L('Mantle inertia','Inertie du manteau'),'mass',.22,.55,.01);
    slider(el('physical'),L('Restoring stiffness','Raideur de rappel'),'restoring',1.6,2.8,.05);
    addSelect(el('physical'),L('Initial radial velocity','Vitesse radiale initiale'),[['1',L('Outward','Sortante')],['-1',L('Inward','Entrante')]],state.sign,v=>{state.sign=Number(v);recalculate();display();});recalculate();
  }
  const spatial=['beat','atom','molecule'].includes(domain);el('observer').hidden=!spatial;if(spatial)slider(el('view'),L('View angle (°)','Angle de vue (°)'),'angle',0,180,1);
  if(domain==='molecule'||domain==='field'||(domain==='thermal'&&['Kr','Xe'].includes(state.item))){
    const n=domain==='field'||domain==='thermal'?4:state.item==='SF6'?66:D.modalScenes[state.item].currents.length;
    state.mode=Math.min(n,Math.max(1,state.mode));const labels=domain==='thermal'?['P','ρᵥ','ρₗ','Δh']:domain==='field'?D.wavenumbers:null;
    addSelect(el('options'),domain==='thermal'?L('Observable','Observable'):domain==='field'?L('Wave number','Nombre d’onde'):L('Mode','Mode'),Array.from({length:n},(_,i)=>[String(i+1),String(labels?labels[i]:i+1)]),state.mode,v=>{state.mode=Number(v);display();});
  }
  if(domain==='sky'||(domain==='spectrum'&&state.item==='SF6')){
    const keys=domain==='sky'?['TT','TE','EE']:Object.keys(D.infrared).sort((a,b)=>a-b);
    if(!keys.includes(state.band))state.band=domain==='sky'?'TT':'800';
    addSelect(el('options'),domain==='sky'?L('Channel','Canal'):L('Band start (cm⁻¹)','Début de bande (cm⁻¹)'),keys,state.band,v=>{state.band=v;display();});
  }
  if(domain==='atom'){
    const elem=D.elements.find(e=>e[1]===state.item), section=el('catalogue'), nu=D.isotopes.filter(e=>e[0]===elem[0]),ions=D.ions.filter(e=>e[0]===elem[0]);
    if(window.parent && window.parent!==window){
      const visit=document.createElement('button');visit.type='button';visit.textContent=L('Open the neutral element film for ','Ouvrir le film de l’élément neutre pour ')+state.item;
      visit.onclick=()=>window.parent.postMessage({bstOpenAtom:state.item},'*');section.append(visit);
    }
    const h=document.createElement('h2');h.textContent=L('Isotopes and ions','Isotopes et ions');section.append(h);
    const box=document.createElement('div');box.className='pair';section.append(box);
    const note=document.createElement('p');section.append(note);
    const update=()=>{const r=nu[Number(n.value)],v=ions[Number(q.value)],unknown=L('unassigned','non attribué');
     const count=v[1]===0?r[7]:null;
     note.textContent=state.item+'-'+r[1]+' · q = '+v[1]+' · '+elem[0]+' p · '+(r[1]-elem[0])+' n · '+v[2]+' e · '+L('BST waves in selected state','Ondes BST de l’état sélectionné')+' '+(count??unknown)+' · '+L('half-life','demi-vie')+' '+(r[3]?L('stable','stable'):r[4]===null?L('unreported','non renseignée'):fixed(r[4],7)+' s'+(r[5]?' ≈':''))+' · '+L('cumulative ionization','ionisation cumulée')+' '+(v[3]===null?'—':fixed(v[3])+' eV')+' · '+L('Displayed neutral reference','Référence neutre représentée')+' '+(elem[3]??elem[4])+' '+L('sites','sites');};
    const n=addSelect(box,L('Nuclear state','État nucléaire'),nu.map((r,i)=>[String(i),state.item+'-'+r[1]+' '+r[2]]),0,update);
    const q=addSelect(box,L('Charge state','État de charge'),ions.map((r,i)=>[String(i),'q = '+r[1]]),0,update);update();
    if(window.parent!==window){
      const show=document.createElement('button');show.type='button';show.textContent=L('Open this isotope and charge state','Ouvrir cet isotope et cet état de charge');
      show.onclick=()=>{const a=nu[Number(n.value)],c=ions[Number(q.value)];window.parent.postMessage({bstOpenAtom:state.item,bstState:{mass:a[1],level:a[2],charge:c[1]}},'*');};
      section.append(show);
    }
  }
  state.title=pick(titles[domain])+' · '+el('item').selectedOptions[0].text;if(domain==='atom')state.title+=' · '+L('neutral reference','référence neutre');el('heading').textContent=state.title;const notesFor=notes[domain];el('scope').textContent=notesFor[state.lang==='fr'?1:0];el('legend').textContent=notesFor[state.lang==='fr'?3:2];
  el('export').disabled=domain==='atom' && !D.atoms[state.item] && !D.modalScenes[state.item]?.mode;
  jointControls();el('reset').textContent=L('Restore benchmark settings','Rétablir le réglage du banc');el('export').textContent=L('Export GIF','Exporter le GIF');el('play').textContent=playing?L('Pause','Pause'):L('Play','Lecture');display();
}
function display(){
  const metrics=drawScene(el('scene'),state,state.phase);el('time').value=state.phase*100;const panel=el('values');panel.replaceChildren();
  for(const [k,v] of metrics){const cell=document.createElement('div'),label=document.createElement('span'),val=document.createElement('strong');label.textContent=k;val.textContent=v;cell.append(label,val);panel.append(cell);}
}
const jointSelection=['','','','','',''];
function jointControls(){
  const root=el('joint-controls');root.replaceChildren();
  const labels=[L('Cosmology','Cosmologie'),L('Optical family','Famille optique'),L('Thermal branch','Branche thermique'),L('Fluoride family','Famille fluorures'),L('Charge calibration','Étalonnage de charge'),L('Metric','Métrique')];
  function count(){
    const rows=D.joint.filter(r=>jointSelection.every((v,j)=>v===''||String(r[j+1])===v));
    const accepted=rows.filter(r=>D.jointIds.every((key,j)=>key==='J00.1'||!['?','X'].includes(r[7][j])));
    const detail=accepted.slice(0,5).map(r=>r[0]).join(', ');
    el('joint-examples').textContent=detail?L('Examples in this selection: ','Exemples dans cette sélection : ')+detail:L('No matching combination.','Aucune combinaison correspondante.');
    el('joint-count').textContent=accepted.length+' / '+rows.length+' '+L('jointly compatible · reference population 576','compatibles ensemble · population de référence 576');
  }
  labels.forEach((label,i)=>{const choices=[...new Set(D.joint.map(r=>String(r[i+1])))].sort();
    const terms={area:L('Area per peak','Aire par sommet'),shape:L('Reduced shape','Forme réduite'),'branch-four':L('Branch four','Branche quatre'),'two-preparations':L('Two preparations','Deux préparations'),unit:L('Unit charge','Charge unité')};
    addSelect(root,label,[['',L('All','Tous')],...choices.map(v=>[v,terms[v]||v])],jointSelection[i],v=>{jointSelection[i]=v;count();});});count();
}
let playing=false,last=0;
function animation(now){if(!playing)return;if(last){state.phase=(state.phase+(now-last)/8000)%1;}last=now;display();requestAnimationFrame(animation);}
function stop(){playing=false;last=0;el('play').textContent=L('Play','Lecture');}
el('play').onclick=()=>{if(playing)stop();else{playing=true;last=0;el('play').textContent='Pause';requestAnimationFrame(animation);}};
el('time').oninput=()=>{stop();state.phase=Number(el('time').value)/100;display();};
el('domain').onchange=()=>{stop();Object.assign(state,{domain:el('domain').value,item:'',mode:1,phase:0});el('status').textContent='';renderControls();};
el('item').onchange=()=>{stop();Object.assign(state,{item:el('item').value,mode:1,phase:0});el('status').textContent='';renderControls();};
el('language').onchange=()=>{state.lang=el('language').value;el('status').textContent='';renderControls();if(window.parent&&window.parent!==window)window.parent.postMessage({bstLanguageChanged:state.lang},'*');};
el('reset').onclick=()=>{stop();Object.assign(state,{sign:1,k:3,mass:.3,restoring:2,phase:0,angle:24});renderControls();};
el('export').onclick=async()=>{if(el('export').disabled)return;stop();const button=el('export');button.disabled=true;try{
  const saved={...state};const c=document.createElement('canvas');c.width=432;c.height=264;const frames=[];
  for(let i=0;i<32;i++){drawScene(c,saved,i/31);frames.push(c.getContext('2d').getImageData(0,0,c.width,c.height).data);if(i%4===0){el('status').textContent=L('Encoding frame ','Encodage de l’image ')+(i+1)+'/32';await new Promise(resolve=>setTimeout(resolve,0));}}
  const blob=window.encodeGif(frames,c.width,c.height,12);const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='bst-'+saved.domain+'-'+saved.item+'.gif';a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);el('status').textContent=L('32 frames exported.','32 images exportées.');
 }catch(e){el('status').textContent=String(e);}finally{button.disabled=state.domain==='atom' && !D.atoms[state.item] && !D.modalScenes[state.item]?.mode;}};
renderControls();

window.addEventListener?.('message',event=>{if(event.source===window.parent&&['en','fr'].includes(event.data?.bstLanguage)){state.lang=event.data.bstLanguage;el('language').value=state.lang;renderControls();}});

if(window.parent&&window.parent!==window)window.parent.postMessage({bstLanguageReady:true},'*');

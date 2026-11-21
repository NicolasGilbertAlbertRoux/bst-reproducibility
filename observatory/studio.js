'use strict';
const P=Projections, ui=id=>document.getElementById(id), state={...P.baseState};let frame=0,playing=false,last=null,benchLoaded=false,activeTab='visual',userPaused=window.matchMedia?.('(prefers-reduced-motion: reduce)').matches??true,exporting=false;
const L=(a,b)=>state.lang==='fr'?b:a;
function makeCanvas(w,h){const c=document.createElement('canvas');c.width=w;c.height=h;return c;}
function choices(node,options,selected){node.replaceChildren();for(const [key,text] of options)node.add(new Option(text,key));node.value=String(selected);}
function control(root,label,rows,val,change){const wrap=document.createElement('label'),title=document.createElement('span'),input=document.createElement('select');title.textContent=label;wrap.append(title,input);root.append(wrap);choices(input,rows,val);input.onchange=()=>{pause();change(input.value);draw();};return input;}
function knob(root,title,key,min,max,step){const wrap=document.createElement('label'),label=document.createElement('span'),range=document.createElement('input');range.type='range';Object.assign(range,{min,max,step,value:state[key]});const update=()=>{label.textContent=title+'  '+state[key];};range.oninput=()=>{pause();state[key]=+range.value;if(state.domain==='gravity')frame=0;update();draw();};update();wrap.append(label,range);root.append(wrap);}
function note(){
 const {domain:d,item:k,view:v}=state;
 const white=L('White shows motion. BST proposes a UV counterpart; no wavelength follows from this readout.','Le blanc représente le mouvement. BST propose un rayonnement ultraviolet associé. Cette lecture cinétique ne détermine aucune longueur d’onde.');
 if(d==='atom'){
   if(state.atlasKind!=='neutral')return L('Bars count the documented constituents. A wave total for this state requires its own identified BST preparation. The neutral reference is listed separately.','Les barres dénombrent les constituants documentés. Le total d’ondes de cet état demande une préparation BST identifiée pour lui. Le repère élémentaire neutre figure séparément.');
   if(v==='reference')return white+' '+L('Distinct 309-site krypton preparation with eight native modes. The grey filter shows the 123.58 nm line of excited Kr I.','Préparation distincte du krypton à 309 sites et huit modes natifs. Le filtre gris montre la raie de Kr I excité à 123,58 nm.');
   const m=P.atom(k);const source=m.sourceCount===m.count?'':k==='H'?L(' The central site is inferred from the neutral count.',' Le site central est inféré du décompte neutre.'):L(' Core positions inherit the preceding period closure at a specified display scale.',' Les positions du cœur reprennent la fermeture de la période précédente à une échelle graphique spécifiée.');
   const motion=v==='mode'?L('Native mode: brightness follows modal speed. The radius of its position marker is fixed.','Mode propre : l’éclat suit la vitesse modale. Le rayon du repère de position est fixe.'):L('All site diameters follow one phase (72–128% of the mean). The faint outline marks position at a turning point.','Tous les diamètres suivent la même phase (72 à 128 % de la moyenne). Le contour discret marque la position au retournement.');
   return white+' '+motion+source;
 }
 if(d==='beat')return white+' '+L('Amber shows energy stored at contact. Time is in model units.','L’ambre représente l’énergie stockée au contact. Le temps est en unités du modèle.');
 if(d==='molecule'&&k==='C2')return L('Measured Swan band heads set the filter colours. Local brightness is weighted by strain rate. The display aligns the internal and molecular time windows.','Les têtes de bande de Swan mesurées fixent la couleur des filtres. L’éclat local est pondéré par la vitesse de déformation. Les fenêtres de temps interne et moléculaire sont alignées pour le rendu.');
 if(d==='molecule'&&k==='I2'){
   const motion=L('Two neutral iodine atlas shapes share a harmonic stretch. Equilibrium bond 2.6663 Å, chosen amplitude 0.06 Å, harmonic constant 214.502 cm⁻¹.','Deux formes de l’atlas de l’iode neutre oscillent autour de la distance d’équilibre de 2,6663 Å. Amplitude choisie de 0,06 Å et constante harmonique de 214,502 cm⁻¹.');
   return motion+(v!=='fine'&&state.colour==='spectral'?' '+OpticalView.summary(state):'');
 }
 if(d==='molecule'&&k==='Kr2')return L('Grey shows strain rate. The bound preparation carries 4.21 meV above the potential minimum.','Le gris représente la vitesse de déformation. La préparation liée se situe à 4,21 meV au-dessus du minimum du potentiel.');
 if(d==='molecule')return L('Bond lengths and mantle radii share one ångström scale. Adopted covalent radii: H 31, C 76, O 66, F 57, S 105 pm. The common internal phase is a display preparation.','Liaisons et manteaux partagent une échelle en ångströms. Rayons covalents adoptés : H 31, C 76, O 66, F 57, S 105 pm. La phase interne commune est une préparation de visualisation.')+' '+(k==='SF6'?L('The directions and mode come from the archived SF₆ contact operator.','Les directions et le mode proviennent de l’opérateur de contact archivé de SF₆.'):L('The vibration of the reference gas-phase geometry is enlarged for visibility.','L’amplitude de la vibration de la géométrie de référence en phase gazeuse est agrandie pour la lecture.'))+(v!=='fine'&&state.colour==='spectral'?' '+OpticalView.summary(state):'');
 if(d==='substrate')return L('Each stiffness and inertia pair selects one of nine archived integrations at 25 times.','Chaque couple raideur–inertie sélectionne une des neuf intégrations archivées à 25 instants.');
 if(d==='thermal'){
   const crystal=v==='solid'||v==='shell';
   const model=crystal?L('Crystallographic reference and reduced harmonic vibrations around occupied sites. The lattice is the initial condition.','Structure cristallographique de référence et vibrations harmoniques réduites autour des sites occupés. Le réseau est une condition initiale.'):L('108 centres with the same reduced Lennard–Jones interaction for every substance. Rotations and the length conversion for the silhouettes are display conventions.','108 centres soumis à la même interaction de Lennard–Jones réduite pour chaque substance. Les rotations et la conversion de longueur des silhouettes sont des conventions de représentation.');
   const reading=state.colour==='spectral'?OpticalView.summary(state):L('Grey shows centre motion.','Le gris montre le mouvement des centres.');
   return model+' '+reading+(v==='shell'?' '+L('Dashed guides join the selected site to its first neighbours.','Les pointillés relient le site choisi à ses premiers voisins.'): '');

 }
 if(d==='field'&&k==='SF6')return L('Left: the six-contact BST molecular mode. Right: exterior line-of-sight induction proportional to (dP/dt × r)/r³, using the archived infrared electric-dipole fit in the near-field approximation. The central region is masked. An overall sign is chosen for the fit. The field is relative; it was not measured in tesla.','À gauche, le mode moléculaire BST à six contacts. À droite, la composante de l’induction extérieure sur la ligne de visée, proportionnelle à (dP/dt × r)/r³, issue du dipôle électrique ajusté en infrarouge, dans l’approximation du champ proche. La région centrale est masquée. Un signe global est choisi pour l’ajustement. L’échelle est relative, sans mesure en teslas.');
 if(d==='field'&&k.startsWith('run'))return L('Nine archived BST time samples. Choose a line of sight and wave number to read the transverse power of the integrated-current preparation.','Neuf instants archivés du calcul BST. Choisir une ligne de visée et un nombre d’onde pour lire la puissance transverse de la préparation de courant.');
 if(d==='field')return L('Classical point-dipole comparator. The arrows follow static field lines.','Comparateur classique à dipôles ponctuels. Les repères suivent les lignes du champ statique.');
 if(d==='gravity')return L('Newtonian exterior gravity of spherical, uniform aggregates. Each body has the mass of its chosen population of constituents. Number density is a preparation input. The trajectory ends at first surface contact or 96 h. This is a macroscopic comparison.','Gravitation newtonienne extérieure d’agrégats sphériques uniformes. Chaque corps a la masse de sa population de constituants. La densité en nombre est une donnée de préparation. Le calcul s’arrête au premier contact des surfaces ou à 96 h. Il s’agit d’une comparaison macroscopique.');
 return L('C00 and the selected family share harmonic phases and beam width. The third panel gives their signed difference with a fixed ±5 µK scale. The two skies use ±85 µK. Frame position selects beam width, not cosmic time.','C00 et la famille choisie partagent les phases harmoniques et la largeur du faisceau. Le troisième panneau affiche l’écart signé avec une échelle fixe de ±5 µK. Les deux ciels utilisent ±85 µK. Le curseur règle le faisceau angulaire.');
}
function setAtlasKind(kind){
 state.atlasKind=kind;
 if(kind==='neutral'){state.charge=0;state.isotope=0;state.nuclearState='gs';}
 if(kind==='ion')state.charge=Math.max(1,state.charge);
 if(kind==='isotope'){
   state.charge=0;
   if(!state.isotope){const a=AtlasState.defaultNucleus(state.item);state.isotope=a[1];state.nuclearState=a[2];}
 }
 state.view=P.readings('atom',state.item,kind)[0];frame=0;
}
function catalogue(){
 const root=ui('catalogue');root.replaceChildren();if(state.domain!=='atom')return;
 const data=AtlasState.rows(state.item),kind=state.atlasKind,symbol=state.item,m=P.atom(symbol);
 const bar=document.createElement('div');bar.className='toolbar';root.append(bar);
 control(bar,L('Atlas entry','Entrée de l’atlas'),[['neutral',L('Neutral atom','Atome neutre')],['isotope',L('Isotopes','Isotopes')+' ('+data.nuclei.length+')'],['ion',L('Ions','Ions')+' ('+(data.charges.length-1)+')']],kind,v=>{setAtlasKind(v);configure();});
 const title=document.createElement('p');title.className='atlas-count';root.append(title);
 const info=document.createElement('p');root.append(info);
 if(kind==='neutral'){
   const status=data.element[3]===null?L('conditional reconstruction','reconstruction conditionnelle'):L('archived count','décompte archivé');
   title.textContent=symbol+' · '+m.count+' '+L('neutral model waves','ondes du modèle neutre')+' ('+status+') · '+(m.count-m.surfaceCount)+' '+L('core','cœur')+' + '+m.surfaceCount+' '+L('surface','surface');
   info.textContent=state.view==='reference'?L('Film showing the separate 309-site, eight-mode preparation.','Le film montre la préparation distincte à 309 sites et huit modes.'):m.count+' / '+m.count+' '+L('represented sites in the atlas film.','sites représentés dans le film de l’atlas.');return;
 }
 const entries=data.nuclei.map(a=>[a[1]+':'+a[2],symbol+'-'+a[1]+' · '+(a[2]==='gs'?L('ground state','fondamental'):a[2])]);
 if(kind==='ion')entries.unshift(['',L('Isotope unspecified','Isotope non précisé')]);
 control(bar,L('Isotope and nuclear state','Isotope et état nucléaire'),entries,state.isotope?state.isotope+':'+state.nuclearState:'',val=>{
    const [mass,level]=val.split(':');state.isotope=Number(mass);state.nuclearState=level||'gs';configure();
 });
 if(kind==='ion')control(bar,L('Ion charge','Charge de l’ion'),data.charges.filter(a=>a[1]>0).map(a=>[a[1],symbol+' · q = +'+a[1]]),state.charge,v=>{state.charge=+v;configure();});
 const a=AtlasState.resolve(state);
 title.textContent=a.id+' · '+L('BST waves for this state','Ondes BST propres à cet état')+' · '+(a.count??L('undetermined','indéterminé'));
 info.textContent=AtlasState.details(state,a);
}
function bodies(){
 const root=ui('bodySettings');root.replaceChildren();if(state.domain!=='gravity')return;
 const n=state.item==='binary'?2:3;
 for(let i=1;i<=n;i++){
   const pane=document.createElement('fieldset'),title=document.createElement('legend');title.textContent=L('Body ','Corps ')+i;pane.append(title);root.append(pane);
   control(pane,L('Composition','Composition'),Object.keys(Gravity.composition).map(k=>[k,P.name(k,state.lang)]),state['body'+i],x=>{state['body'+i]=x;frame=0;});
   knob(pane,L('Radius (km)','Rayon (km)'),'radius'+i,200,3000,10);
   knob(pane,L('Number density (10²⁸ m⁻³)','Densité (10²⁸ m⁻³)'),'density'+i,.5,6,.1);
 }
}
function configure(){const lang=state.lang,d=state.domain,k=state.item,v=state.view;document.documentElement.lang=lang;ui('brand').textContent=L('Projection observatory','Observatoire des projections');ui('bench').textContent=L('Measurements and controls','Mesures et réglages');ui('areaLabel').textContent=L('Domain','Domaine');ui('objectLabel').textContent=L('Preparation','Préparation');ui('readLabel').textContent=L('View','Vue');ui('save').textContent=L('Export GIF','Exporter le GIF');ui('timeLabel').textContent=L('Recorded frame','Image enregistrée');ui('time').setAttribute('aria-label',ui('timeLabel').textContent);ui('image').setAttribute('aria-label',P.name(d,lang)+' – '+P.name(k,lang));ui('play').textContent=playing?'Pause':L('Play','Lecture');ui('notice').textContent='';
 choices(ui('area'),Object.keys(P.categories).map(a=>[a,P.name(a,lang)]),d);choices(ui('object'),P.options(d).map((a,i)=>[a,(d==='atom'?(i+1)+' · ':'')+P.name(a,lang)]),k);choices(ui('reading'),P.readings(d,k,state.atlasKind).map(a=>[a,d==='atom'&&a==='reference'?L('Eight-mode krypton film · 309 sites','Film du krypton à huit modes · 309 sites'):P.name(a,lang)]),v);
 ui('visual').textContent=L('Simulations','Simulations');ui('comparison').title=L('Numerical comparisons','Comparaisons numériques');ui('timeLabel').textContent=d==='sky'?L('Angular beam','Faisceau angulaire'):L('Frame','Image');
 const cfg=ui('config');cfg.replaceChildren();
 if(d==='gravity'){knob(cfg,L('Initial separation (1,000 km)','Séparation initiale (1 000 km)'),'separation',8,22,.25);knob(cfg,L('Initial velocity factor','Facteur de vitesse initiale'),'speed',.2,1.7,.05);}
 if(d==='field'&&k==='SF6'){
  control(cfg,L('Dipole fit','Ajustement du dipôle'),[['extended',L('With the 1995 comparator','Avec le comparateur de 1995')],['modern',L('Modern comparator','Comparateur moderne')]],state.fit,x=>state.fit=x);
  control(cfg,L('Molecular mode','Mode moléculaire'),Array.from({length:66},(_,j)=>[j+1,j+1]),state.mode,x=>state.mode=+x);knob(cfg,L('Viewing angle (°)','Angle de vue (°)'),'angle',0,180,1);
 }
 if(d==='field'&&k.startsWith('run')){
  control(cfg,L('Line of sight','Ligne de visée'),['x','y','z','(1,1,1)/√3'].map((v,i)=>[i,v]),state.direction,x=>state.direction=+x);
  control(cfg,L('Wave number','Nombre d’onde'),BST_DATA.wavenumbers.map((v,i)=>[i,v]),state.waveNumber,x=>state.waveNumber=+x);
 }if(d==='beat'&&v==='model'){if(k==='pair')knob(cfg,L('Contact stiffness','Raideur de contact'),'k',1,5,.05);knob(cfg,L('Mantle inertia','Inertie du manteau'),'mass',.22,.55,.01);knob(cfg,L('Restoring stiffness','Raideur de rappel'),'restoring',1.6,2.8,.05);control(cfg,L('Initial velocity','Vitesse initiale'),[[1,L('Outward','Sortante')],[-1,L('Inward','Entrante')]],state.sign,x=>state.sign=+x);}
 if(d==='substrate'){for(const [key,title,points] of [['rigidity',L('Stiffness K','Raideur K'),[1,2,4]],['inertia',L('Inertia χ','Inertie χ'),[4,8,16]]])control(cfg,title,points.map(v=>[v,v]),state[key],a=>state[key]=+a);control(cfg,L('Field','Champ'),[['shape',L('Shape','Forme')],['scalar',L('Scalar','Scalaire')],['charge',L('Charge','Charge')]],state.quantity,a=>state.quantity=a);}
 if(d==='molecule'&&k==='C2'&&!['reference','fine'].includes(v))control(cfg,L('Spectral filter','Filtre spectral'),[516.5,473.7,563.5].map(v=>[v,v+' nm']),state.band,a=>state.band=+a);
 if((d==='atom'&&v!=='reference'&&state.atlasKind==='neutral')||(d==='molecule'&&!['C2','Kr2'].includes(k)))knob(cfg,L('Viewing angle (°)','Angle de vue (°)'),'angle',0,180,1);
 if(d==='thermal'){
   knob(cfg,L('Azimuth (°)','Azimut (°)'),'angle',0,360,1);
   knob(cfg,L('Elevation (°)','Élévation (°)'),'elevation',-90,90,1);
 }
 if(OpticalView.supported(state)){
   control(cfg,L('Optical readout','Lecture optique'),[['spectral',L('Spectrum and motion','Spectre et cinétique')],['motion',L('Motion only','Cinétique seule')]],state.colour,value=>{state.colour=value;configure();});
   if(OpticalView.cell(state)&&state.colour==='spectral')control(cfg,L('Optical path','Trajet optique'),OpticalView.paths.map(d=>[d,OpticalView.pathName(state,d)]),state.pathM,value=>{state.pathM=Number(value);configure();});
 }
 if(d==='molecule'&&['H2O','CO2','SF6'].includes(k)){const rows=k==='SF6'?Array.from({length:66},(_,i)=>[i+1,i+1]):k==='CO2'?[[1,L('Bend','Flexion')],[2,L('Asymmetric stretch','Étirement antisymétrique')]]:[[1,L('Bend','Flexion')],[2,L('Symmetric stretch','Étirement symétrique')],[3,L('Asymmetric stretch','Étirement antisymétrique')]];control(cfg,L('Mode','Mode'),rows,state.mode,a=>state.mode=+a);}
 catalogue();bodies();ui('note').textContent=note();draw();}
function choose(){
 state.atlasKind='neutral';state.isotope=0;state.nuclearState='gs';state.charge=0;
 state.view=P.readings(state.domain,state.item)[0];state.mode=state.domain==='field'&&state.item==='SF6'?11:1;
 if(state.domain!=='thermal')state.angle=Math.min(180,state.angle);
 frame=0;pause();configure();if(!userPaused)start();
}
function draw(){const [w,h,n]=P.size(state),canvas=ui('image');if(canvas.width!==w)canvas.width=w;if(canvas.height!==h)canvas.height=h;frame=Math.min(n-1,Math.max(0,frame));ui('caption').textContent=P.render(canvas,state,frame,makeCanvas);ui('play').disabled=n===1;ui('time').disabled=n===1;
 ui('save').textContent=n===1?L('Export state card · GIF','Exporter la fiche · GIF'):L('Export GIF','Exporter le GIF');
 if(state.domain==='gravity'){const sim=Gravity.history(state);ui('caption').textContent+=' · '+sim.bodies.map((b,i)=>'M'+(i+1)+' = '+b.mass.toExponential(3)+' kg').join(' · ');}
 ui('time').max=n-1;ui('time').value=frame;ui('counter').textContent=(frame+1)+' / '+n;}
function pause(){playing=false;last=null;ui('play').textContent=L('Play','Lecture');}
function tick(ms){if(!playing)return;const n=P.size(state)[2],delay=state.view==='reference'?P.film(state).frames[frame].delay:state.domain==='sky'?440:90;if(last===null)last=ms;if(ms-last>=delay){frame=(frame+1)%n;draw();last=ms;}requestAnimationFrame(tick);}
function start(){if(playing||P.size(state)[2]<2)return;playing=true;last=null;ui('play').textContent='Pause';requestAnimationFrame(tick);}
ui('play').onclick=()=>{if(playing){userPaused=true;pause();return;}userPaused=false;start();};ui('time').oninput=()=>{userPaused=true;pause();frame=+ui('time').value;draw();};
ui('area').onchange=()=>{state.domain=ui('area').value;state.item=state.domain==='atom'?'Kr':state.domain==='sky'?'C01':P.options(state.domain)[0];choose();};ui('object').onchange=()=>{state.item=ui('object').value;choose();};ui('reading').onchange=()=>{
 pause();
 const previous=state.view, count=P.size(state)[2], bands=[516.5,473.7,563.5];
 state.view=ui('reading').value;
 if(state.domain==='atom'&&state.view==='reference')state.atlasKind='neutral';
 if(state.domain==='molecule' && state.item==='C2' && previous==='reference'){
   state.band=bands[Math.floor(frame/72)];
   frame%=72;
 }else if(state.domain==='molecule' && state.item==='C2' && state.view==='reference'){
   frame=72*bands.indexOf(state.band)+frame;
 }else{
   frame=count>1?Math.round(frame/(count-1)*(P.size(state)[2]-1)):0;
 }
 configure();
};
function syncLab(){const win=ui('comparison').contentWindow;if(win)win.postMessage({bstLanguage:state.lang},'*');}
window.addEventListener('message',e=>{if(e.source!==ui('comparison').contentWindow)return;if(e.data?.bstLanguageReady)syncLab();const atom=e.data?.bstOpenAtom;if(typeof atom==='string'&&P.options('atom').includes(atom)){tabs('visual');state.domain='atom';state.item=atom;choose();const a=e.data.bstState;
 if(a){pause();state.isotope=a.mass;state.nuclearState=a.level;state.charge=a.charge;setAtlasKind(a.charge?'ion':'isotope');configure();}return;}const lang=e.data?.bstLanguageChanged;if(['en','fr'].includes(lang)&&state.lang!==lang){state.lang=ui('language').value=lang;configure();}});
ui('language').onchange=()=>{state.lang=ui('language').value;configure();if(benchLoaded)syncLab();};
function tabs(key){pause();activeTab=key;ui('studio').hidden=key!=='visual';ui('lab').hidden=key!=='bench';ui('visual').setAttribute('aria-pressed',String(key==='visual'));ui('bench').setAttribute('aria-pressed',String(key==='bench'));if(key==='bench'&&!benchLoaded){benchLoaded=true;ui('comparison').onload=syncLab;if(window.BST_COMPARISON_HTML)ui('comparison').srcdoc=window.BST_COMPARISON_HTML;else ui('comparison').src='comparisons.html';}}
ui('visual').onclick=()=>tabs('visual');ui('bench').onclick=()=>tabs('bench');
ui('save').onclick=async()=>{if(exporting)return;pause();const config={...state};const btn=ui('save');btn.disabled=true;exporting=true;try{let blob;if(config.view==='reference')blob=new Blob([P.film(config).raw],{type:'image/gif'});else{const [w,h,n]=P.size(config);const c=makeCanvas(w,h),encoder=createGifEncoder(w,h,config.domain==='sky'?44:9,OpticalView.swatches(config));for(let i=0;i<n;i++){P.render(c,config,i,makeCanvas);encoder.addFrame(c.getContext('2d').getImageData(0,0,w,h).data);if(i%4===0){ui('notice').textContent=L('Encoding frame ','Encodage de l’image ')+(i+1)+' / '+n;await new Promise(r=>setTimeout(r,0));}}blob=encoder.finish();}const u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download=(config.domain==='atom'&&config.atlasKind!=='neutral'?'atom-'+AtlasState.resolve(config).id.replace(/[^a-z0-9-]+/gi,'_')+'-state':[config.domain,config.item,config.view].join('-'))+'.gif';a.click();setTimeout(()=>URL.revokeObjectURL(u),20000);ui('notice').textContent=L('GIF exported.','GIF exporté.');}catch(e){ui('notice').textContent=e.message;}finally{btn.disabled=false;exporting=false;}};configure();if(!userPaused)start();

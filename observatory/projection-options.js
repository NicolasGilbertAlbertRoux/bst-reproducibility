'use strict';
function projectionSettings(P,domain,item,args){if(!P.categories[domain]||!P.options(domain).includes(item))throw Error('Unsupported domain or preparation');const s={...P.baseState,domain,item,view:P.readings(domain,item)[0]};const keys={language:'lang',view:'view',mode:'mode',band:'band',angle:'angle',elevation:'elevation',colour:'colour','path-length':'pathM',isotope:'isotope','nuclear-state':'nuclearState',charge:'charge',stiffness:'k',inertia:'mass',restoring:'restoring',sign:'sign',rigidity:'rigidity','field-inertia':'inertia',quantity:'quantity',fit:'fit',direction:'direction','wave-number':'waveNumber',...Object.fromEntries(Object.keys(Gravity.defaults).map(k=>[k,k]))};
 if(args.length%2)throw Error('Options need a value');for(let i=0;i<args.length;i+=2){const key=keys[args[i].slice(2)];if(!key||!args[i].startsWith('--'))throw Error('Unknown option '+args[i]);s[key]=typeof P.baseState[key]==='number'?Number(args[i+1]):args[i+1];}
 if(domain==='beat'&&!args.includes('--view')&&['--stiffness','--inertia','--restoring','--sign'].some(k=>args.includes(k)))s.view='model';
 if(domain==='molecule'&&item==='C2'&&args.includes('--band')&&!args.includes('--view'))s.view='molecular';
 if(args.includes('--elevation')&&domain!=='thermal')throw Error('Elevation is available for thermal views');
 if(args.includes('--colour')&&!(domain==='thermal'||(domain==='molecule'&&['H2O','CO2','SF6','I2'].includes(item))))throw Error('Unsupported colour control for this preparation');
 // Earlier exported commands used this value for the structural overlay.
 if(s.colour==='composition')s.colour='motion';
 if(!['spectral','motion'].includes(s.colour))throw Error('Unsupported optical reading');
 if(args.includes('--path-length')&&(!OpticalView.cell(s)||s.colour!=='spectral'||!OpticalView.paths.includes(s.pathM)))throw Error('Optical path requires a supported transmission cell and spectral readout');
 if(['--isotope','--nuclear-state','--charge'].some(k=>args.includes(k))){
    if(domain!=='atom'||!Number.isInteger(s.isotope)||s.isotope<0||!Number.isInteger(s.charge)||s.charge<0||
       (args.includes('--isotope')&&!s.isotope)||(args.includes('--nuclear-state')&&!s.isotope))throw Error('Invalid atomic state');
    s.atlasKind=s.charge?'ion':s.isotope?'isotope':'neutral';
    if(s.atlasKind!=='neutral'){AtlasState.resolve(s);if(!args.includes('--view'))s.view='state';}
 }
 if(!P.readings(domain,item,s.atlasKind).includes(s.view)||!['fr','en'].includes(s.lang))throw Error('Unsupported view or language');const n=item==='SF6'?66:item==='H2O'?3:item==='CO2'?2:1;
 if(domain==='field'&&item==='SF6'&&!args.includes('--mode'))s.mode=11;
 if(!['modern','extended'].includes(s.fit)||![0,1,2,3].includes(s.direction)||![0,1,2,3].includes(s.waveNumber))throw Error('Invalid field settings');
 if(domain==='gravity'){for(const k of ['body1','body2','body3'])if(!Object.hasOwn(Gravity.composition,s[k]))throw Error('Unknown composition');for(const k of Object.keys(Gravity.defaults).filter(k=>!k.startsWith('body'))){const range=k.startsWith('radius')?[200,3000]:k.startsWith('density')?[.5,6]:k==='speed'?[.2,1.7]:[8,22];if(!Number.isFinite(s[k])||s[k]<range[0]||s[k]>range[1])throw Error('Gravity parameter outside range: '+k);}}
 if(!Number.isInteger(s.mode)||s.mode<1||s.mode>n||![473.7,516.5,563.5].includes(s.band)||![1,2,4].includes(s.rigidity)||![4,8,16].includes(s.inertia)||!['shape','scalar','charge'].includes(s.quantity)||![-1,1].includes(s.sign))throw Error('Unsupported mode, filter or parameter');for(const [v,lo,hi] of [[s.angle,0,domain==='thermal'?360:180],[s.elevation,-90,90],[s.k,1,5],[s.mass,.22,.55],[s.restoring,1.6,2.8]])if(!Number.isFinite(v)||v<lo||v>hi)throw Error('Value outside the supported range');return s;}

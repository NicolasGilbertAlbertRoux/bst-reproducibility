'use strict';
const ComparativeViews=(()=>{
  const tx=(s,en,fr)=>s.lang==='fr'?fr:en;
  function title(c,x,y,label,colour='#c8dddf',size=14){c.fillStyle=colour;c.font=size+'px sans-serif';c.fillText(label,x,y);}
  function skyArray(key){
    if(!skyArray.cache)skyArray.cache=new Map();
    if(!skyArray.cache.has(key)){
      if(skyArray.cache.size>2)skyArray.cache.clear();
      const a=pako.inflate(bytes64(BST_EXTENDED.sky.families[key].map));
      skyArray.cache.set(key,new Int16Array(a.buffer,a.byteOffset,a.byteLength/2));
    }
    return skyArray.cache.get(key);
  }
  function sky(c,s,i,make){
    const q=BST_EXTENDED.sky,{w,h}=q,n=w*h,ref=skyArray('C00'),v=skyArray(s.item),f=q.families[s.item];
    // Each of the three panels uses the same beam and harmonic phases. Residuals have their own fixed, labelled scale.
    const limits=[85,85,5]; let sum=0,wt=0;
    for(let panel=0;panel<3;panel++){
      const tile=make(w,h),cx=tile.getContext('2d'),out=cx.createImageData(w,h);
      for(let y=0;y<h;y++){
        const area=Math.cos(-Math.PI/2+y*Math.PI/(h-1));
        for(let x=0;x<w;x++){
          const j=i*n+y*w+x,delta=(v[j]-ref[j])/32;
          if(panel===2){sum+=area*delta**2;wt+=area;}
          const k=panel===0?ref[j]/32:panel===1?v[j]/32:delta;
          const vabs=Math.min(1,Math.abs(k)/limits[panel]),p=((h-1-y)*w+x)*4,colour=k<0?[87,164,231]:[250,167,77];
          for(let a=0;a<3;a++)out.data[p+a]=15+(colour[a]-15)*Math.sqrt(vabs);
          out.data[p+3]=255;
        }
      }
      cx.putImageData(out,0,0);const x=9+208*panel;c.drawImage(tile,x,116,202,112);
      const headings=['C00',s.item,s.item+' − C00'];title(c,x,100,headings[panel]);title(c,x,251,'±'+limits[panel]+' µK','#a9c3c7',12);
    }
    title(c,11,282,tx(s,'Same phases and beam · temperature in µK','Mêmes phases et même faisceau · température en µK'));
    return 'ℓ = 48–84   ·   FWHM = '+q.beams[i]+'°   ·   RMS = '+f.rms[i].toFixed(2)+' µK   ·   RMS(ΔT) = '+Math.sqrt(sum/wt).toFixed(3)+' µK';
  }
  function moleculeField(c,s,i,make){
    const fit=BST_DATA.projectionInputs.dipoleFits[s.fit];
    const v=fit.vectors[s.mode-1],g=AtlasView.molecule(c,{...s,item:'SF6',view:'molecular'},i,make,[154,184],44);
    const phase=2*Math.PI*(i/72+.14),raw=v.map(x=>-x*g.freq*Math.sin(phase));
    const d=AtlasView.camera(raw,s.angle),max=Math.max(...fit.vectors.map((row,j)=>Math.hypot(...row)*BST_DATA.fluoride.frequencies[j]))||1;
    title(c,22,66,tx(s,'Molecular sites · SF₆','Sites moléculaires · SF₆'));title(c,347,66,tx(s,'Near-field induction · line of sight','Induction de proximité · ligne de visée'));
    const nx=93,ny=77,tile=make(nx,ny),cx=tile.getContext('2d'),pixels=cx.createImageData(nx,ny);
    for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
      const rx=(x-(nx-1)/2)/10,ry=((ny-1)/2-y)/10,r=Math.hypot(rx,ry),at=4*(y*nx+x);
      let val=0;if(r>=1.8)val=(d[0]*ry-d[1]*rx)/r**3/max;
      const cl=val<0?[100,176,250]:[252,169,87];const power=Math.min(1,Math.sqrt(5*Math.abs(val)));
      for(let j=0;j<3;j++)pixels.data[at+j]=10+power*(cl[j]-10);pixels.data[at+3]=255;
    }
    cx.putImageData(pixels,0,0);c.drawImage(tile,340,80,278,225);
    title(c,350,321,tx(s,'Blue −   /   amber +   ·   relative units','Bleu −   /   ambre +   ·   unités relatives'),'#add0d0',12);
    return 'SF₆   ·   '+tx(s,'mode','mode')+' '+s.mode+'   ·   ω = '+g.freq.toFixed(4)+'   ·   |dP/dt| = '+Math.hypot(...raw).toFixed(2)+'  (rel.)';
  }
  function current(c,s,i){
    const n=Number(s.item.slice(-1)),run=BST_DATA.field[n],info=BST_DATA.projectionInputs.fieldRuns[n],j=run.current[i];
    const chan=s.direction,ki=s.waveNumber,values=run.powers.map(a=>a[chan][ki]),max=Math.max(...BST_DATA.field.flatMap(r=>r.powers.map(a=>a[chan][ki])),1e-9);
    title(c,30,83,tx(s,'Integrated substrate current','Courant intégré du substrat'));
    const jmax=Math.max(...BST_DATA.field.flatMap(r=>r.current.flatMap(v=>v.map(Math.abs))));
    ['x','y','z'].forEach((axis,k)=>{const yy=131+k*56;title(c,28,yy,'J'+axis);c.fillStyle='#31464b';c.fillRect(63,yy-7,210,1);c.fillStyle=j[k]>=0?'#75d2c1':'#e2a477';c.fillRect(168+Math.min(0,j[k]/jmax*100),yy-12,Math.abs(j[k]/jmax*100),8);title(c,270,yy,j[k].toFixed(2),'#bdd0d0',12);});
    title(c,339,84,tx(s,'Projected transverse power','Puissance transverse projetée'));
    const X=k=>344+32.2*k,Y=p=>270-p/max*153;c.strokeStyle='#86d9d7';c.lineWidth=2;c.beginPath();values.forEach((v,k)=>k?c.lineTo(X(k),Y(v)):c.moveTo(X(k),Y(v)));c.stroke();
    c.beginPath();c.arc(X(i),Y(values[i]),4.2,0,2*Math.PI);c.fillStyle='#f3be7e';c.fill();
    title(c,339,300,'t = 0',undefined,12);title(c,556,300,'0.32',undefined,12);
    return info.recipe+' · '+info.grid+'³   ·   Δt = '+info.dt+'   ·   t = '+run.t[i].toFixed(2)+'   ·   P⊥ = '+values[i].toPrecision(5);
  }
  return {sky,moleculeField,current};
})();

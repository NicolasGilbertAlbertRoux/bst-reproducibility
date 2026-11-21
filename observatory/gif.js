/* GIF89a with an explicit grayscale ramp for structural deformation. */
window.createGifEncoder = function(width, height, delay, swatches = []) {
  const chunks = [], emit = (...bytes) => chunks.push(Uint8Array.from(bytes));
  const word = value => [value & 255, (value >> 8) & 255];
  const text = value => chunks.push(Uint8Array.from(value, c => c.charCodeAt(0)));
  text("GIF89a"); emit(...word(width), ...word(height), 0xf7, 0, 0);
  for (let i = 0; i < 64; i++) { const v=Math.round(255*i/63); emit(v,v,v); }
  // The colour cube repeats the black and white entries of the grey ramp. These
  // two slots can preserve measured sample colours exactly in a 256-colour GIF.
  const exact = new Map();
  const replacements = new Map();
  for(const [r,g,b] of swatches){
    const key=(r<<16)|(g<<8)|b;
    if(key===0||key===0xffffff||exact.has(key))continue;
    if(exact.size===2)throw Error('A single GIF supports two exact spectral samples');
    const slot=exact.size===0?64:255;
    exact.set(key,slot); replacements.set(slot,[r,g,b]);
  }
  for(let r=0;r<6;r++)for(let g=0;g<8;g++)for(let b=0;b<4;b++){
    const index=64+r*32+g*4+b;
    emit(...(replacements.get(index)||[Math.round(255*r/5),Math.round(255*g/7),Math.round(255*b/3)]));
  }
  emit(0x21,0xff,11); text("NETSCAPE2.0"); emit(3,1,0,0,0);
  function addFrame(frame) {
    emit(0x21,0xf9,4,0,...word(delay),0,0,0x2c,0,0,0,0,...word(width),...word(height),0,8);
    const compressed=[]; let buffer=0,bits=0;
    function code(value){buffer|=value<<bits;bits+=9;while(bits>=8){compressed.push(buffer&255);buffer>>>=8;bits-=8;}}
    const dictionary=new Map(); let next=258, prefix=null;
    code(256);
    for(let j=0;j<width*height;j++){
      const r=frame[j*4],g=frame[j*4+1],b=frame[j*4+2];
      let index=Math.max(r,g,b)-Math.min(r,g,b)<9?Math.round((r+g+b)/3*63/255):64+Math.round(r*5/255)*32+Math.round(g*7/255)*4+Math.round(b*3/255);
      if(exact.size){
        const special=exact.get((r<<16)|(g<<8)|b);
        if(special!==undefined)index=special;
        else if(index===64)index=0;
        else if(index===255)index=63;
      }
      if(prefix===null){prefix=index;continue;}
      const key=prefix*256+index, found=dictionary.get(key);
      if(found!==undefined){prefix=found;continue;}
      code(prefix);
      if(next<505){dictionary.set(key,next++);}else{code(256);dictionary.clear();next=258;}
      prefix=index;
    }
    if(prefix!==null)code(prefix);code(257);if(bits)compressed.push(buffer&255);
    for(let i=0;i<compressed.length;i+=255){const part=compressed.slice(i,i+255);emit(part.length);chunks.push(Uint8Array.from(part));}
    emit(0);
  }
  return {addFrame, finish(){emit(0x3b);return new Blob(chunks,{type:"image/gif"});}};
};

window.encodeGif = function(frames,width,height,delay){
  const encoder=createGifEncoder(width,height,delay);
  for(const frame of frames)encoder.addFrame(frame);
  return encoder.finish();
};

/* Sequential GIF89a reader for frame-accurate scrubbing of the four reference plates. */
'use strict';
function bytes64(s){
 if(typeof Uint8Array.fromBase64==='function')return Uint8Array.fromBase64(s);
 const binary=atob(s),bytes=new Uint8Array(binary.length);
 for(let i=0;i<bytes.length;i++)bytes[i]=binary.charCodeAt(i);
 return bytes;
}
class ReferenceFilm{
 constructor(encoded){
  this.raw=bytes64(encoded);const a=this.raw;let off=6;const word=()=>{const x=a[off]|a[off+1]<<8;off+=2;return x;};
  this.width=word();this.height=word();const packed=a[off++];this.background=a[off++];off++;
  const palette=flag=>{if(!(flag&128))return null;const n=3*(2<<(flag&7));const v=a.subarray(off,off+n);off+=n;return v;};
  this.global=palette(packed);this.frames=[];this.duration=0;let control={};
  function blocks(){const start=off;while(a[off])off+=1+a[off];off++;return [start,off];}
  while(off<a.length){const type=a[off++];if(type===0x3b)break;
   if(type===0x21){const code=a[off++];if(code===0xf9){off++;const flags=a[off++],delay=word();control={delay:Math.max(1,delay)*10,dispose:(flags>>2)&7,trans:(flags&1)?a[off]:-1};off+=2;}else blocks();continue;}
   if(type!==0x2c)throw Error('Invalid reference GIF');const x=word(),y=word(),w=word(),h=word(),flags=a[off++],pal=palette(flags)||this.global;
   const size=a[off++],[start,end]=blocks();if(flags&64)throw Error('The distributed reference GIFs use sequential scanlines');
   this.frames.push({x,y,w,h,pal,size,start,end,at:this.duration,trans:-1,dispose:0,delay:70,...control});this.duration+=this.frames.at(-1).delay;control={};
  }
  this.pixels=new Uint8ClampedArray(this.width*this.height*4);this.index=-1;
 }
 read(i){
  if(i<this.index){this.index=-1;this.pixels.fill(0);this.previous=null;}
  const a=this.raw,out=this.pixels;
  while(this.index<i){const last=this.frames[this.index];
   if(last?.dispose===2){for(let y=last.y;y<last.y+last.h;y++)out.fill(0,4*(y*this.width+last.x),4*(y*this.width+last.x+last.w));}
   if(last?.dispose===3&&this.previous)out.set(this.previous);
   const f=this.frames[++this.index];if(f.dispose===3)this.previous=out.slice();
   const clear=1<<f.size,eof=clear+1,pre=new Uint16Array(4096),suf=new Uint8Array(4096),stack=new Uint8Array(4097);
   let offset=f.start,left=0,buf=0,bits=0,size=f.size+1,next=eof+1,prior=-1,first=0,count=0;
   while(count<f.w*f.h){while(bits<size){if(!left){left=a[offset++];if(!left)throw Error('Short GIF frame');}buf|=a[offset++]<<bits;bits+=8;left--;}
    let code=buf&((1<<size)-1);buf>>>=size;bits-=size;if(code===clear){size=f.size+1;next=eof+1;prior=-1;continue;}if(code===eof)break;
    const original=code;let n=0;if(code===next&&prior>=0){stack[n++]=first;code=prior;}if(code>=next)throw Error('Bad GIF dictionary');
    while(code>=clear){stack[n++]=suf[code];code=pre[code];}first=code;stack[n++]=first;
    while(n){const index=stack[--n];if(index!==f.trans){const j=4*((f.y+Math.floor(count/f.w))*this.width+f.x+count%f.w);out[j]=f.pal[index*3];out[j+1]=f.pal[index*3+1];out[j+2]=f.pal[index*3+2];out[j+3]=255;}count++;}
    if(prior>=0&&next<4096){pre[next]=prior;suf[next++]=first;if(next===(1<<size)&&size<12)size++;}prior=original;
   }
   if(count!==f.w*f.h)throw Error('Incomplete GIF frame');
  }return out;
 }
}

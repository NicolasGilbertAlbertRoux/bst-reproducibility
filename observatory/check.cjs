"use strict";
// Exercise the actual event handlers and canvas renderer without requiring a display server.
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const assert = require("node:assert/strict");
const {createCanvas} = require("@napi-rs/canvas");
const markup=fs.readFileSync(path.join(__dirname,"comparisons.html"),"utf8");
const nodes=new Map(), translated=[];
let downloaded=null;
class Element {
  constructor(tag) { this.tag=tag;this.children=[];this.options=[];this._value="";this.dataset={};this.hidden=false;this.disabled=false; }
  append(...children){this.children.push(...children);}
  replaceChildren(){this.children.length=0;this.options.length=0;this._value="";}
  add(option){this.options.push(option);if(this.options.length===1)this.value=option.value;}
  get value(){return this._value;}
  set value(value){this._value=String(value);}
  get selectedOptions(){return this.options.filter(option=>String(option.value)===this.value);}
  click(){if(this.tag==="a")downloaded={url:this.href,name:this.download};}
}
function create(tag){return tag==="canvas"?createCanvas(720,440):new Element(tag);}
global.Option=class {constructor(text,value){this.text=String(text);this.value=String(value);}};
for(const match of markup.matchAll(/<(\w+)\b([^>]*)>/g)){
  const attrs=match[2],id=attrs.match(/\bid="([^"]+)"/),eng=attrs.match(/\bdata-en="([^"]+)"/),french=attrs.match(/\bdata-fr="([^"]+)"/);
  if(!id&&!eng)continue;
  const node=create(match[1]);
  if(id)nodes.set(id[1],node);
  if(eng){node.dataset={en:eng[1],fr:french[1]};translated.push(node);}
}
global.document={documentElement:{lang:""},createElement:create,getElementById:id=>nodes.get(id),querySelectorAll:()=>translated};
global.window=global;
global.requestAnimationFrame=()=>{};
const blobs=new Map();
URL.createObjectURL=blob=>{blobs.set("memory:movie",blob);return "memory:movie";};
URL.revokeObjectURL=url=>blobs.delete(url);
for(const file of ["data.js","gif.js","scenes.js","app.js"]){vm.runInThisContext(fs.readFileSync(path.join(__dirname,file),"utf8"),{filename:file});}
function select(node,value){node.value=value;node.onchange();}
function nested(node,tag){return node.children.flatMap(c=>c.tag===tag?[c]:nested(c,tag));}
const at=id=>nodes.get(id);
async function main(){
  assert.match(at("joint-count").textContent,/^192 \/ 576 /);
  const controls=nested(at("joint-controls"),"select");
  assert.equal(controls.length,6);
  select(controls[0],"C00");assert.match(at("joint-count").textContent,/\/ 48 /);
  select(controls[0],"");assert.match(at("joint-count").textContent,/^192 \/ 576 /);
  const history=vm.runInThisContext("state.history");
  const difference=Math.max(...history.flatMap((r,i)=>r.slice(0,5).map((v,j)=>Math.abs(v-D.baselineBeat[i][j]))));
  assert.ok(difference<1e-9);
  const k=nested(at("physical"),"input")[0];k.value=4;k.oninput();
  assert.notDeepEqual(vm.runInThisContext("state.history.at(-1)"),history.at(-1));at("reset").onclick();
  let total=0,atoms=0,modes=0;
  const domains=at("domain").options.map(o=>o.value);
  for(const domain of domains){
    select(at("domain"),domain);const items=at("item").options.map(o=>o.value);
    for(const id of items){
      select(at("item"),id);at("time").value=65;at("time").oninput();total++;
      assert.ok(at("heading").textContent.length>3);
      if(domain==="atom"){
        atoms++;const lists=nested(at("catalogue"),"select");
        assert.ok(lists.every(s=>s.options.length>0));lists.forEach(n=>select(n,n.options.at(-1).value));
        assert.equal(at("export").disabled,!D.atoms[id]&&!D.modalScenes[id]?.mode);
      }
      const choices=nested(at("options"),"select");
      if(choices.length)for(const option of choices[0].options){select(choices[0],option.value);if(domain==="molecule"&&id==="SF6")modes++;}
    }
  }
  assert.equal(atoms,118);assert.equal(modes,66);
  select(at("domain"),"molecule");select(at("item"),"SF6");
  select(at("language"),"fr");assert.equal(document.documentElement.lang,"fr");assert.match(at("heading").textContent,/moléculaire/);
  assert.ok(translated.every(e=>e.textContent===e.dataset.fr));
  const choose=nested(at("options"),"select")[0];select(choose,"66");
  await at("export").onclick();assert.equal(downloaded.name,"bst-molecule-SF6.gif");
  const buffer=Buffer.from(await blobs.get(downloaded.url).arrayBuffer());
  assert.equal(buffer.subarray(0,6).toString(),"GIF89a");assert.equal(buffer.readUInt16LE(6),432);assert.equal(buffer.readUInt16LE(8),264);
  select(at("language"),"en");assert.equal(document.documentElement.lang,"en");
  console.log(JSON.stringify({domains:domains.length,preparations:total,element_views:atoms,fluoride_modes:modes,beat_max_abs_difference:difference,exported_bytes:buffer.length},null,2));
}
main().then(()=>process.exit(0)).catch(e=>{console.error(e);process.exit(1);});

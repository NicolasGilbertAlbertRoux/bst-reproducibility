"use strict";
// Batch export uses exactly the same scene code, input data and GIF encoder as the HTML page.
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { createCanvas } = require("@napi-rs/canvas");

global.window = global;
for (const name of ["data.js", "gif.js", "scenes.js"]) {
  vm.runInThisContext(fs.readFileSync(path.join(__dirname, name), "utf8"), { filename: name });
}

async function main() {
  const [domain, item, file, ...options] = process.argv.slice(2);
  const config = { mode: 1, angle: 24, language: "en", band: domain === "sky" ? "TT" : "800", stiffness: 3, inertia: 0.3, restoring: 2, sign: 1 };
  if (!file || options.length % 2) throw Error("Usage: node render.cjs DOMAIN ITEM OUTPUT.gif [--mode N] [--band BAND] [--angle DEGREES] [--language en|fr] [--stiffness K] [--inertia M] [--restoring K] [--sign -1|1]");
  for (let i = 0; i < options.length; i += 2) {
    const key = options[i].replace(/^--/, "");
    if (!Object.hasOwn(config, key) || !options[i].startsWith("--")) throw Error("Unknown option: " + options[i]);
    config[key] = typeof config[key] === "number" ? Number(options[i + 1]) : options[i + 1];
  }
  if (!["en","fr"].includes(config.language)) throw Error("Select en or fr");
  function range(value, low, high, label) {
    if (!Number.isFinite(value) || value < low || value > high) throw Error("Outside the supported range: " + label);
  }
  range(config.angle,0,180,"angle");range(config.stiffness,1,5,"stiffness");range(config.inertia,.22,.55,"inertia");range(config.restoring,1.6,2.8,"restoring");
  if (![-1,1].includes(config.sign)) throw Error("The initial sign is -1 or 1");
  const supported = {
    beat: ["pair"],
    atom: [...new Set([...Object.keys(D.atoms), ...Object.keys(D.modalScenes).filter(k=>D.modalScenes[k].mode)])],
    molecule: ["SF6","H2O","CO2"], spectrum: ["SF6","H",...new Set(D.optical.map(r=>r.observation.split(" ")[0]))],
    field: D.field.map((_,i)=>String(i+1)), thermal: ["Kr","Xe","H2O","CO2"], sky: Object.keys(D.sky), nuclear: D.beta.map(r=>r[0])
  };
  if (!supported[domain]?.includes(item)) throw Error("No movie for this object. Available: " + (supported[domain]||Object.keys(supported)).join(", "));
  const count = domain==="molecule" ? (item==="SF6" ? 66 : D.modalScenes[item].currents.length) :
    domain==="field" || (domain==="thermal" && ["Kr","Xe"].includes(item)) ? 4 : 1;
  if (!Number.isInteger(config.mode)) throw Error("The mode is an integer");
  range(config.mode,1,count,"mode");
  const bands = domain==="sky" ? Object.keys(D.sky[item]) : domain==="spectrum" && item==="SF6" ? Object.keys(D.infrared) : null;
  if (bands && !bands.includes(config.band)) throw Error("Supported bands: " + bands.join(", "));
  const en={beat:"Mantle pair",atom:"Atomic contact mode",molecule:"Molecular motion",spectrum:"Spectral observation",field:"Projected current",thermal:"Thermal states",sky:"CMB band spectra",nuclear:"Beta decay"};
  const fr={beat:"Paire de manteaux",atom:"Mode de contact atomique",molecule:"Mouvement moléculaire",spectrum:"Observation spectrale",field:"Courant projeté",thermal:"États thermiques",sky:"Spectres du fond diffus",nuclear:"Désintégration bêta"};
  const state={domain,item,mode:config.mode,angle:config.angle,lang:config.language,band:config.band,
    title:(config.language==="fr"?fr:en)[domain]+" · "+item,history:beatHistory(config.stiffness,config.sign,config.inertia,config.restoring)};
  const canvas=createCanvas(432,264),context=canvas.getContext("2d"),frames=[];
  for(let i=0;i<32;i++){drawScene(canvas,state,i/31);frames.push(context.getImageData(0,0,432,264).data);}
  const result=await encodeGif(frames,432,264,12).arrayBuffer();
  fs.mkdirSync(path.dirname(path.resolve(file)),{recursive:true});
  fs.writeFileSync(file,Buffer.from(result),{flag:"wx"});
}
main().catch(error=>{console.error(error.message);process.exitCode=1;});

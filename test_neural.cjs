const fs=require('fs'),assert=require('node:assert/strict');const {predictNeural}=require('./docs/neural.js');
const model=JSON.parse(fs.readFileSync('artifacts/neural-model.json')),parity=JSON.parse(fs.readFileSync('artifacts/inference-parity.json'));
for(const row of parity){const r=predictNeural(model,row.pixels);assert.equal(r.prediction,row.expected.prediction);r.probabilities.forEach((p,i)=>assert.ok(Math.abs(p-row.expected.probabilities[i])<1e-9));}
assert.throws(()=>predictNeural(model,Array(64).fill(NaN)));console.log('Exported Python weights match JavaScript inference for held-out images');

const assert=require('node:assert/strict');
const {run}=require('./docs/demo.js');
const r=run({threshold:0.5});assert.equal(r.test_samples,60);assert.ok(r.training_loss.at(-1)<r.training_loss[0]);assert.deepEqual(r,run({threshold:0.5}));assert.throws(()=>run({threshold:2}));
console.log('Browser demo behavior checks passed');

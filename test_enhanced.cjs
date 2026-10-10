const assert=require('node:assert/strict');
const {runEnhanced,auc}=require('./enhanced.js');assert.equal(auc([1,0],[.5,.5]),.5);const r=runEnhanced();assert.equal(Object.values(r.metrics.confusion_matrix).reduce((a,b)=>a+b),100);assert.ok(r.metrics.roc_auc>.7);assert.equal(r.calibration.reduce((s,b)=>s+b.count,0),100);
console.log('Enhanced behavior tests passed');

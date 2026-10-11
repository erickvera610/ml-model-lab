/* Inference with parameters exported from the Python-trained neural network. */
function predictNeural(model,pixels){
 if(!Array.isArray(pixels)||pixels.length!==64||pixels.some(p=>!Number.isFinite(p)||p<0||p>16))throw Error('64 finite pixels in [0,16] required');
 let x=pixels.map((v,i)=>(v-model.mean[i])/model.scale[i]);
 model.weights.forEach((weights,layer)=>{x=model.biases[layer].map((b,j)=>b+x.reduce((sum,v,i)=>sum+v*weights[i][j],0));if(layer<model.weights.length-1)x=x.map(v=>Math.max(0,v));});
 const max=Math.max(...x),scores=x.map(v=>Math.exp(v-max)),total=scores.reduce((a,b)=>a+b,0),probabilities=scores.map(v=>v/total);
 return {prediction:model.classes[probabilities.indexOf(Math.max(...probabilities))],probabilities};
}
if(typeof module!=='undefined')module.exports={predictNeural};

function randomGenerator(seed=42) {
    let state=seed>>>0;
    return ()=> {
        state=(1664525*state+1013904223)>>>0;
        return state/4294967296;
    }
    ;
}
function sigmoid(z) {
    return 1/(1+Math.exp(-Math.max(-60,Math.min(60,z))));
}
function classification(actual,scores,threshold) {
    let TP=0,TN=0,FP=0,FN=0;
    actual.forEach((y,i)=> {
        const p=scores[i]>=threshold;if(y&&p)TP++;else if(!y&&!p)TN++;else if(p)FP++;else FN++;
    }
    );
    const precision=TP/(TP+FP)||0,recall=TP/(TP+FN)||0;
    return  {
        accuracy:(TP+TN)/actual.length,precision,recall,f1:2*precision*recall/(precision+recall)||0,confusion_matrix: {
            TP,TN,FP,FN
        }
    }
    ;
}
function auc(actual,scores) {
    const positive=scores.filter((s,i)=>actual[i]),negative=scores.filter((s,i)=>!actual[i]);
    if(!positive.length||!negative.length)return null;
    return positive.reduce((sum,a)=>sum+negative.reduce((s,b)=>s+Number(a>b)+.5*Number(a===b),0),0)/(positive.length*negative.length);
}
function runEnhanced(p= {
}
) {
    const random=randomGenerator(p.seed??42),rows=Array.from( {
        length:500
    }
    ,()=> {
        const x=[1,random()*4-2,random()*4-2];return  {
            x,y:Number(random()<sigmoid(2*x[1]-1.5*x[2]+.3))
        }
        ;
    }
    );
    for(let i=499;i>0;i--) {
        const j=Math.floor(random()*(i+1));
        [rows[i],rows[j]]=[rows[j],rows[i]];
    }
    const training=rows.slice(0,300),validation=rows.slice(300,400),test=rows.slice(400);
    let w=[0,0,0];
    const loss=[];
    for(let epoch=0;epoch<400;epoch++) {
        const g=[0,0,0];
        let l=0;
        for(const row of training) {
            const prob=sigmoid(w.reduce((s,v,j)=>s+v*row.x[j],0));
            row.x.forEach((x,j)=>g[j]+=(prob-row.y)*x);
            l-=row.y*Math.log(Math.max(prob,1e-12))+(1-row.y)*Math.log(Math.max(1-prob,1e-12));
        }
        w=w.map((v,j)=>v-.15*g[j]/300);
        if(epoch%20===0)loss.push(l/300);
    }
    const probabilities=rows=>rows.map(row=>sigmoid(w.reduce((s,v,j)=>s+v*row.x[j],0))),vs=probabilities(validation),vy=validation.map(r=>r.y),thresholds=Array.from( {
        length:19
    }
    ,(_,i)=>(i+1)/20),trials=thresholds.map(threshold=>( {
        threshold,validation_f1:classification(vy,vs,threshold).f1
    }
    )),chosen=trials.reduce((a,b)=>a.validation_f1>=b.validation_f1?a:b).threshold,scores=probabilities(test),actual=test.map(r=>r.y),m=classification(actual,scores,chosen),majority=Number(training.reduce((s,r)=>s+r.y,0)>=150);
    m.roc_auc=auc(actual,scores);
    m.brier_score=scores.reduce((s,v,i)=>s+(v-actual[i])**2,0)/100;
    const calibration=Array.from( {
        length:5
    }
    ,(_,k)=> {
        const ix=scores.map((s,i)=>i).filter(i=>k/5<=scores[i]&&scores[i]<(k+1)/5);return  {
            range:[k/5,(k+1)/5],count:ix.length,mean_probability:ix.length?ix.reduce((s,i)=>s+scores[i],0)/ix.length:null,observed_positive_rate:ix.length?ix.reduce((s,i)=>s+actual[i],0)/ix.length:null
        }
        ;
    }
    );
    return  {
        metrics:m,baseline_accuracy:actual.filter(v=>v===majority).length/100,selected_threshold:chosen,threshold_selection:trials,train_samples:300,validation_samples:100,test_samples:100,training_loss:loss,weights:w,calibration,roc_curve:[0,...thresholds,1].map(threshold=> {
            const c=classification(actual,scores,threshold);return  {
                threshold,tpr:c.recall,fpr:c.confusion_matrix.FP/(c.confusion_matrix.FP+c.confusion_matrix.TN)
            }
            ;
        }
        ),note:'Synthetic data. Validation chooses threshold; test estimates performance. Browser RNG differs from Python.'
    }
    ;
}
if(typeof module!=='undefined')module.exports= {
    runEnhanced,auc,classification
}
;

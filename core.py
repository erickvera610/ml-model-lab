import math,random
def sigmoid(z): return 1/(1+math.exp(-max(-60,min(60,z))))
def train(rows,epochs=400,lr=0.15):
    weights=[0.,0.,0.]; losses=[]
    for epoch in range(epochs):
        gradient=[0.,0.,0.];loss=0.
        for x,y in rows:
            p=sigmoid(sum(w*v for w,v in zip(weights,[1]+x)))
            loss-=y*math.log(max(p,1e-12))+(1-y)*math.log(max(1-p,1e-12))
            for j,v in enumerate([1]+x): gradient[j]+=(p-y)*v
        weights=[w-lr*g/len(rows) for w,g in zip(weights,gradient)]
        if epoch%20==0: losses.append(loss/len(rows))
    return weights,losses
def metrics(actual,predicted):
    tp=sum(a==1 and p==1 for a,p in zip(actual,predicted));tn=sum(a==0 and p==0 for a,p in zip(actual,predicted))
    fp=sum(a==0 and p==1 for a,p in zip(actual,predicted));fn=sum(a==1 and p==0 for a,p in zip(actual,predicted))
    return dict(accuracy=(tp+tn)/len(actual),precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn) if tp+fn else 0,confusion_matrix={'TP':tp,'TN':tn,'FP':fp,'FN':fn})
def run(payload):
    threshold=float(payload.get('threshold',0.5))
    if not 0<threshold<1: raise ValueError('Threshold must be between 0 and 1')
    rng=random.Random(42);rows=[]
    for _ in range(300):
        x=[rng.uniform(-2,2),rng.uniform(-2,2)]
        y=int(rng.random()<sigmoid(2*x[0]-1.5*x[1]+0.3));rows.append((x,y))
    rng.shuffle(rows);training,test=rows[:240],rows[240:]
    weights,loss=train(training)
    predictions=[int(sigmoid(sum(w*v for w,v in zip(weights,[1]+x)))>=threshold) for x,y in test]
    return {'metrics':metrics([y for x,y in test],predictions),'weights':weights,'training_loss':loss,'train_samples':240,'test_samples':60,'baseline_accuracy':max(sum(y for x,y in test),sum(1-y for x,y in test))/len(test),'note':'Synthetic demonstration; not a real-world performance claim.'}

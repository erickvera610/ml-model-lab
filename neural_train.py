"""Train/export a neural network on bundled real digits with auditable splits."""
import json,hashlib,os
from pathlib import Path
from datetime import datetime,timezone
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,log_loss
ROOT=Path(__file__).parent

def export(model):
    scale,net=model.steps[0][1],model.steps[1][1]
    return {'format':'standardized_relu_softmax_mlp_v1','feature_count':64,'pixel_range':[0,16],'mean':scale.mean_.tolist(),'scale':scale.scale_.tolist(),
        'classes':net.classes_.tolist(),'weights':[w.tolist() for w in net.coefs_],'biases':[b.tolist() for b in net.intercepts_],
        'note':'Numeric parameters only. UCI 8x8 digit images; hand-drawn input may differ from training data.'}

def infer(artifact,pixels):
    x=np.asarray(pixels,dtype=float)
    if x.shape!=(64,) or not np.isfinite(x).all() or (x<0).any() or (x>16).any():raise ValueError('Exactly 64 finite pixel intensities in [0,16] required')
    x=(x-np.array(artifact['mean']))/np.array(artifact['scale'])
    for i,(w,b) in enumerate(zip(artifact['weights'],artifact['biases'])):
        x=x@np.array(w)+np.array(b)
        if i<len(artifact['weights'])-1:x=np.maximum(x,0)
    probabilities=np.exp(x-x.max());probabilities/=probabilities.sum()
    return {'prediction':int(artifact['classes'][int(probabilities.argmax())]),'probabilities':probabilities.tolist()}

def train(seed=42):
    data=load_digits()
    tr,rest=train_test_split(np.arange(len(data.target)),test_size=.4,stratify=data.target,random_state=seed)
    va,te=train_test_split(rest,test_size=.5,stratify=data.target[rest],random_state=seed)
    candidates={str(size):make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(size,),activation='relu',max_iter=250,
        early_stopping=True,validation_fraction=.15,n_iter_no_change=20,random_state=seed,batch_size=64)) for size in [32,64]}
    validation={}
    for name,model in candidates.items():
        model.fit(data.data[tr],data.target[tr]);validation[name]=float(f1_score(data.target[va],model.predict(data.data[va]),average='macro'))
    winner=max(validation,key=validation.get);model=candidates[winner]
    fit_ids=np.concatenate([tr,va]);model.fit(data.data[fit_ids],data.target[fit_ids])
    pred=model.predict(data.data[te]);prob=model.predict_proba(data.data[te])
    logistic=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=seed)).fit(data.data[fit_ids],data.target[fit_ids])
    baseline=DummyClassifier(strategy='most_frequent').fit(data.data[tr],data.target[tr]);net=model.steps[-1][1]
    examples=[{'id':int(idx),'pixels':data.data[idx].tolist(),'actual':int(data.target[idx]),'predicted':int(pred[i]),'probability':float(prob[i,pred[i]])} for i,idx in enumerate(te)]
    report={'executed_at_utc':datetime.now(timezone.utc).isoformat(),'seed':seed,'dataset':'UCI Optical Recognition of Handwritten Digits / sklearn load_digits',
        'source':'https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits','sklearn_version':sklearn.__version__,
        'dataset_sha256':hashlib.sha256(data.data.tobytes()+data.target.tobytes()).hexdigest(),
        'split_ids':{'train':tr.tolist(),'validation':va.tolist(),'test':te.tolist()},
        'selection':'Hidden width selected on outer validation macro F1. Early stopping uses internal training-only validation. Test used after selection.',
        'validation_macro_f1':validation,'selected_hidden_width':int(winner),'architecture':[64,int(winner),10],'epochs':int(net.n_iter_),
        'loss_curve':net.loss_curve_,'internal_validation_accuracy':net.validation_scores_,
        'test':{'accuracy':float(accuracy_score(data.target[te],pred)),'macro_f1':float(f1_score(data.target[te],pred,average='macro')),
            'log_loss':float(log_loss(data.target[te],prob)),'confusion_matrix':confusion_matrix(data.target[te],pred).tolist(),
            'constant_baseline_accuracy':float(accuracy_score(data.target[te],baseline.predict(data.data[te]))),
            'logistic_accuracy':float(accuracy_score(data.target[te],logistic.predict(data.data[te])))},
        'examples':examples[:16],'errors':[e for e in examples if e['actual']!=e['predicted']],
        'limitations':'Small historical digits benchmark. Not a road detector, LLM or production deep-learning claim. Multiple seeds assess sensitivity; no test-based tuning.'}
    return report,export(model),model,data,te

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',default='42,7,19');args=parser.parse_args();runs=[];first=None
    for seed in [int(s) for s in args.seeds.split(',')]:
        report,artifact,_,_,_=train(seed);runs.append({'seed':seed,**report['test'],'selected_hidden_width':report['selected_hidden_width']})
        if first is None:first=(report,artifact)
    folder=ROOT/'artifacts';folder.mkdir(exist_ok=True);report,artifact=first
    artifact_json=json.dumps(artifact,separators=(',',':'),allow_nan=False)
    report['model_sha256']=hashlib.sha256(artifact_json.encode()).hexdigest()
    report['seed_sensitivity']={'runs':runs,'mean_accuracy':float(np.mean([r['accuracy'] for r in runs])),'std_accuracy':float(np.std([r['accuracy'] for r in runs]))}
    (folder/'neural-model.json').write_text(artifact_json);(folder/'neural-results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({'architecture':report['architecture'],'accuracy':report['test']['accuracy'],'logistic':report['test']['logistic_accuracy'],'mean_accuracy':report['seed_sensitivity']['mean_accuracy']}))

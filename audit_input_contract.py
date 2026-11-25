"""Reproduce the input-coercion regression and compare Python/JavaScript."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
from neural_train import infer
ROOT=Path(__file__).parent

def main():
    model=json.loads((ROOT/'artifacts/neural-model.json').read_text())
    cases={'numeric_zeros':[0]*64,'numeric_strings':['0']*64,'booleans':[True]*64,'mixed_boolean':[True]+[0]*63,'nulls':[None]*64,'too_short':[0]*63,'out_of_range':[17]*64}
    rows=[]
    for name,pixels in cases.items():
        try:
            old=np.asarray(pixels,dtype=float)
            legacy=bool(old.shape==(64,) and np.isfinite(old).all() and (old>=0).all() and (old<=16).all())
        except (ValueError,TypeError): legacy=False
        try: infer(model,pixels); accepted=True
        except (ValueError,TypeError): accepted=False
        rows.append({'case':name,'legacy_python_accepted':legacy,'python_accepted':accepted})
    js="const fs=require('fs'),{predictNeural}=require('./docs/neural.js');const x=JSON.parse(fs.readFileSync(0,'utf8'));console.log(JSON.stringify(Object.values(x.cases).map(p=>{try{predictNeural(x.model,p);return true}catch(e){return false}})));"
    accepted_js=json.loads(subprocess.check_output(['node','-e',js],input=json.dumps({'model':model,'cases':cases}).encode(),cwd=ROOT))
    for row,accepted in zip(rows,accepted_js): row['javascript_accepted']=accepted
    assert all(r['python_accepted']==r['javascript_accepted'] for r in rows)
    archived=json.loads((ROOT/'artifacts/neural-results.json').read_text())['examples']
    assert all(infer(model,r['pixels'])['prediction']==r['predicted'] for r in archived)
    report={'scope':'Input contract regression; not model accuracy improvement','cases':rows,'archived_predictions_checked':len(archived),'unchanged_predictions':len(archived),'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['neural_train.py','docs/neural.js','artifacts/neural-model.json']}}
    (ROOT/'artifacts/input-contract-results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()

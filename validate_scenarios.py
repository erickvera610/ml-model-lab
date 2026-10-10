"""Ten genuine automated scenarios, not user traffic or contribution history."""
import json, tempfile, os, math, sqlite3
from pathlib import Path
from datetime import datetime, timezone
PROJECT = 'ml-model-lab'
def validate():
    results=[]
    if PROJECT=='mercado-aprende':
        from app import create_app
        def post(client,path,data=None):
            client.get('/login')
            with client.session_transaction() as session: token=session['csrf']
            return client.post(path,data={**(data or {}),'csrf':token})
        for i in range(20):
            with tempfile.TemporaryDirectory() as tmp:
                app=create_app({'TESTING':True,'SECRET_KEY':'isolated-test-only','DATABASE':str(Path(tmp)/'market.db')})
                seller,buyer=app.test_client(),app.test_client()
                for client,user in [(seller,f'seller{i}'),(buyer,f'buyer{i}')]:
                    assert post(client,'/register',{'username':user,'password':'fictional-test-password'}).status_code==302
                    assert post(client,'/login',{'username':user,'password':'fictional-test-password'}).status_code==302
                title=f'Sample product {i+1}'
                data={'title':title,'category':['Deportes','Libros','Electrónica'][i%3],'condition':'Usado','price':f'{10+i*7}.50','description':f'Fictional product for automated scenario {i+1}.'}
                response=post(seller,'/sell',data)
                assert response.status_code==302
                assert title.encode() in buyer.get('/').data
                assert post(buyer,'/items/1/buy').status_code==302
                assert title.encode() in buyer.get('/account').data
                assert title.encode() not in buyer.get('/').data
                assert post(buyer,'/items/1/buy').status_code==409
                results.append({'scenario':i+1,'input':data,'checks':['register','login','publish','catalog','purchase','history','double-purchase rejected'],'passed':True})
    else:
        import core
        with tempfile.TemporaryDirectory() as tmp:
            previous=os.getcwd();os.chdir(tmp)
            try:
                for i in range(20):
                    if PROJECT=='internship-tracker':
                        payload={'company':f'Fictional Company {i+1}','role':['Software Intern','ML Intern','Data Intern'][i%3],'status':core.STATUSES[i%5],'deadline':f'2027-03-{i+1:02}'}
                        out=core.run(payload);assert len(out['applications'])==i+1
                        assert out['applications'][0]['company']==payload['company']
                        summary={'stored_records':len(out['applications']),'counts':out['counts']}
                    elif PROJECT=='ml-model-lab':
                        payload={'threshold':round(.025+i*.05,3)};out=core.run(payload)
                        assert out['train_samples']==240 and out['test_samples']==60
                        assert sum(out['metrics']['confusion_matrix'].values())==60
                        assert out['training_loss'][-1]<out['training_loss'][0]
                        summary=out['metrics']
                    elif PROJECT=='research-search':
                        payload={'query':['machine learning validation','SQLite','logistic classification','software tests','Sobel edges','graph routes','data leakage','REST APIs','convolution','unknownxyz'][i%10]}
                        out=core.run(payload)
                        assert all(0<=r['score']<=1 for r in out['results'])
                        assert out['results'] or i%10==9
                        summary={'ranked_titles':[r['title'] for r in out['results']],'scores':[r['score'] for r in out['results']]}
                    elif PROJECT=='vision-playground':
                        mode=['original','blur','edges'][i%3]
                        image=[[float((x*(i+1)*17+y*11)%256) for x in range(16)] for y in range(16)]
                        payload={'mode':mode,'width':16,'height':16,'synthetic_pattern':i+1}
                        out=core.transform(image,mode)
                        assert len(out)==16 and all(len(r)==16 for r in out)
                        assert all(math.isfinite(v) and 0<=v<=255+1e-9 for r in out for v in r)
                        summary={'mean_intensity':sum(map(sum,out))/256,'min':min(map(min,out)),'max':max(map(max,out))}
                    else:
                        blocked=[[i%10+1,y] for y in range(12) if y!=i%10 and (i<10 or y!=11)]
                        payload={'blocked':blocked};out=core.run(payload)
                        assert out['path'][0]==[0,0] and out['path'][-1]==[11,11]
                        assert not any(p in blocked for p in out['path'])
                        assert all(abs(a[0]-b[0])+abs(a[1]-b[1])==1 for a,b in zip(out['path'],out['path'][1:]))
                        summary={'distance':out['distance'],'visited':out['visited']}
                    results.append({'scenario':i+1,'input':payload,'output_summary':summary,'passed':True})
            finally:os.chdir(previous)
    return {'project':PROJECT,'executed_at_utc':datetime.now(timezone.utc).isoformat(),'kind':'practical exercises using fictional or synthetic data','execution_method':'Python script','scenario_count':len(results),'passed':sum(r['passed'] for r in results),'results':results}
if __name__=='__main__':
    report=validate()
    destination=Path(__file__).parent/'validation-results.json'
    destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f"{PROJECT}: {report['passed']}/{report['scenario_count']} scenarios passed")

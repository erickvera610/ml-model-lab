import json,threading,unittest,urllib.request,urllib.error
from pathlib import Path
from http.server import ThreadingHTTPServer
from app import Handler
class ModelAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler);cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start();cls.url='http://127.0.0.1:'+str(cls.server.server_port)
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def post(self,payload):
        request=urllib.request.Request(self.url+'/api/predict',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request) as response:return json.load(response)
    def test_saved_example_inference(self):
        report=json.loads((Path(__file__).parent/'artifacts'/'neural-results.json').read_text());row=report['examples'][0]
        response=self.post({'pixels':row['pixels']})
        self.assertEqual(response['prediction'],row['predicted'])
        with urllib.request.urlopen(self.url+'/neural') as response:self.assertEqual(response.status,200)
    def test_bad_inputs_fail_and_server_recovers(self):
        for payload in [{},[],{'pixels':None}]:
            with self.assertRaises(urllib.error.HTTPError) as error:self.post(payload)
            self.assertEqual(error.exception.code,400)
        with urllib.request.urlopen(self.url+'/health') as response:self.assertEqual(response.status,200)

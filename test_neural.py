import unittest,json,hashlib
from pathlib import Path
import numpy as np
from neural_train import infer,train
class NeuralTests(unittest.TestCase):
    def test_export_matches_training_library(self):
        report,artifact,model,data,test=train(42)
        for idx in test[:12]:np.testing.assert_allclose(infer(artifact,data.data[idx])['probabilities'],model.predict_proba(data.data[[idx]])[0],atol=1e-10)
        a,b,c=[set(report['split_ids'][k]) for k in ['train','validation','test']]
        self.assertFalse(a&b or a&c or b&c)
        self.assertGreater(report['test']['accuracy'],report['test']['constant_baseline_accuracy'])
    def test_invalid_pixels(self):
        artifact=json.loads((Path(__file__).parent/'artifacts/neural-model.json').read_text())
        for pixels in [[0]*63,[float('nan')]*64,[17]*64,[-1]*64]:
            with self.assertRaises(ValueError):infer(artifact,pixels)
    def test_report_and_export_identity(self):
        folder=Path(__file__).parent/'artifacts';r=json.loads((folder/'neural-results.json').read_text())
        self.assertEqual(hashlib.sha256((folder/'neural-model.json').read_bytes()).hexdigest(),r['model_sha256'])

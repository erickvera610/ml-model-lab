import json
from pathlib import Path
import unittest
import numpy as np
from neural_train import infer

ROOT=Path(__file__).parent
class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=json.loads((ROOT/'artifacts/neural-model.json').read_text())
    def test_rejects_coercible_non_numeric_values(self):
        for pixels in [['0']*64,[True]*64,[True]+[0]*63,[0j]*64,np.array([0]*64,dtype=object)]:
            with self.subTest(kind=str(type(pixels))):
                with self.assertRaises(ValueError): infer(self.model,pixels)
    def test_all_archived_predictions_unchanged(self):
        report=json.loads((ROOT/'artifacts/neural-results.json').read_text())
        for row in report['examples']:
            result=infer(self.model,row['pixels'])
            self.assertEqual(result['prediction'],row['predicted'])
            self.assertAlmostEqual(result['probabilities'][row['predicted']],row['probability'],places=10)
    def test_numeric_arrays_and_lists_agree(self):
        row=json.loads((ROOT/'artifacts/neural-results.json').read_text())['examples'][0]
        expected=infer(self.model,row['pixels'])
        for dtype in [np.int64,np.float64]:
            self.assertEqual(infer(self.model,np.array(row['pixels'],dtype=dtype)),expected)

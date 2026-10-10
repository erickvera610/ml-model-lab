import unittest
from evaluation import run_evaluation, auc

class EvaluationTests(unittest.TestCase):

    def test_auc_with_ties(self):
        self.assertEqual(auc([1, 0], [0.5, 0.5]), 0.5)

    def test_ranking_direction(self):
        self.assertEqual(auc([1, 0], [0.9, 0.1]), 1)

    def test_holdout_metrics(self):
        r = run_evaluation({})
        self.assertEqual(sum(r['metrics']['confusion_matrix'].values()), 100)
        self.assertGreater(r['metrics']['roc_auc'], 0.7)
        self.assertEqual(sum((x['count'] for x in r['calibration'])), 100)

import unittest
from core import sigmoid,train,run,metrics
class Tests(unittest.TestCase):
    def test_numerical_stability(self): self.assertGreater(sigmoid(1000),0.99);self.assertLess(sigmoid(-1000),0.01)
    def test_learning(self):
        w,loss=train([([-1,0],0),([1,0],1)])
        self.assertLess(loss[-1],loss[0]);self.assertGreater(w[1],0)
    def test_confusion(self): self.assertEqual(metrics([1,0],[1,0])['accuracy'],1)
    def test_threshold_validation(self):
        with self.assertRaises(ValueError): run({'threshold':2})
    def test_reproducibility(self): self.assertEqual(run({}),run({}))

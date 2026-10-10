import unittest
from public_benchmark import benchmark
class PublicBenchmarkTests(unittest.TestCase):
    def test_holdout_and_baseline(self):
        r = benchmark()
        a, b, c = [set(r["split_ids"][k]) for k in ["train", "validation", "test"]]
        self.assertFalse(a & b or a & c or b & c)
        self.assertEqual(len(a | b | c), 1797)
        self.assertGreater(r["test_accuracy"], r["baseline_accuracy"])
        self.assertEqual(sum(map(sum, r["confusion_matrix"])), len(c))

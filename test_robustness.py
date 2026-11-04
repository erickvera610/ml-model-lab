"""Regression tests for the robustness protocol (no training required)."""
import unittest
import numpy as np
from robustness_benchmark import corrupt, paired_metrics, CORRUPTION_SEEDS

class RobustnessTests(unittest.TestCase):
    def test_corruptions_preserve_source_shape_and_range(self):
        source = np.arange(128, dtype=float).reshape(2, 64) % 17
        original = source.copy()
        for mode, seed in CORRUPTION_SEEDS.items():
            changed = corrupt(source, mode, np.random.default_rng(seed))
            np.testing.assert_array_equal(source, original)
            self.assertEqual(changed.shape, source.shape)
            self.assertTrue(np.isfinite(changed).all())
            self.assertTrue(((changed >= 0) & (changed <= 16)).all())

    def test_condition_order_does_not_change_inputs(self):
        source = np.full((3, 64), 8.)
        def generate(items):
            return {m: corrupt(source, m, np.random.default_rng(s)) for m, s in items}
        forward = generate(CORRUPTION_SEEDS.items())
        backward = generate(reversed(list(CORRUPTION_SEEDS.items())))
        for mode in forward:
            np.testing.assert_array_equal(forward[mode], backward[mode])

    def test_exact_occlusion_and_brightness(self):
        x = np.full((1, 64), 10, dtype=int)
        expected = x.astype(float).reshape(1, 8, 8)
        expected[:, 3:5, 3:5] = 0
        np.testing.assert_array_equal(corrupt(x, 'center_occlusion', np.random.default_rng(0)), expected.reshape(1, 64))
        np.testing.assert_array_equal(corrupt(x, 'brightness_70pct', np.random.default_rng(0)), np.full((1, 64), 7.))

    def test_reject_invalid_pixels(self):
        for x in [np.zeros(64), np.empty((0, 64)), np.zeros((2, 63)), np.full((1, 64), np.nan), np.full((1, 64), 17)]:
            with self.assertRaises(ValueError):
                corrupt(x, 'gaussian_noise', np.random.default_rng(0))
        with self.assertRaises(ValueError):
            corrupt(np.zeros((1, 64)), 'unknown', np.random.default_rng(0))

    def test_paired_counts_explain_net_drop(self):
        result = paired_metrics([0, 1, 2, 3], [0, 1, 0, 0], [1, 1, 2, 0])
        self.assertEqual(result['correct_to_wrong'], 1)
        self.assertEqual(result['wrong_to_correct'], 1)
        self.assertEqual(result['prediction_changes'], 2)
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['per_class_recall']['2'], 1.)

    def test_reject_mismatched_pairs(self):
        with self.assertRaises(ValueError):
            paired_metrics([0, 1], [0], [0, 1])

if __name__ == '__main__':
    unittest.main()

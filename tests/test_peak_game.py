import unittest

import numpy as np

from peak_divergence.core import PeakGameConfig, PeakLandscape
from peak_divergence.game import level_to_capacity, run_game, score_positions, value_positions
from peak_divergence.strategies import make_population


class PeakDivergenceGameTests(unittest.TestCase):
    def test_value_at_peak_center_equals_height(self):
        landscape = PeakLandscape(
            centers=np.array([[50.0, 50.0]]),
            heights=np.array([100.0]),
            widths=np.array([10.0]),
        )
        positions = np.array([[50.0, 50.0]])
        values, peak_ids, _ = value_positions(positions, landscape)
        self.assertTrue(np.allclose(values, [100.0]))
        self.assertEqual(int(peak_ids[0]), 0)

    def test_multiplicative_score_uses_value_gate(self):
        config = PeakGameConfig(num_agents=2, dimensions=2, beta_diversity=0.01, gamma_origin=0.01)
        landscape = PeakLandscape(
            centers=np.array([[0.0, 0.0]]),
            heights=np.array([0.0]),
            widths=np.array([10.0]),
        )
        positions = np.array([[100.0, 100.0], [0.0, 0.0]])
        scores, values, diversity, origin, _, _ = score_positions(positions, landscape, config)
        self.assertTrue(np.allclose(values, 0.0))
        self.assertTrue(np.allclose(scores, 0.0))
        self.assertGreater(diversity[0], 0.0)
        self.assertGreater(origin[0], 0.0)

    def test_capacity_levels(self):
        self.assertEqual(level_to_capacity(0, 8), 0)
        self.assertEqual(level_to_capacity(5, 8), 5)
        self.assertEqual(level_to_capacity(100, 8), 7)
        self.assertEqual(level_to_capacity("all", 8), 7)

    def test_run_game_is_deterministic_for_seed(self):
        config = PeakGameConfig(num_agents=16, dimensions=4, rounds=3, num_peaks=4)
        first = run_game(make_population("independent_search", 16), config, seed=5)
        second = run_game(make_population("independent_search", 16), config, seed=5)
        self.assertTrue(np.allclose(first.positions, second.positions))
        self.assertTrue(np.allclose(first.final_scores, second.final_scores))


if __name__ == "__main__":
    unittest.main()

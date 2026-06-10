import unittest

import numpy as np

from hypercube_divergence.core import CollaborationAction, GameConfig
from hypercube_divergence.game import level_to_capacity, run_game, score_positions
from hypercube_divergence.strategies import make_population


class HypercubeGameTests(unittest.TestCase):
    def test_origin_maximizer_score_has_no_diversity(self):
        positions = np.full((5, 10), 100.0)
        scores, diversity, origin = score_positions(positions, lambda_origin=0.35)
        self.assertTrue(np.allclose(diversity, 0.0))
        self.assertTrue(np.allclose(origin, 100.0))
        self.assertTrue(np.allclose(scores, 35.0))

    def test_capacity_levels(self):
        self.assertEqual(level_to_capacity(0, 10), 0)
        self.assertEqual(level_to_capacity(5, 10), 5)
        self.assertEqual(level_to_capacity(100, 10), 9)
        self.assertEqual(level_to_capacity("all", 10), 9)

    def test_run_game_is_deterministic_for_seed(self):
        config = GameConfig(num_agents=20, rounds=3)
        first = run_game(make_population("corner_random", 20), config, seed=123)
        second = run_game(make_population("corner_random", 20), config, seed=123)
        self.assertTrue(np.allclose(first.positions, second.positions))
        self.assertTrue(np.allclose(first.final_scores, second.final_scores))

    def test_round_metrics_include_collaboration_capacities(self):
        config = GameConfig(num_agents=20, rounds=2)
        result = run_game(make_population("full_collaboration", 20), config, seed=7)
        self.assertEqual(len(result.round_metrics), 2)
        self.assertEqual(result.round_metrics[0]["mean_share_capacity"], 19.0)
        self.assertEqual(result.round_metrics[0]["mean_read_capacity"], 19.0)


if __name__ == "__main__":
    unittest.main()

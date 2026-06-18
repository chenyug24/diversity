import unittest

import numpy as np

from peak_divergence.core import PeakGameConfig, PeakLandscape
from peak_divergence.game import level_to_capacity, run_game, score_positions, value_positions
from peak_divergence.llm_agent import build_llm_prompt, parse_llm_decision
from peak_divergence.strategies import Strategy, make_population


class ObservationInspectorStrategy(Strategy):
    name = "observation_inspector"

    def update_position(self, observation, rng, config):
        assert not hasattr(observation, "own_value")
        assert not hasattr(observation, "own_diversity")
        assert not hasattr(observation, "own_origin")
        assert not hasattr(observation, "observed_values")
        return observation.own_position


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

    def test_agent_observation_is_black_box(self):
        config = PeakGameConfig(num_agents=6, dimensions=3, rounds=2, num_peaks=2)
        run_game([ObservationInspectorStrategy() for _ in range(6)], config, seed=12)

    def test_parse_llm_decision_clips_position(self):
        config = PeakGameConfig(num_agents=4, dimensions=3)
        decision = parse_llm_decision(
            '{"position":[-5,50,123],"next_share":5,"next_read":"all"}',
            config,
        )
        self.assertTrue(np.allclose(decision.position, [0.0, 50.0, 100.0]))
        self.assertEqual(decision.next_share, 5)
        self.assertEqual(decision.next_read, "all")

    def test_llm_prompt_does_not_reveal_score_formula(self):
        config = PeakGameConfig(num_agents=6, dimensions=3, rounds=2, num_peaks=2)
        captured = {}

        class PromptInspectorStrategy(Strategy):
            def update_position(self, observation, rng, config):
                captured["prompt"] = build_llm_prompt(observation, config, [])
                return observation.own_position

        run_game([PromptInspectorStrategy() for _ in range(6)], config, seed=14)
        prompt = captured["prompt"]
        self.assertIn("You do not know the scoring formula", prompt)
        self.assertNotIn("beta_diversity", prompt)
        self.assertNotIn("gamma_origin", prompt)


if __name__ == "__main__":
    unittest.main()

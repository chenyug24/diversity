import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

import numpy as np

from peak_divergence.core import PeakGameConfig, PeakLandscape
from peak_divergence.game import (
    level_to_capacity,
    run_game,
    score_positions,
    score_upper_bound,
    system_optimization_index,
    value_positions,
)
from peak_divergence.llm_agent import build_llm_prompt, load_local_env, parse_llm_decision
from peak_divergence.strategies import Strategy, make_population


class ObservationInspectorStrategy(Strategy):
    name = "observation_inspector"

    def update_position(self, observation, rng, config):
        assert not hasattr(observation, "own_value")
        assert not hasattr(observation, "own_diversity")
        assert not hasattr(observation, "own_origin")
        assert not hasattr(observation, "observed_values")
        return observation.own_position


class BarrierStrategy(Strategy):
    name = "barrier_strategy"

    def __init__(self, barrier):
        self.barrier = barrier

    def update_position(self, observation, rng, config):
        self.barrier.wait(timeout=1.0)
        return observation.own_position


class PeakThenLeaveStrategy(Strategy):
    name = "peak_then_leave"

    def initial_position(self, agent_id, rng, config):
        if agent_id == 0:
            return np.array([0.0, 0.0])
        return np.array([100.0, 100.0])

    def update_position(self, observation, rng, config):
        return np.array([100.0, 100.0])


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

    def test_value_only_score_ignores_diversity(self):
        config = PeakGameConfig(num_agents=2, dimensions=2, beta_diversity=0.01, gamma_origin=0.01)
        landscape = PeakLandscape(
            centers=np.array([[0.0, 0.0]]),
            heights=np.array([0.0]),
            widths=np.array([10.0]),
        )
        positions = np.array([[100.0, 100.0], [0.0, 0.0]])
        scores, values, diversity, origin, _, _ = score_positions(positions, landscape, config)
        self.assertTrue(np.allclose(values, 0.0))
        self.assertGreater(diversity[0], 0.0)
        self.assertTrue(np.allclose(scores, values))
        self.assertGreater(origin[0], 0.0)

    def test_origin_and_beta_do_not_affect_score(self):
        config = PeakGameConfig(num_agents=2, dimensions=2, beta_diversity=0.01, gamma_origin=100.0)
        landscape = PeakLandscape(
            centers=np.array([[0.0, 0.0], [100.0, 100.0]]),
            heights=np.array([10.0, 10.0]),
            widths=np.array([1.0, 1.0]),
        )
        positions = np.array([[0.0, 0.0], [100.0, 100.0]])
        scores, values, diversity, origin, _, _ = score_positions(positions, landscape, config)
        self.assertTrue(np.allclose(values, [10.0, 10.0]))
        self.assertTrue(np.allclose(diversity, [100.0, 100.0]))
        self.assertTrue(np.allclose(scores, [10.0, 10.0]))
        self.assertEqual(score_upper_bound(landscape, config), 10.0)
        self.assertEqual(system_optimization_index(scores, landscape, config), 1.0)
        self.assertNotEqual(float(origin[0]), float(origin[1]))

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
        self.assertEqual(len(first.position_history), config.rounds + 1)
        self.assertTrue(np.allclose(first.position_history[-1], first.positions))
        self.assertEqual(len(first.action_history), config.rounds)
        self.assertEqual(len(first.negotiation_history), config.rounds)
        self.assertEqual(len(first.communication_history), config.rounds)
        self.assertEqual(len(first.observed_count_history), config.rounds)

    def test_final_summary_tracks_best_value_found_across_rounds(self):
        config = PeakGameConfig(num_agents=2, dimensions=2, rounds=1, num_peaks=1)
        landscape = PeakLandscape(
            centers=np.array([[0.0, 0.0]]),
            heights=np.array([100.0]),
            widths=np.array([1.0]),
        )
        result = run_game(
            [PeakThenLeaveStrategy(), PeakThenLeaveStrategy()],
            config=config,
            seed=7,
            landscape=landscape,
        )
        summary = result.final_summary()
        self.assertLess(summary["best_value"], 1.0)
        self.assertEqual(summary["best_value_found"], 100.0)
        self.assertEqual(summary["best_value_found_ratio"], 1.0)
        self.assertEqual(summary["best_value_found_round"], 0.0)
        self.assertEqual(summary["best_value_found_agent_id"], 0.0)

    def test_agent_updates_are_parallel_within_round(self):
        num_agents = 4
        barrier = Barrier(num_agents)
        config = PeakGameConfig(num_agents=num_agents, dimensions=2, rounds=1, num_peaks=2)
        run_game([BarrierStrategy(barrier) for _ in range(num_agents)], config, seed=22)

    def test_agent_observation_is_black_box(self):
        config = PeakGameConfig(num_agents=6, dimensions=3, rounds=2, num_peaks=2)
        run_game([ObservationInspectorStrategy() for _ in range(6)], config, seed=12)

    def test_parse_llm_decision_clips_position(self):
        config = PeakGameConfig(num_agents=4, dimensions=3)
        decision = parse_llm_decision(
            (
                '{"position":[-5,50,123],"next_visibility":1,'
                '"next_request_count":"all","offer_reciprocal":true,'
                '"accept_probability":0.75}'
            ),
            config,
        )
        self.assertTrue(np.allclose(decision.position, [0.0, 50.0, 100.0]))
        self.assertEqual(decision.next_visibility, 1)
        self.assertEqual(decision.next_request_count, "all")
        self.assertTrue(decision.offer_reciprocal)
        self.assertEqual(decision.accept_probability, 0.75)

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
        self.assertIn("negotiation phase", prompt)
        self.assertNotIn("beta_diversity", prompt)
        self.assertNotIn("gamma_origin", prompt)

    def test_llm_prompt_supports_incentive_conditions(self):
        config = PeakGameConfig(num_agents=3, dimensions=2, rounds=1, num_peaks=1)
        observation = type(
            "Observation",
            (),
            {
                "round_index": 0,
                "own_position": np.array([1.0, 2.0]),
                "own_score": 3.0,
                "observed_ids": np.array([], dtype=int),
                "observed_positions": np.empty((0, 2)),
                "observed_scores": np.empty(0),
            },
        )()
        cooperative = build_llm_prompt(observation, config, [], incentive="cooperative")
        competitive = build_llm_prompt(observation, config, [], incentive="competitive")
        self.assertIn("Cooperative objective", cooperative)
        self.assertIn("whole group", cooperative)
        self.assertIn("Competitive objective", competitive)
        self.assertIn("outperform", competitive)

    def test_load_local_env_reads_missing_keys(self):
        with TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text('OPENAI_API_KEY="sk-test"\nOPENAI_AGENT_MODEL=gpt-test\n')
            with patch.dict("os.environ", {}, clear=True):
                load_local_env(env_path)
                import os

                self.assertEqual(os.getenv("OPENAI_API_KEY"), "sk-test")
                self.assertEqual(os.getenv("OPENAI_AGENT_MODEL"), "gpt-test")


if __name__ == "__main__":
    unittest.main()

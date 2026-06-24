#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from peak_divergence.experiments import DEFAULT_STRATEGIES, run_homogeneous_suite
from peak_divergence.strategies import available_strategies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Peak-Divergence Game baseline experiments."
    )
    parser.add_argument("--agents", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=14)
    parser.add_argument("--peaks", type=int, default=12)
    parser.add_argument(
        "--beta-diversity",
        type=float,
        default=0.0,
        help="Deprecated compatibility option; score is fixed to S = V.",
    )
    parser.add_argument(
        "--gamma-origin",
        type=float,
        default=0.0,
        help="Deprecated compatibility option; origin distance is logged but not rewarded.",
    )
    parser.add_argument("--dimensions", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/default"))
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(DEFAULT_STRATEGIES),
        choices=available_strategies(),
    )
    parser.add_argument("--observation-noise", type=float, default=0.0)
    parser.add_argument("--delayed-observation", action="store_true")
    parser.add_argument("--sequential-agent-updates", action="store_true")
    parser.add_argument("--max-parallel-agent-updates", type=int, default=None)
    parser.add_argument("--write-agent-scores", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    summary = run_homogeneous_suite(
        out_dir=args.out,
        num_agents=args.agents,
        rounds=args.rounds,
        num_peaks=args.peaks,
        beta_diversity=args.beta_diversity,
        gamma_origin=args.gamma_origin,
        dimensions=args.dimensions,
        seeds=seeds,
        strategies=args.strategies,
        observation_noise=args.observation_noise,
        delayed_observation=args.delayed_observation,
        parallel_agent_updates=not args.sequential_agent_updates,
        max_parallel_agent_updates=args.max_parallel_agent_updates,
        write_agent_scores=args.write_agent_scores,
    )

    print(f"Wrote results to {args.out}")
    print()
    for strategy, metrics in sorted(
        summary.items(),
        key=lambda item: item[1]["best_value_found_ratio_mean"],
        reverse=True,
    ):
        print(
            f"{strategy:25s} "
            f"best_found={metrics['best_value_found_mean']:.3f} "
            f"best_found_opt={100.0 * metrics['best_value_found_ratio_mean']:.1f}% "
            f"best_found_gap={metrics['best_value_found_gap_mean']:.3f} "
            f"final_mean={metrics['mean_score_mean']:.3f} "
            f"final_best_opt={100.0 * metrics['best_value_ratio_mean']:.1f}% "
            f"diversity={metrics['mean_diversity_mean']:.3f} "
            f"coverage={metrics['peak_coverage_mean']:.1f} "
            f"max_occ={metrics['max_peak_occupancy_mean']:.1f}"
        )


if __name__ == "__main__":
    main()

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
    parser.add_argument("--beta-diversity", type=float, default=0.015)
    parser.add_argument("--gamma-origin", type=float, default=0.010)
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
        write_agent_scores=args.write_agent_scores,
    )

    print(f"Wrote results to {args.out}")
    print()
    for strategy, metrics in sorted(
        summary.items(),
        key=lambda item: item[1]["mean_score_mean"],
        reverse=True,
    ):
        print(
            f"{strategy:25s} "
            f"score={metrics['mean_score_mean']:.3f} "
            f"value={metrics['mean_value_mean']:.3f} "
            f"diversity={metrics['mean_diversity_mean']:.3f} "
            f"coverage={metrics['peak_coverage_mean']:.1f} "
            f"max_occ={metrics['max_peak_occupancy_mean']:.1f}"
        )


if __name__ == "__main__":
    main()

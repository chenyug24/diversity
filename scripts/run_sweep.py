#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from peak_divergence.experiments import DEFAULT_STRATEGIES, run_homogeneous_suite
from peak_divergence.strategies import available_strategies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep Peak-Divergence Game parameters."
    )
    parser.add_argument("--agents", type=int, nargs="+", default=[200])
    parser.add_argument("--peaks", type=int, nargs="+", default=[6, 12, 24])
    parser.add_argument("--rounds", type=int, default=14)
    parser.add_argument(
        "--gamma-origin",
        type=float,
        default=0.0,
        help="Deprecated compatibility option; origin distance is logged but not rewarded.",
    )
    parser.add_argument("--dimensions", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/sweep"))
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    rows: list[dict[str, float | int | str]] = []

    for agents in args.agents:
        for peaks in args.peaks:
            run_dir = args.out / f"agents_{agents}_peaks_{peaks}_value_only"
            summary = run_homogeneous_suite(
                out_dir=run_dir,
                num_agents=agents,
                rounds=args.rounds,
                num_peaks=peaks,
                beta_diversity=0.0,
                gamma_origin=args.gamma_origin,
                dimensions=args.dimensions,
                seeds=seeds,
                strategies=args.strategies,
                observation_noise=args.observation_noise,
                delayed_observation=args.delayed_observation,
                parallel_agent_updates=not args.sequential_agent_updates,
                max_parallel_agent_updates=args.max_parallel_agent_updates,
            )
            for strategy, metrics in summary.items():
                rows.append(
                    {
                        "agents": agents,
                        "peaks": peaks,
                        "beta_diversity": 0.0,
                        "diversity_rewarded": False,
                        "strategy": strategy,
                        **metrics,
                    }
                )

    output_path = args.out / "sweep_summary.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote sweep to {args.out}")
    print(f"Combined summary: {output_path}")


if __name__ == "__main__":
    main()

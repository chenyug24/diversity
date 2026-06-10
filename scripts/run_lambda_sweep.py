#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypercube_divergence.experiments import DEFAULT_STRATEGIES, run_homogeneous_suite
from hypercube_divergence.strategies import available_strategies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep lambda_origin for the Hypercube Divergence Game."
    )
    parser.add_argument("--agents", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--dimensions", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.35, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--out", type=Path, default=Path("results/lambda_sweep"))
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(DEFAULT_STRATEGIES),
        choices=available_strategies(),
    )
    parser.add_argument("--observation-noise", type=float, default=0.0)
    parser.add_argument("--delayed-observation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    rows: list[dict[str, float | str]] = []

    for lambda_origin in args.lambdas:
        safe_lambda = str(lambda_origin).replace(".", "_")
        run_dir = args.out / f"lambda_{safe_lambda}"
        summary = run_homogeneous_suite(
            out_dir=run_dir,
            num_agents=args.agents,
            rounds=args.rounds,
            lambda_origin=lambda_origin,
            dimensions=args.dimensions,
            seeds=seeds,
            strategies=args.strategies,
            observation_noise=args.observation_noise,
            delayed_observation=args.delayed_observation,
        )
        for strategy, metrics in summary.items():
            rows.append(
                {
                    "lambda_origin": lambda_origin,
                    "strategy": strategy,
                    **metrics,
                }
            )

    output_path = args.out / "sweep_summary.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote lambda sweep to {args.out}")
    print(f"Combined summary: {output_path}")


if __name__ == "__main__":
    main()

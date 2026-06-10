from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

from .core import GameConfig
from .game import run_game
from .strategies import available_strategies, make_population


DEFAULT_STRATEGIES = (
    "independent",
    "origin_maximizer",
    "corner_random",
    "full_collaboration",
    "diversity_only",
    "random_collaboration",
    "strategic_collaboration",
)


def run_homogeneous_suite(
    *,
    out_dir: Path,
    num_agents: int = 200,
    rounds: int = 12,
    lambda_origin: float = 0.35,
    seeds: Iterable[int] = range(10),
    strategies: Iterable[str] = DEFAULT_STRATEGIES,
    dimensions: int = 10,
    observation_noise: float = 0.0,
    delayed_observation: bool = False,
    write_agent_scores: bool = False,
) -> dict[str, dict[str, float]]:
    """Run each strategy as a homogeneous population over repeated seeds."""

    out_dir.mkdir(parents=True, exist_ok=True)
    final_rows: list[dict[str, float | int | str]] = []
    round_rows: list[dict[str, float | int | str]] = []
    agent_rows: list[dict[str, float | int | str]] = []

    for strategy_name in strategies:
        for seed in seeds:
            config = GameConfig(
                num_agents=num_agents,
                dimensions=dimensions,
                rounds=rounds,
                lambda_origin=lambda_origin,
                observation_noise=observation_noise,
                delayed_observation=delayed_observation,
            )
            result = run_game(
                make_population(strategy_name, num_agents),
                config=config,
                seed=int(seed),
            )
            final_summary = result.final_summary()
            final_rows.append(
                {
                    "strategy": strategy_name,
                    "seed": int(seed),
                    "num_agents": num_agents,
                    "rounds": rounds,
                    "lambda_origin": lambda_origin,
                    **final_summary,
                }
            )
            for metrics in result.round_metrics:
                round_rows.append(
                    {
                        "strategy": strategy_name,
                        "seed": int(seed),
                        "num_agents": num_agents,
                        "rounds": rounds,
                        "lambda_origin": lambda_origin,
                        **metrics,
                    }
                )
            if write_agent_scores:
                for record in result.agent_records:
                    agent_rows.append(
                        {
                            "strategy_run": strategy_name,
                            "seed": int(seed),
                            **record,
                        }
                    )

    _write_csv(out_dir / "final_metrics.csv", final_rows)
    _write_csv(out_dir / "round_metrics.csv", round_rows)
    if write_agent_scores:
        _write_csv(out_dir / "agent_scores.csv", agent_rows)

    summary = _aggregate(final_rows, group_key="strategy")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "summary.md").write_text(_summary_markdown(summary) + "\n")
    return summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(
    rows: list[dict[str, object]],
    group_key: str,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)

    metrics = [
        "mean_score",
        "best_score",
        "mean_diversity",
        "mean_origin",
        "mean_pairwise_distance",
        "corner_coverage",
        "max_corner_occupancy",
        "corner_entropy",
    ]
    summary: dict[str, dict[str, float]] = {}
    for group, group_rows in grouped.items():
        summary[group] = {}
        for metric in metrics:
            values = [float(row[metric]) for row in group_rows]
            summary[group][f"{metric}_mean"] = mean(values)
            summary[group][f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
    return summary


def _summary_markdown(summary: dict[str, dict[str, float]]) -> str:
    ordered = sorted(
        summary.items(),
        key=lambda item: item[1]["mean_score_mean"],
        reverse=True,
    )
    lines = [
        "# Hypercube Divergence Summary",
        "",
        "| Strategy | Mean score | Pairwise | Origin | Coverage | Max occupancy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy, metrics in ordered:
        lines.append(
            "| "
            + " | ".join(
                [
                    strategy,
                    f"{metrics['mean_score_mean']:.3f} +/- {metrics['mean_score_std']:.3f}",
                    f"{metrics['mean_pairwise_distance_mean']:.3f}",
                    f"{metrics['mean_origin_mean']:.3f}",
                    f"{metrics['corner_coverage_mean']:.1f}",
                    f"{metrics['max_corner_occupancy_mean']:.1f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)

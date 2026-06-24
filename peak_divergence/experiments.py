from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

from .core import DEPRECATED_DIVERSITY_WEIGHT, PeakGameConfig
from .game import run_game
from .strategies import available_strategies, make_population


DEFAULT_STRATEGIES = (
    "random",
    "random_corner",
    "origin_maximizer",
    "independent_search",
    "score_following",
    "diversity_only",
    "full_collaboration",
    "random_collaboration",
    "strategic_collaboration",
)


def run_homogeneous_suite(
    *,
    out_dir: Path,
    num_agents: int = 200,
    rounds: int = 14,
    num_peaks: int = 12,
    beta_diversity: float = DEPRECATED_DIVERSITY_WEIGHT,
    gamma_origin: float = 0.0,
    seeds: Iterable[int] = range(10),
    strategies: Iterable[str] = DEFAULT_STRATEGIES,
    dimensions: int = 10,
    observation_noise: float = 0.0,
    delayed_observation: bool = False,
    parallel_agent_updates: bool = True,
    max_parallel_agent_updates: int | None = None,
    write_agent_scores: bool = False,
) -> dict[str, dict[str, float]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed_beta = DEPRECATED_DIVERSITY_WEIGHT
    final_rows: list[dict[str, float | int | str]] = []
    round_rows: list[dict[str, float | int | str]] = []
    peak_rows: list[dict[str, float | int | str]] = []
    agent_rows: list[dict[str, float | int | str]] = []

    for strategy_name in strategies:
        for seed in seeds:
            config = PeakGameConfig(
                num_agents=num_agents,
                dimensions=dimensions,
                rounds=rounds,
                num_peaks=num_peaks,
                beta_diversity=fixed_beta,
                gamma_origin=gamma_origin,
                observation_noise=observation_noise,
                delayed_observation=delayed_observation,
                parallel_agent_updates=parallel_agent_updates,
                max_parallel_agent_updates=max_parallel_agent_updates,
            )
            result = run_game(
                make_population(strategy_name, num_agents),
                config=config,
                seed=int(seed),
            )
            final_rows.append(
                {
                    "strategy": strategy_name,
                    "seed": int(seed),
                    "num_agents": num_agents,
                    "rounds": rounds,
                    "num_peaks": num_peaks,
                    "beta_diversity": fixed_beta,
                    "gamma_origin": gamma_origin,
                    "origin_rewarded": False,
                    "diversity_rewarded": False,
                    **result.final_summary(),
                }
            )
            for metrics in result.round_metrics:
                round_rows.append(
                    {
                        "strategy": strategy_name,
                        "seed": int(seed),
                        "num_agents": num_agents,
                        "rounds": rounds,
                        "num_peaks": num_peaks,
                        "beta_diversity": fixed_beta,
                        "gamma_origin": gamma_origin,
                        "origin_rewarded": False,
                        "diversity_rewarded": False,
                        **metrics,
                    }
                )
            for peak_id in range(num_peaks):
                peak_rows.append(
                    {
                        "strategy": strategy_name,
                        "seed": int(seed),
                        "peak_id": peak_id,
                        "height": float(result.landscape.heights[peak_id]),
                        "width": float(result.landscape.widths[peak_id]),
                        **{
                            f"mu{k}": float(result.landscape.centers[peak_id, k])
                            for k in range(dimensions)
                        },
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
    _write_csv(out_dir / "peaks.csv", peak_rows)
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
        "total_score",
        "system_optimization",
        "global_optimum",
        "best_value_ratio",
        "optimality_gap",
        "best_score_found",
        "best_value_found",
        "best_value_found_ratio",
        "best_value_found_gap",
        "best_value_found_round",
        "best_score",
        "mean_value",
        "best_value",
        "mean_diversity",
        "mean_origin",
        "mean_pairwise_distance",
        "peak_coverage",
        "max_peak_occupancy",
        "peak_entropy",
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
        key=lambda item: item[1]["best_value_found_ratio_mean"],
        reverse=True,
    )
    lines = [
        "# Peak-Divergence Summary",
        "",
        "| Strategy | Best found | Best found opt. | Best found gap | Final mean | Final best opt. | Diversity | Peak coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy, metrics in ordered:
        lines.append(
            "| "
            + " | ".join(
                [
                    strategy,
                    f"{metrics['best_value_found_mean']:.3f} +/- {metrics['best_value_found_std']:.3f}",
                    f"{100.0 * metrics['best_value_found_ratio_mean']:.1f}%",
                    f"{metrics['best_value_found_gap_mean']:.3f}",
                    f"{metrics['mean_score_mean']:.3f}",
                    f"{100.0 * metrics['best_value_ratio_mean']:.1f}%",
                    f"{metrics['mean_diversity_mean']:.3f}",
                    f"{metrics['peak_coverage_mean']:.1f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def validate_strategy_names(strategies: Iterable[str]) -> None:
    available = set(available_strategies())
    unknown = sorted(set(strategies) - available)
    if unknown:
        raise ValueError(f"Unknown strategies: {unknown}. Available: {sorted(available)}")

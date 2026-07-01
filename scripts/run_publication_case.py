#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from peak_divergence.core import PeakGameConfig
from peak_divergence.game import top_peak_ids, value_positions
from peak_divergence.publication_game import run_publication_game
from peak_divergence.strategies import available_strategies, make_population


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the public research publication scenario: publishing is optional, "
            "there is no private exchange, and exact published locations cannot be reused."
        )
    )
    parser.add_argument("--strategy", choices=available_strategies(), default="score_following")
    parser.add_argument("--agents", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--peaks", type=int, default=2)
    parser.add_argument("--dimensions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--peak-height-min", type=float, default=70.0)
    parser.add_argument("--peak-height-max", type=float, default=120.0)
    parser.add_argument("--peak-width-min", type=float, default=28.0)
    parser.add_argument("--peak-width-max", type=float, default=48.0)
    parser.add_argument("--top-peak-count", type=int, default=3)
    parser.add_argument("--top-peak-discovery-fraction", type=float, default=0.70)
    parser.add_argument("--sequential-agent-updates", action="store_true")
    parser.add_argument("--max-parallel-agent-updates", type=int, default=None)
    parser.add_argument("--exact-match-atol", type=float, default=1e-9)
    parser.add_argument("--max-blocked-resubmissions", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("results/publication_case"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    config = PeakGameConfig(
        num_agents=args.agents,
        dimensions=args.dimensions,
        rounds=args.rounds,
        num_peaks=args.peaks,
        peak_height_range=(args.peak_height_min, args.peak_height_max),
        peak_width_range=(args.peak_width_min, args.peak_width_max),
        top_peak_count=args.top_peak_count,
        top_peak_discovery_fraction=args.top_peak_discovery_fraction,
        parallel_agent_updates=not args.sequential_agent_updates,
        max_parallel_agent_updates=args.max_parallel_agent_updates,
    )
    result = run_publication_game(
        make_population(args.strategy, args.agents),
        config=config,
        seed=args.seed,
        exact_match_atol=args.exact_match_atol,
        max_blocked_resubmissions=args.max_blocked_resubmissions,
    )

    write_summary(args.out / "summary.json", result, args.strategy)
    write_landscape(args.out / "hidden_landscape_for_evaluator.csv", result)
    write_public_registry(args.out / "public_registry.csv", result)
    write_publication_decisions(args.out / "publication_decisions.csv", result)
    write_metrics(args.out / "metrics_by_round.csv", result)
    if args.dimensions == 2:
        plot_2d_publication(args.out / "publication_movement.png", result)
    plot_publication_metrics(args.out / "publication_metrics.png", result)

    summary = result.final_summary()
    print(f"Wrote public publication case to {args.out}")
    print(
        "best_found="
        f"{summary['best_value_found']:.3f} "
        f"ratio={summary['best_value_found_ratio']:.3f} "
        f"top_peaks={summary['top_peak_coverage_count']:.0f}/"
        f"{summary['top_peak_count']:.0f} "
        f"published={summary['published_count']:.0f}"
    )
    print(f"public registry: {args.out / 'public_registry.csv'}")
    print(f"publication decisions: {args.out / 'publication_decisions.csv'}")


def write_summary(path: Path, result, strategy: str) -> None:
    payload = {
        "case": "public_research_publication",
        "strategy": strategy,
        "seed": result.seed,
        "num_agents": result.config.num_agents,
        "dimensions": result.config.dimensions,
        "rounds": result.config.rounds,
        "num_peaks": result.config.num_peaks,
        "rules": {
            "publish_is_optional": True,
            "private_exchange_allowed": False,
            "published_value": "true_score",
            "blocked_location_rule": "exact match with public registry is not allowed",
            "continuous_space": True,
            "success_condition": "discover the highest-value peaks",
            "top_peak_count": result.config.top_peak_count,
            "top_peak_discovery_fraction": result.config.top_peak_discovery_fraction,
        },
        **result.final_summary(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_landscape(path: Path, result) -> None:
    ranked_peak_ids = list(np.argsort(result.landscape.heights)[::-1].astype(int))
    top_ids = {int(peak_id) for peak_id in top_peak_ids(result.landscape, result.config).tolist()}
    fieldnames = [
        "peak_id",
        "height_rank",
        "is_top_peak_target",
        "height",
        "width",
        *[f"mu{k}" for k in range(result.config.dimensions)],
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for peak_id in range(result.config.num_peaks):
            writer.writerow(
                {
                    "peak_id": peak_id,
                    "height_rank": ranked_peak_ids.index(peak_id) + 1,
                    "is_top_peak_target": peak_id in top_ids,
                    "height": float(result.landscape.heights[peak_id]),
                    "width": float(result.landscape.widths[peak_id]),
                    **{
                        f"mu{k}": float(result.landscape.centers[peak_id, k])
                        for k in range(result.config.dimensions)
                    },
                }
            )


def write_public_registry(path: Path, result) -> None:
    fieldnames = [
        "record_id",
        "round",
        "agent_id",
        "score",
        "position",
        *[f"z{k}" for k in range(result.config.dimensions)],
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record_id, record in enumerate(result.published_records):
            writer.writerow(
                {
                    "record_id": record_id,
                    "round": record.round_index,
                    "agent_id": record.agent_id,
                    "score": float(record.score),
                    "position": json.dumps(_round_vector(record.position)),
                    **{
                        f"z{k}": float(record.position[k])
                        for k in range(result.config.dimensions)
                    },
                }
            )


def write_publication_decisions(path: Path, result) -> None:
    fieldnames = [
        "round",
        "agent_id",
        "score",
        "publish_decision",
        "published",
        "current_location_already_published",
        "public_registry_size_before",
        "blocked_resubmissions",
        "next_position_blocked_initially",
        "current_position",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for round_index, rows in enumerate(result.publication_history):
            current_positions = result.position_history[round_index]
            for row in rows:
                agent_id = int(row["agent_id"])
                writer.writerow(
                    {
                        **{key: row.get(key, "") for key in fieldnames if key != "current_position"},
                        "current_position": json.dumps(_round_vector(current_positions[agent_id])),
                    }
                )


def write_metrics(path: Path, result) -> None:
    fieldnames = sorted({key for row in result.round_metrics for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.round_metrics:
            writer.writerow(row)


def plot_2d_publication(path: Path, result) -> None:
    config = result.config
    grid_size = 140
    x = np.linspace(config.lower, config.upper, grid_size)
    y = np.linspace(config.lower, config.upper, grid_size)
    xx, yy = np.meshgrid(x, y)
    terrain, _, _ = value_positions(np.stack([xx.ravel(), yy.ravel()], axis=1), result.landscape)
    terrain = terrain.reshape(grid_size, grid_size)

    fig, ax = plt.subplots(figsize=(8.0, 7.2), dpi=160)
    image = ax.imshow(
        terrain,
        origin="lower",
        extent=(config.lower, config.upper, config.lower, config.upper),
        cmap="viridis",
        alpha=0.86,
        aspect="auto",
    )
    fig.colorbar(image, ax=ax, label="hidden value")

    history = np.stack(result.position_history, axis=0)
    for agent_id in range(config.num_agents):
        ax.plot(
            history[:, agent_id, 0],
            history[:, agent_id, 1],
            linewidth=1.2,
            alpha=0.68,
        )
        ax.scatter(
            history[-1, agent_id, 0],
            history[-1, agent_id, 1],
            s=42,
            edgecolor="white",
            linewidth=0.8,
        )

    if result.published_records:
        published = np.vstack([record.position for record in result.published_records])
        ax.scatter(
            published[:, 0],
            published[:, 1],
            s=72,
            marker="*",
            color="#facc15",
            edgecolor="#111827",
            linewidth=0.55,
            label="published locations",
        )

    ax.scatter(
        result.landscape.centers[:, 0],
        result.landscape.centers[:, 1],
        marker="X",
        s=90,
        color="#ef4444",
        edgecolor="white",
        linewidth=0.8,
        label="hidden peaks (evaluator only)",
    )
    ax.set_title("Public Research Publication Movement", loc="left", fontweight="bold")
    ax.set_xlabel("dimension 0")
    ax.set_ylabel("dimension 1")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_publication_metrics(path: Path, result) -> None:
    rounds = np.array([row["round"] for row in result.round_metrics], dtype=int)
    best_ratio = np.array([row["best_value_found_ratio"] for row in result.round_metrics], dtype=float)
    top_peak_ratio = np.array(
        [row["top_peak_coverage_ratio"] for row in result.round_metrics],
        dtype=float,
    )
    registry_size = np.array([row["public_registry_size"] for row in result.round_metrics], dtype=float)
    published = np.array([row["published_count_this_round"] for row in result.round_metrics], dtype=float)
    blocked = np.array([row["blocked_resubmission_count"] for row in result.round_metrics], dtype=float)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(9.2, 6.8),
        dpi=160,
        sharex=True,
        gridspec_kw={"hspace": 0.24},
    )
    ax_top.plot(rounds, best_ratio * 100.0, color="#2563eb", marker="o")
    ax_top.plot(rounds, top_peak_ratio * 100.0, color="#16a34a", marker="o")
    ax_top.set_title("Best Value and Top-Peak Coverage", loc="left", fontweight="bold")
    ax_top.set_ylabel("%")
    ax_top.legend(["best value / global optimum", "top-peak coverage"], frameon=False)
    ax_top.grid(True, alpha=0.24)

    ax_bottom.bar(rounds, published, color="#93c5fd", label="published this round")
    ax_bottom.plot(rounds, registry_size, color="#111827", marker="o", label="registry size")
    ax_bottom.plot(rounds, blocked, color="#ef4444", marker="o", label="blocked resubmissions")
    ax_bottom.set_title("Public Registry Growth", loc="left", fontweight="bold")
    ax_bottom.set_xlabel("round")
    ax_bottom.set_ylabel("count")
    ax_bottom.grid(True, alpha=0.24)
    ax_bottom.legend(frameon=False, ncol=3)

    fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.08)
    fig.savefig(path)
    plt.close(fig)


def _round_vector(position: np.ndarray) -> list[float]:
    return [round(float(value), 6) for value in position.tolist()]


if __name__ == "__main__":
    main()

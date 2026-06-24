#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from peak_divergence.core import PeakGameConfig
from peak_divergence.game import (
    best_value_ratio,
    level_to_capacity,
    optimality_gap,
    run_game,
    score_positions,
    score_upper_bound,
    system_optimization_index,
    value_positions,
)
from peak_divergence.strategies import available_strategies, make_population


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize every iteration of one Peak-Divergence game run."
    )
    parser.add_argument("--strategy", choices=available_strategies(), default="strategic_collaboration")
    parser.add_argument("--agents", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--peaks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dimensions", type=int, default=10)
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
    parser.add_argument("--observation-noise", type=float, default=0.0)
    parser.add_argument("--delayed-observation", action="store_true")
    parser.add_argument("--peak-height-min", type=float, default=70.0)
    parser.add_argument("--peak-height-max", type=float, default=120.0)
    parser.add_argument("--peak-width-min", type=float, default=28.0)
    parser.add_argument("--peak-width-max", type=float, default=48.0)
    parser.add_argument("--sequential-agent-updates", action="store_true")
    parser.add_argument("--max-parallel-agent-updates", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/iteration_visualization"))
    parser.add_argument("--grid-size", type=int, default=120)
    parser.add_argument("--trail", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frame_dir = args.out / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    config = PeakGameConfig(
        num_agents=args.agents,
        dimensions=args.dimensions,
        rounds=args.rounds,
        num_peaks=args.peaks,
        beta_diversity=args.beta_diversity,
        gamma_origin=args.gamma_origin,
        peak_height_range=(args.peak_height_min, args.peak_height_max),
        peak_width_range=(args.peak_width_min, args.peak_width_max),
        observation_noise=args.observation_noise,
        delayed_observation=args.delayed_observation,
        parallel_agent_updates=not args.sequential_agent_updates,
        max_parallel_agent_updates=args.max_parallel_agent_updates,
    )
    result = run_game(make_population(args.strategy, args.agents), config=config, seed=args.seed)

    history = np.stack(result.position_history, axis=0)
    if args.dimensions == 2:
        projected_history = history.copy()
        projected_peaks = result.landscape.centers.copy()
        terrain, extent = direct_2d_terrain(result.landscape, config, grid_size=args.grid_size)
        x_label = "z0 coordinate"
        y_label = "z1 coordinate"
        terrain_label = "Hidden value landscape"
    else:
        all_positions = history.reshape(-1, args.dimensions)
        combined = np.vstack([all_positions, result.landscape.centers])
        projected, mean, components = fit_pca_2d(combined)
        projected_history = projected[: len(all_positions)].reshape(
            history.shape[0], args.agents, 2
        )
        projected_peaks = projected[len(all_positions) :]
        terrain, extent = projected_terrain(
            projected_history,
            projected_peaks,
            mean,
            components,
            result.landscape,
            config,
            grid_size=args.grid_size,
        )
        x_label = "2D PCA projection of high-dimensional space"
        y_label = "2D PCA projection of high-dimensional space"
        terrain_label = "Hidden value slice"

    score_history, value_history, diversity_history, origin_history, peak_history = score_history_for(
        history, result.landscape, config
    )
    write_trajectory_csv(
        args.out / "trajectory.csv",
        history,
        projected_history,
        score_history,
        value_history,
        diversity_history,
        origin_history,
        peak_history,
        result.action_history,
        result.communication_history,
        result.observed_count_history,
        config,
    )
    write_metrics_csv(
        args.out / "metrics_by_round.csv",
        score_history,
        value_history,
        diversity_history,
        origin_history,
        history,
        result.landscape,
        config,
        result.action_history,
        result.communication_history,
        result.observed_count_history,
    )
    write_negotiation_csv(
        args.out / "negotiation_exchanges.csv",
        result.negotiation_history,
    )
    write_system_performance_plot(
        args.out / "system_performance.png",
        score_history,
        value_history,
        diversity_history,
        history,
        result.landscape,
        config,
    )

    score_min = float(np.min(score_history))
    score_max = float(np.max(score_history))

    frame_paths = []
    for round_index in range(history.shape[0]):
        path = frame_dir / f"round_{round_index:02d}.png"
        plot_round(
            path=path,
            round_index=round_index,
            strategy=args.strategy,
            projected_history=projected_history,
            projected_peaks=projected_peaks,
            score_history=score_history,
            value_history=value_history,
            diversity_history=diversity_history,
            origin_history=origin_history,
            peak_history=peak_history,
            terrain=terrain,
            extent=extent,
            score_min=score_min,
            score_max=score_max,
            trail=args.trail,
            x_label=x_label,
            y_label=y_label,
            terrain_label=terrain_label,
        )
        frame_paths.append(path)

    print(f"Wrote {len(frame_paths)} frame images to {frame_dir}")
    print(f"Wrote trajectory CSV to {args.out / 'trajectory.csv'}")
    print(f"Wrote negotiation CSV to {args.out / 'negotiation_exchanges.csv'}")
    print(f"Wrote per-round metrics to {args.out / 'metrics_by_round.csv'}")
    print(f"Wrote system performance plot to {args.out / 'system_performance.png'}")
    print(f"First frame: {frame_paths[0]}")
    print(f"Last frame: {frame_paths[-1]}")


def fit_pca_2d(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    projected = centered @ components
    return projected, mean.squeeze(axis=0), components


def direct_2d_terrain(
    landscape,
    config: PeakGameConfig,
    grid_size: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    x = np.linspace(config.lower, config.upper, grid_size)
    y = np.linspace(config.lower, config.upper, grid_size)
    xx, yy = np.meshgrid(x, y)
    positions = np.stack([xx.ravel(), yy.ravel()], axis=1)
    terrain, _, _ = value_positions(positions, landscape)
    return terrain.reshape(grid_size, grid_size), (
        float(config.lower),
        float(config.upper),
        float(config.lower),
        float(config.upper),
    )


def score_history_for(
    history: np.ndarray,
    landscape,
    config: PeakGameConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scores = []
    values = []
    diversities = []
    origins = []
    peak_ids = []
    for positions in history:
        score, value, diversity, origin, peak_id, _ = score_positions(
            positions,
            landscape,
            config,
        )
        scores.append(score)
        values.append(value)
        diversities.append(diversity)
        origins.append(origin)
        peak_ids.append(peak_id)
    return (
        np.vstack(scores),
        np.vstack(values),
        np.vstack(diversities),
        np.vstack(origins),
        np.vstack(peak_ids),
    )


def projected_terrain(
    projected_history: np.ndarray,
    projected_peaks: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    landscape,
    config: PeakGameConfig,
    grid_size: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    projected_points = np.vstack([projected_history.reshape(-1, 2), projected_peaks])
    mins = projected_points.min(axis=0)
    maxs = projected_points.max(axis=0)
    padding = np.maximum((maxs - mins) * 0.12, 8.0)
    mins -= padding
    maxs += padding

    x = np.linspace(mins[0], maxs[0], grid_size)
    y = np.linspace(mins[1], maxs[1], grid_size)
    xx, yy = np.meshgrid(x, y)
    plane_points = mean + np.stack([xx.ravel(), yy.ravel()], axis=1) @ components.T
    plane_points = np.clip(plane_points, config.lower, config.upper)
    terrain, _, _ = value_positions(plane_points, landscape)
    return terrain.reshape(grid_size, grid_size), (float(x.min()), float(x.max()), float(y.min()), float(y.max()))


def write_trajectory_csv(
    path: Path,
    history: np.ndarray,
    projected_history: np.ndarray,
    score_history: np.ndarray,
    value_history: np.ndarray,
    diversity_history: np.ndarray,
    origin_history: np.ndarray,
    peak_history: np.ndarray,
    action_history: list,
    communication_history: list[list[dict]],
    observed_count_history: list[np.ndarray],
    config: PeakGameConfig,
) -> None:
    dimensions = history.shape[2]
    fieldnames = [
        "round",
        "agent_id",
        "x2d",
        "y2d",
        "score",
        "value",
        "diversity",
        "origin",
        "peak_id",
        "visibility",
        "request_count",
        "offer_reciprocal",
        "accept_probability",
        "visibility_capacity",
        "request_capacity",
        "initial_visible_count",
        "requests_sent",
        "requests_received",
        "accepted_requests_sent",
        "accepted_requests_received",
        "rejected_requests_sent",
        "rejected_requests_received",
        "reciprocal_offers_sent",
        "reciprocal_exchanges",
        "observed_count",
        *[f"z{k}" for k in range(dimensions)],
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for round_index in range(history.shape[0]):
            for agent_id in range(history.shape[1]):
                action = action_history[round_index][agent_id] if round_index < len(action_history) else None
                observed_count = (
                    int(observed_count_history[round_index][agent_id])
                    if round_index < len(observed_count_history)
                    else 0
                )
                communication = (
                    communication_history[round_index][agent_id]
                    if round_index < len(communication_history)
                    else {}
                )
                writer.writerow(
                    {
                        "round": round_index,
                        "agent_id": agent_id,
                        "x2d": float(projected_history[round_index, agent_id, 0]),
                        "y2d": float(projected_history[round_index, agent_id, 1]),
                        "score": float(score_history[round_index, agent_id]),
                        "value": float(value_history[round_index, agent_id]),
                        "diversity": float(diversity_history[round_index, agent_id]),
                        "origin": float(origin_history[round_index, agent_id]),
                        "peak_id": int(peak_history[round_index, agent_id]),
                        "visibility": action.visibility if action is not None else "",
                        "request_count": action.request_count if action is not None else "",
                        "offer_reciprocal": action.offer_reciprocal if action is not None else "",
                        "accept_probability": (
                            float(action.accept_probability) if action is not None else ""
                        ),
                        "visibility_capacity": (
                            level_to_capacity(action.visibility, config.num_agents)
                            if action is not None
                            else 0
                        ),
                        "request_capacity": (
                            level_to_capacity(action.request_count, config.num_agents)
                            if action is not None
                            else 0
                        ),
                        "initial_visible_count": communication.get("initial_visible_count", 0),
                        "requests_sent": communication.get("requests_sent", 0),
                        "requests_received": communication.get("requests_received", 0),
                        "accepted_requests_sent": communication.get("accepted_requests_sent", 0),
                        "accepted_requests_received": communication.get(
                            "accepted_requests_received", 0
                        ),
                        "rejected_requests_sent": communication.get("rejected_requests_sent", 0),
                        "rejected_requests_received": communication.get(
                            "rejected_requests_received", 0
                        ),
                        "reciprocal_offers_sent": communication.get(
                            "reciprocal_offers_sent", 0
                        ),
                        "reciprocal_exchanges": communication.get("reciprocal_exchanges", 0),
                        "observed_count": observed_count,
                        **{
                            f"z{k}": float(history[round_index, agent_id, k])
                            for k in range(dimensions)
                        },
                    }
                )


def write_negotiation_csv(path: Path, negotiation_history: list[list]) -> None:
    fieldnames = [
        "round",
        "requester_id",
        "target_id",
        "reciprocal_offer",
        "accepted",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for exchanges in negotiation_history:
            for exchange in exchanges:
                writer.writerow(
                    {
                        "round": exchange.round_index,
                        "requester_id": exchange.requester_id,
                        "target_id": exchange.target_id,
                        "reciprocal_offer": exchange.reciprocal_offer,
                        "accepted": exchange.accepted,
                    }
                )


def write_metrics_csv(
    path: Path,
    score_history: np.ndarray,
    value_history: np.ndarray,
    diversity_history: np.ndarray,
    origin_history: np.ndarray,
    history: np.ndarray,
    landscape,
    config: PeakGameConfig,
    action_history: list,
    communication_history: list[list[dict]],
    observed_count_history: list[np.ndarray],
) -> None:
    with path.open("w", newline="") as handle:
        fieldnames = [
            "round",
            "total_score",
            "mean_score",
            "best_score",
            "score_upper_bound",
            "system_optimization",
            "global_optimum",
            "best_value_ratio",
            "optimality_gap",
            "mean_value",
            "best_value",
            "mean_diversity",
            "mean_origin",
            "peak_coverage",
            "max_peak_occupancy",
            "mean_visibility_capacity",
            "mean_request_capacity",
            "mean_observed_count",
            "request_count",
            "acceptance_rate",
            "rejection_rate",
            "reciprocal_offer_rate",
            "reciprocal_exchange_rate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for round_index in range(history.shape[0]):
            _, peak_ids, contributions = value_positions(history[round_index], landscape)
            thresholds = landscape.heights * config.discovery_value_fraction
            discovered = contributions[np.arange(history.shape[1]), peak_ids] >= thresholds[peak_ids]
            if np.any(discovered):
                _, counts = np.unique(peak_ids[discovered], return_counts=True)
                peak_coverage = float(len(counts))
                max_peak_occupancy = float(counts.max())
            else:
                peak_coverage = 0.0
                max_peak_occupancy = 0.0
            communication = (
                communication_history[round_index]
                if round_index < len(communication_history)
                else []
            )
            total_requests = sum(row.get("requests_sent", 0) for row in communication)
            accepted_requests = sum(
                row.get("accepted_requests_sent", 0) for row in communication
            )
            rejected_requests = sum(
                row.get("rejected_requests_sent", 0) for row in communication
            )
            reciprocal_offers = sum(
                row.get("reciprocal_offers_sent", 0) for row in communication
            )
            reciprocal_exchanges = sum(
                row.get("reciprocal_exchanges", 0) for row in communication
            )
            denominator = total_requests if total_requests else 1
            writer.writerow(
                {
                    "round": round_index,
                    "total_score": float(score_history[round_index].sum()),
                    "mean_score": float(score_history[round_index].mean()),
                    "best_score": float(score_history[round_index].max()),
                    "score_upper_bound": score_upper_bound(landscape, config),
                    "system_optimization": system_optimization_index(
                        score_history[round_index],
                        landscape,
                        config,
                    ),
                    "global_optimum": score_upper_bound(landscape, config),
                    "best_value_ratio": best_value_ratio(
                        value_history[round_index],
                        landscape,
                    ),
                    "optimality_gap": optimality_gap(
                        value_history[round_index],
                        landscape,
                    ),
                    "mean_value": float(value_history[round_index].mean()),
                    "best_value": float(value_history[round_index].max()),
                    "mean_diversity": float(diversity_history[round_index].mean()),
                    "mean_origin": float(origin_history[round_index].mean()),
                    "peak_coverage": peak_coverage,
                    "max_peak_occupancy": max_peak_occupancy,
                    "mean_visibility_capacity": float(
                        np.mean(
                            [
                                level_to_capacity(action.visibility, config.num_agents)
                                for action in action_history[round_index]
                            ]
                        )
                    )
                    if round_index < len(action_history)
                    else 0.0,
                    "mean_request_capacity": float(
                        np.mean(
                            [
                                level_to_capacity(action.request_count, config.num_agents)
                                for action in action_history[round_index]
                            ]
                        )
                    )
                    if round_index < len(action_history)
                    else 0.0,
                    "mean_observed_count": float(observed_count_history[round_index].mean())
                    if round_index < len(observed_count_history)
                    else 0.0,
                    "request_count": float(total_requests),
                    "acceptance_rate": float(accepted_requests / denominator),
                    "rejection_rate": float(rejected_requests / denominator),
                    "reciprocal_offer_rate": float(reciprocal_offers / denominator),
                    "reciprocal_exchange_rate": float(reciprocal_exchanges / denominator),
                }
            )


def write_system_performance_plot(
    path: Path,
    score_history: np.ndarray,
    value_history: np.ndarray,
    diversity_history: np.ndarray,
    history: np.ndarray,
    landscape,
    config: PeakGameConfig,
) -> None:
    rounds = np.arange(history.shape[0])
    upper_bound = score_upper_bound(landscape, config)
    mean_score = score_history.mean(axis=1)
    mean_value = value_history.mean(axis=1)
    best_value = value_history.max(axis=1)
    mean_diversity = diversity_history.mean(axis=1)
    system_optimization = np.array(
        [
            system_optimization_index(score_history[round_index], landscape, config)
            for round_index in range(history.shape[0])
        ]
    )
    best_ratio = np.array(
        [
            best_value_ratio(value_history[round_index], landscape)
            for round_index in range(history.shape[0])
        ]
    )

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(10.5, 7.2),
        dpi=170,
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 1.15], "hspace": 0.16},
    )

    ax_top.plot(
        rounds,
        best_ratio * 100.0,
        color="#2563eb",
        linewidth=2.4,
        marker="o",
        markersize=4,
        label="best value / global optimum",
    )
    ax_top.plot(
        rounds,
        system_optimization * 100.0,
        color="#059669",
        linewidth=2.0,
        marker="o",
        markersize=3,
        label="mean value / global optimum",
    )
    ax_top.fill_between(rounds, best_ratio * 100.0, color="#93c5fd", alpha=0.25)
    ax_top.set_ylabel("Optimality (%)")
    ax_top.set_title(
        "Progress Toward Global Optimum",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    ax_top.grid(True, alpha=0.24)
    ax_top.set_ylim(0.0, max(100.0, float(np.max(best_ratio * 100.0)) * 1.10))
    ax_top.text(
        0.01,
        0.93,
        f"value-only score; global optimum = max peak height = {upper_bound:.2f}",
        transform=ax_top.transAxes,
        fontsize=8,
        color="#4b5563",
        va="top",
    )
    ax_top.legend(frameon=False, loc="lower right")

    ax_bottom.plot(rounds, mean_value, color="#16a34a", linewidth=2.2, label="mean value")
    ax_bottom.plot(rounds, best_value, color="#111827", linewidth=2.0, label="best value found")
    ax_bottom.axhline(upper_bound, color="#6b7280", linestyle="--", linewidth=1.1, label="global optimum")
    ax_diversity = ax_bottom.twinx()
    ax_diversity.plot(
        rounds,
        mean_diversity,
        color="#f97316",
        linewidth=1.7,
        linestyle=":",
        label="mean diversity diagnostic",
    )
    ax_bottom.set_xlabel("Round")
    ax_bottom.set_ylabel("Value")
    ax_diversity.set_ylabel("Mean diversity")
    ax_bottom.set_title(
        "Value Search With Diversity Diagnostic",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    ax_bottom.grid(True, alpha=0.24)
    lines, labels = ax_bottom.get_legend_handles_labels()
    div_lines, div_labels = ax_diversity.get_legend_handles_labels()
    ax_bottom.legend(
        lines + div_lines,
        labels + div_labels,
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
    )

    fig.suptitle(
        "Population-Level Optimization Over Time",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(top=0.90, bottom=0.15)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_round(
    *,
    path: Path,
    round_index: int,
    strategy: str,
    projected_history: np.ndarray,
    projected_peaks: np.ndarray,
    score_history: np.ndarray,
    value_history: np.ndarray,
    diversity_history: np.ndarray,
    origin_history: np.ndarray,
    peak_history: np.ndarray,
    terrain: np.ndarray,
    extent: tuple[float, float, float, float],
    score_min: float,
    score_max: float,
    trail: int,
    x_label: str,
    y_label: str,
    terrain_label: str,
) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "legend.frameon": False,
        }
    )

    positions = projected_history[round_index]
    scores = score_history[round_index]
    values = value_history[round_index]
    diversities = diversity_history[round_index]
    peaks = peak_history[round_index]

    fig, ax = plt.subplots(figsize=(8.8, 7.0))
    image = ax.imshow(
        terrain,
        origin="lower",
        extent=extent,
        cmap="YlGnBu",
        alpha=0.72,
        aspect="auto",
    )
    contours = ax.contour(
        terrain,
        levels=8,
        extent=extent,
        colors="#1f2937",
        linewidths=0.55,
        alpha=0.38,
    )
    ax.clabel(contours, inline=True, fontsize=7, fmt="%.0f")

    start = max(0, round_index - trail)
    for agent_id in range(projected_history.shape[1]):
        trail_points = projected_history[start : round_index + 1, agent_id]
        if len(trail_points) > 1:
            ax.plot(
                trail_points[:, 0],
                trail_points[:, 1],
                color="#374151",
                alpha=0.32,
                linewidth=1.0,
            )

    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=scores,
        s=58,
        cmap="plasma",
        vmin=score_min,
        vmax=max(score_max, score_min + 1e-9),
        edgecolor="white",
        linewidth=0.75,
        alpha=0.92,
        label="Agents",
        zorder=4,
    )
    ax.scatter(
        projected_peaks[:, 0],
        projected_peaks[:, 1],
        marker="*",
        s=190,
        color="black",
        edgecolor="white",
        linewidth=0.8,
        label="Hidden peaks",
        zorder=5,
    )

    for peak_id, point in enumerate(projected_peaks):
        ax.text(
            point[0],
            point[1],
            f" P{peak_id}",
            ha="left",
            va="center",
            fontsize=8,
            color="black",
            zorder=6,
        )

    unique_peaks, peak_counts = np.unique(peaks, return_counts=True)
    occupancy = ", ".join(
        f"P{int(peak)}:{int(count)}" for peak, count in zip(unique_peaks, peak_counts)
    )
    round_label = "initial" if round_index == 0 else f"after update {round_index}"
    ax.set_title(
        f"{strategy} | Round {round_index} ({round_label})\n"
        f"mean score={scores.mean():.2f}, value={values.mean():.2f}, "
        f"diversity={diversities.mean():.2f}"
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.text(
        0.01,
        0.01,
        f"Nearest peak occupancy: {occupancy}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#111827",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 4},
    )
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label=terrain_label)
    fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.08, label="Agent total score")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

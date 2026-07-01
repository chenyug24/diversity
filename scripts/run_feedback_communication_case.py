#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from peak_divergence.core import PeakGameConfig
from peak_divergence.game import level_to_capacity, run_game, score_positions
from peak_divergence.strategies import available_strategies, make_population


LIST_FIELDS = {
    "initial_visible_source_ids",
    "requested_target_ids",
    "accepted_target_ids",
    "rejected_target_ids",
    "incoming_requester_ids",
    "accepted_incoming_requester_ids",
    "rejected_incoming_requester_ids",
    "reciprocal_observed_agent_ids",
    "observed_agent_ids",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the first feedback-only communication case: agents see only "
            "their own score, peer examples revealed by communication, and "
            "communication outcome data."
        )
    )
    parser.add_argument("--strategy", choices=available_strategies(), default="strategic_collaboration")
    parser.add_argument("--agents", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--peaks", type=int, default=2)
    parser.add_argument("--dimensions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--peak-height-min", type=float, default=70.0)
    parser.add_argument("--peak-height-max", type=float, default=120.0)
    parser.add_argument("--peak-width-min", type=float, default=28.0)
    parser.add_argument("--peak-width-max", type=float, default=48.0)
    parser.add_argument("--observation-noise", type=float, default=0.0)
    parser.add_argument("--delayed-observation", action="store_true")
    parser.add_argument("--sequential-agent-updates", action="store_true")
    parser.add_argument("--max-parallel-agent-updates", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/feedback_communication_case"))
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
        observation_noise=args.observation_noise,
        delayed_observation=args.delayed_observation,
        parallel_agent_updates=not args.sequential_agent_updates,
        max_parallel_agent_updates=args.max_parallel_agent_updates,
    )
    result = run_game(make_population(args.strategy, args.agents), config=config, seed=args.seed)

    write_summary(args.out / "summary.json", result, args.strategy)
    write_landscape(args.out / "hidden_landscape_for_evaluator.csv", result)
    write_metrics(args.out / "metrics_by_round.csv", result)
    write_agent_round_feedback(args.out / "agent_round_feedback.csv", result)
    write_observed_examples(args.out / "observed_peer_examples.csv", result)
    write_negotiation_edges(args.out / "negotiation_edges.csv", result)
    plot_communication_network(args.out / "communication_network.png", result)
    plot_score_and_communication(args.out / "score_and_communication.png", result)

    summary = result.final_summary()
    print(f"Wrote feedback-only communication case to {args.out}")
    print(
        "best_found="
        f"{summary['best_value_found']:.3f} "
        f"ratio={summary['best_value_found_ratio']:.3f} "
        f"gap={summary['best_value_found_gap']:.3f}"
    )
    print(f"communication table: {args.out / 'agent_round_feedback.csv'}")
    print(f"observed examples: {args.out / 'observed_peer_examples.csv'}")
    print(f"network plot: {args.out / 'communication_network.png'}")


def write_summary(path: Path, result, strategy: str) -> None:
    summary = result.final_summary()
    payload = {
        "case": "feedback_only_communication",
        "strategy": strategy,
        "seed": result.seed,
        "num_agents": result.config.num_agents,
        "dimensions": result.config.dimensions,
        "rounds": result.config.rounds,
        "num_peaks": result.config.num_peaks,
        "agent_information": [
            "own current position",
            "own current score",
            "own position/score history",
            "peer position/score examples revealed through communication",
            "communication outcome counts and peer ids",
        ],
        "not_revealed_to_agents": [
            "score formula",
            "hidden peak centers",
            "hidden peak heights",
            "hidden peak widths",
            "global optimum",
            "other agents' unshared data",
        ],
        **summary,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_landscape(path: Path, result) -> None:
    fieldnames = ["peak_id", "height", "width", *[f"mu{k}" for k in range(result.config.dimensions)]]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for peak_id in range(result.config.num_peaks):
            writer.writerow(
                {
                    "peak_id": peak_id,
                    "height": float(result.landscape.heights[peak_id]),
                    "width": float(result.landscape.widths[peak_id]),
                    **{
                        f"mu{k}": float(result.landscape.centers[peak_id, k])
                        for k in range(result.config.dimensions)
                    },
                }
            )


def write_metrics(path: Path, result) -> None:
    fieldnames = sorted({key for row in result.round_metrics for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.round_metrics:
            writer.writerow(row)


def write_agent_round_feedback(path: Path, result) -> None:
    fieldnames = [
        "round",
        "agent_id",
        "score",
        "position",
        "visibility",
        "request_count",
        "offer_reciprocal",
        "accept_probability",
        "visibility_capacity",
        "request_capacity",
        "initial_visible_count",
        "initially_visible_to_count",
        "requests_sent",
        "requests_received",
        "accepted_requests_sent",
        "accepted_requests_received",
        "rejected_requests_sent",
        "rejected_requests_received",
        "reciprocal_offers_sent",
        "reciprocal_exchanges",
        "observed_count",
        "initial_visible_source_ids",
        "requested_target_ids",
        "accepted_target_ids",
        "rejected_target_ids",
        "incoming_requester_ids",
        "accepted_incoming_requester_ids",
        "rejected_incoming_requester_ids",
        "reciprocal_observed_agent_ids",
        "observed_agent_ids",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for round_index, positions in enumerate(result.position_history[:-1]):
            scores, _, _, _, _, _ = score_positions(positions, result.landscape, result.config)
            for agent_id in range(result.config.num_agents):
                action = result.action_history[round_index][agent_id]
                feedback = result.communication_history[round_index][agent_id]
                writer.writerow(
                    {
                        "round": round_index,
                        "agent_id": agent_id,
                        "score": float(scores[agent_id]),
                        "position": json.dumps(_round_vector(positions[agent_id])),
                        "visibility": action.visibility,
                        "request_count": action.request_count,
                        "offer_reciprocal": action.offer_reciprocal,
                        "accept_probability": float(action.accept_probability),
                        "visibility_capacity": level_to_capacity(
                            action.visibility,
                            result.config.num_agents,
                        ),
                        "request_capacity": level_to_capacity(
                            action.request_count,
                            result.config.num_agents,
                        ),
                        **{field: _csv_value(feedback.get(field, 0)) for field in fieldnames if field not in {
                            "round",
                            "agent_id",
                            "score",
                            "position",
                            "visibility",
                            "request_count",
                            "offer_reciprocal",
                            "accept_probability",
                            "visibility_capacity",
                            "request_capacity",
                        }},
                    }
                )


def write_observed_examples(path: Path, result) -> None:
    fieldnames = [
        "round",
        "agent_id",
        "observed_agent_id",
        "observed_score",
        "observed_position",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for round_index, positions in enumerate(result.position_history[:-1]):
            scores, _, _, _, _, _ = score_positions(positions, result.landscape, result.config)
            for agent_id, feedback in enumerate(result.communication_history[round_index]):
                for observed_agent_id in feedback.get("observed_agent_ids", []):
                    writer.writerow(
                        {
                            "round": round_index,
                            "agent_id": agent_id,
                            "observed_agent_id": int(observed_agent_id),
                            "observed_score": float(scores[int(observed_agent_id)]),
                            "observed_position": json.dumps(
                                _round_vector(positions[int(observed_agent_id)])
                            ),
                        }
                    )


def write_negotiation_edges(path: Path, result) -> None:
    fieldnames = ["round", "requester_id", "target_id", "reciprocal_offer", "accepted"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for exchanges in result.negotiation_history:
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


def plot_communication_network(path: Path, result) -> None:
    accepted = Counter()
    rejected = Counter()
    for exchanges in result.negotiation_history:
        for exchange in exchanges:
            key = (exchange.requester_id, exchange.target_id)
            if exchange.accepted:
                accepted[key] += 1
            else:
                rejected[key] += 1

    num_agents = result.config.num_agents
    angles = np.linspace(0, 2 * np.pi, num_agents, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])

    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=160)
    ax.set_title("Negotiated Communication Network", fontsize=13, fontweight="bold")
    for (source, target), count in rejected.items():
        draw_edge(ax, points[source], points[target], "#ef4444", count, alpha=0.18)
    for (source, target), count in accepted.items():
        draw_edge(ax, points[source], points[target], "#16a34a", count, alpha=0.35)

    ax.scatter(points[:, 0], points[:, 1], s=420, color="#dbeafe", edgecolor="#1d4ed8", zorder=3)
    for agent_id, (x, y) in enumerate(points):
        ax.text(x, y, str(agent_id), ha="center", va="center", fontsize=10, fontweight="bold", zorder=4)

    total_accepted = sum(accepted.values())
    total_rejected = sum(rejected.values())
    ax.text(
        0.02,
        0.02,
        f"green=accepted requests ({total_accepted}), red=rejected requests ({total_rejected})",
        transform=ax.transAxes,
        fontsize=9,
        color="#374151",
    )
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.04)
    fig.savefig(path)
    plt.close(fig)


def plot_score_and_communication(path: Path, result) -> None:
    rounds = np.array([row["round"] for row in result.round_metrics], dtype=int)
    best_found_ratio = np.array(
        [row["best_value_found_ratio"] for row in result.round_metrics],
        dtype=float,
    )
    mean_score = np.array([row["mean_score"] for row in result.round_metrics], dtype=float)
    request_count = np.array(
        [row.get("request_count", 0.0) for row in result.round_metrics],
        dtype=float,
    )
    acceptance_rate = np.array(
        [row.get("acceptance_rate", 0.0) for row in result.round_metrics],
        dtype=float,
    )
    mean_observed = np.array(
        [row.get("mean_observed_count", 0.0) for row in result.round_metrics],
        dtype=float,
    )

    fig, (ax_score, ax_comm) = plt.subplots(
        2,
        1,
        figsize=(9.5, 7.0),
        dpi=160,
        sharex=True,
        gridspec_kw={"hspace": 0.18},
    )
    ax_score.plot(rounds, best_found_ratio * 100, marker="o", color="#2563eb", label="best found / optimum")
    ax_score.plot(rounds, mean_score, marker="o", color="#059669", label="mean score")
    ax_score.set_title("Feedback-Only Search Progress", loc="left", fontsize=13, fontweight="bold")
    ax_score.set_ylabel("Value / %")
    ax_score.grid(True, alpha=0.25)
    ax_score.legend(frameon=False)

    ax_comm.bar(rounds, request_count, color="#93c5fd", label="requests")
    ax_comm.plot(rounds, acceptance_rate, color="#16a34a", marker="o", label="acceptance rate")
    ax_comm.plot(rounds, mean_observed, color="#f97316", marker="o", label="mean observed peers")
    ax_comm.set_title("Communication Outcomes", loc="left", fontsize=13, fontweight="bold")
    ax_comm.set_xlabel("Round")
    ax_comm.set_ylabel("Count / rate")
    ax_comm.grid(True, alpha=0.25)
    ax_comm.legend(frameon=False, ncol=3)

    fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.08, hspace=0.26)
    fig.savefig(path)
    plt.close(fig)


def draw_edge(ax, start: np.ndarray, end: np.ndarray, color: str, count: int, alpha: float) -> None:
    direction = end - start
    start_point = start + direction * 0.14
    end_point = start + direction * 0.86
    ax.annotate(
        "",
        xy=end_point,
        xytext=start_point,
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "lw": 0.5 + 0.45 * min(count, 8),
            "alpha": alpha,
            "shrinkA": 2,
            "shrinkB": 2,
        },
        zorder=1,
    )


def _round_vector(position: np.ndarray) -> list[float]:
    return [round(float(value), 3) for value in position.tolist()]


def _csv_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        return json.dumps(value)
    return value


if __name__ == "__main__":
    main()

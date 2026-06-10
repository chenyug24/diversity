from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .core import CollaborationAction, GameConfig, Level, Observation, RunResult


def level_to_capacity(level: Level, num_agents: int) -> int:
    if level == "all":
        return max(0, num_agents - 1)
    value = int(level)
    if value < 0:
        raise ValueError(f"Collaboration level must be non-negative, got {level!r}")
    return min(value, max(0, num_agents - 1))


def normalized_l1_matrix(positions: np.ndarray) -> np.ndarray:
    """Return the all-pairs normalized L1 distance matrix."""

    if positions.ndim != 2:
        raise ValueError("positions must have shape (num_agents, dimensions)")
    return np.abs(positions[:, None, :] - positions[None, :, :]).mean(axis=2)


def score_positions(
    positions: np.ndarray, lambda_origin: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute final score, diversity term, and origin term for each agent."""

    num_agents = positions.shape[0]
    if num_agents < 2:
        raise ValueError("At least two agents are required to compute diversity")

    distances = normalized_l1_matrix(positions)
    diversity = distances.sum(axis=1) / (num_agents - 1)
    origin = positions.mean(axis=1)
    scores = diversity + lambda_origin * origin
    return scores, diversity, origin


def summarize_positions(
    positions: np.ndarray,
    lambda_origin: float,
    share_levels: Sequence[Level] | None = None,
    read_levels: Sequence[Level] | None = None,
) -> dict[str, float]:
    """Diagnostic metrics for one population state."""

    scores, diversity, origin = score_positions(positions, lambda_origin)
    distances = normalized_l1_matrix(positions)
    num_agents = positions.shape[0]
    upper_triangle = distances[np.triu_indices(num_agents, k=1)]

    corner_bits = positions >= 50.0
    corner_ids = np.packbits(corner_bits.astype(np.uint8), axis=1, bitorder="little")
    corner_keys = [tuple(row.tolist()) for row in corner_ids]
    _, counts = np.unique(corner_keys, return_counts=True, axis=0)
    probabilities = counts / counts.sum()
    entropy = -float(np.sum(probabilities * np.log2(probabilities + 1e-12)))

    centroid = positions.mean(axis=0)
    distance_to_centroid = np.abs(positions - centroid).mean(axis=1)

    metrics = {
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std()),
        "best_score": float(scores.max()),
        "worst_score": float(scores.min()),
        "mean_diversity": float(diversity.mean()),
        "mean_origin": float(origin.mean()),
        "mean_pairwise_distance": float(upper_triangle.mean()),
        "min_pairwise_distance": float(upper_triangle.min()),
        "mean_distance_to_centroid": float(distance_to_centroid.mean()),
        "corner_coverage": float(len(counts)),
        "max_corner_occupancy": float(counts.max()),
        "corner_entropy": entropy,
    }

    if share_levels is not None:
        metrics["mean_share_capacity"] = float(
            np.mean([level_to_capacity(level, num_agents) for level in share_levels])
        )
    if read_levels is not None:
        metrics["mean_read_capacity"] = float(
            np.mean([level_to_capacity(level, num_agents) for level in read_levels])
        )

    return metrics


def _visible_sources_by_reader(
    share_levels: Sequence[Level], rng: np.random.Generator
) -> list[list[int]]:
    num_agents = len(share_levels)
    visible_sources: list[list[int]] = [[] for _ in range(num_agents)]
    agent_ids = np.arange(num_agents)

    for source_id, level in enumerate(share_levels):
        capacity = level_to_capacity(level, num_agents)
        if capacity == 0:
            continue
        recipients = agent_ids[agent_ids != source_id]
        if capacity < len(recipients):
            recipients = rng.choice(recipients, size=capacity, replace=False)
        for recipient_id in recipients:
            visible_sources[int(recipient_id)].append(source_id)

    return visible_sources


def _allocate_observations(
    positions: np.ndarray,
    share_levels: Sequence[Level],
    read_levels: Sequence[Level],
    rng: np.random.Generator,
    config: GameConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Apply share/read budgets and return observed ids and positions per agent."""

    num_agents = positions.shape[0]
    visible_sources = _visible_sources_by_reader(share_levels, rng)
    observations: list[tuple[np.ndarray, np.ndarray]] = []

    for reader_id, visible in enumerate(visible_sources):
        read_capacity = level_to_capacity(read_levels[reader_id], num_agents)
        if read_capacity == 0 or not visible:
            observed_ids = np.array([], dtype=int)
        elif read_capacity < len(visible):
            observed_ids = np.array(
                rng.choice(np.array(visible, dtype=int), size=read_capacity, replace=False),
                dtype=int,
            )
        else:
            observed_ids = np.array(visible, dtype=int)

        observed_positions = positions[observed_ids].copy()
        if config.observation_noise > 0.0 and len(observed_ids) > 0:
            observed_positions += rng.normal(
                loc=0.0,
                scale=config.observation_noise,
                size=observed_positions.shape,
            )
            observed_positions = np.clip(observed_positions, config.lower, config.upper)

        observations.append((observed_ids, observed_positions))

    return observations


def run_game(
    strategies: Sequence[Any],
    config: GameConfig,
    seed: int = 0,
) -> RunResult:
    """Run one simultaneous-move Hypercube Divergence Game."""

    if len(strategies) != config.num_agents:
        raise ValueError(
            f"Expected {config.num_agents} strategies, got {len(strategies)}"
        )

    rng = np.random.default_rng(seed)
    agent_rngs = [
        np.random.default_rng(int(agent_seed))
        for agent_seed in rng.integers(0, np.iinfo(np.int32).max, size=config.num_agents)
    ]

    positions = np.vstack(
        [
            strategy.initial_position(agent_id, agent_rngs[agent_id], config)
            for agent_id, strategy in enumerate(strategies)
        ]
    )
    positions = np.clip(positions.astype(float), config.lower, config.upper)

    round_metrics: list[dict[str, Any]] = []
    previous_metrics: dict[str, float] | None = None
    previous_positions = positions.copy()

    for round_index in range(config.rounds):
        actions: list[CollaborationAction] = [
            strategy.choose_collaboration(
                round_index=round_index,
                position=positions[agent_id].copy(),
                config=config,
                rng=agent_rngs[agent_id],
                previous_metrics=previous_metrics,
            )
            for agent_id, strategy in enumerate(strategies)
        ]
        share_levels = [action.share for action in actions]
        read_levels = [action.read for action in actions]

        observable_positions = previous_positions if config.delayed_observation else positions
        allocated = _allocate_observations(
            observable_positions, share_levels, read_levels, rng, config
        )

        next_positions = []
        for agent_id, strategy in enumerate(strategies):
            observed_ids, observed_positions = allocated[agent_id]
            observation = Observation(
                round_index=round_index,
                agent_id=agent_id,
                own_position=positions[agent_id].copy(),
                observed_ids=observed_ids,
                observed_positions=observed_positions,
                previous_metrics=previous_metrics,
            )
            next_position = strategy.update_position(
                observation=observation,
                rng=agent_rngs[agent_id],
                config=config,
            )
            next_positions.append(next_position)

        previous_positions = positions
        positions = np.clip(np.vstack(next_positions).astype(float), config.lower, config.upper)
        metrics = summarize_positions(
            positions,
            config.lambda_origin,
            share_levels=share_levels,
            read_levels=read_levels,
        )
        metrics["round"] = round_index
        round_metrics.append(metrics)
        previous_metrics = metrics

    final_scores, final_diversity, final_origin = score_positions(
        positions, config.lambda_origin
    )
    agent_records = [
        {
            "agent_id": agent_id,
            "strategy": getattr(strategies[agent_id], "name", type(strategies[agent_id]).__name__),
            "score": float(final_scores[agent_id]),
            "diversity": float(final_diversity[agent_id]),
            "origin": float(final_origin[agent_id]),
            **{
                f"z{k}": float(positions[agent_id, k])
                for k in range(config.dimensions)
            },
        }
        for agent_id in range(config.num_agents)
    ]

    return RunResult(
        config=config,
        seed=seed,
        positions=positions,
        final_scores=final_scores,
        final_diversity=final_diversity,
        final_origin=final_origin,
        round_metrics=round_metrics,
        agent_records=agent_records,
    )

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .core import (
    CollaborationAction,
    Level,
    PeakGameConfig,
    PeakLandscape,
    PeakObservation,
    PeakRunResult,
)


def level_to_capacity(level: Level, num_agents: int) -> int:
    if level == "all":
        return max(0, num_agents - 1)
    value = int(level)
    if value < 0:
        raise ValueError(f"Collaboration level must be non-negative, got {level!r}")
    return min(value, max(0, num_agents - 1))


def generate_landscape(config: PeakGameConfig, rng: np.random.Generator) -> PeakLandscape:
    """Generate hidden Gaussian peaks with a light separation constraint."""

    centers: list[np.ndarray] = []
    attempts = 0
    max_attempts = max(100, config.num_peaks * 100)
    while len(centers) < config.num_peaks and attempts < max_attempts:
        attempts += 1
        candidate = rng.uniform(config.lower, config.upper, size=config.dimensions)
        if not centers:
            centers.append(candidate)
            continue
        distances = np.linalg.norm(np.vstack(centers) - candidate, axis=1)
        if float(distances.min()) >= config.min_peak_l2_distance:
            centers.append(candidate)

    while len(centers) < config.num_peaks:
        centers.append(rng.uniform(config.lower, config.upper, size=config.dimensions))

    heights = rng.uniform(
        config.peak_height_range[0],
        config.peak_height_range[1],
        size=config.num_peaks,
    )
    widths = rng.uniform(
        config.peak_width_range[0],
        config.peak_width_range[1],
        size=config.num_peaks,
    )
    return PeakLandscape(
        centers=np.vstack(centers).astype(float),
        heights=heights.astype(float),
        widths=widths.astype(float),
    )


def normalized_l1_matrix(positions: np.ndarray) -> np.ndarray:
    if positions.ndim != 2:
        raise ValueError("positions must have shape (num_agents, dimensions)")
    return np.abs(positions[:, None, :] - positions[None, :, :]).mean(axis=2)


def value_positions(
    positions: np.ndarray,
    landscape: PeakLandscape,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return max peak value, nearest-by-value peak id, and all peak contributions."""

    deltas = positions[:, None, :] - landscape.centers[None, :, :]
    squared_l2 = np.sum(deltas * deltas, axis=2)
    widths_squared = landscape.widths[None, :] ** 2
    contributions = landscape.heights[None, :] * np.exp(-squared_l2 / (2.0 * widths_squared))
    peak_ids = np.argmax(contributions, axis=1)
    values = contributions[np.arange(len(positions)), peak_ids]
    return values, peak_ids.astype(int), contributions


def score_positions(
    positions: np.ndarray,
    landscape: PeakLandscape,
    config: PeakGameConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_agents = positions.shape[0]
    if num_agents < 2:
        raise ValueError("At least two agents are required to compute diversity")

    distances = normalized_l1_matrix(positions)
    diversity = distances.sum(axis=1) / (num_agents - 1)
    origin = positions.mean(axis=1)
    values, peak_ids, contributions = value_positions(positions, landscape)
    scores = values * (1.0 + config.beta_diversity * diversity + config.gamma_origin * origin)
    return scores, values, diversity, origin, peak_ids, contributions


def summarize_positions(
    positions: np.ndarray,
    landscape: PeakLandscape,
    config: PeakGameConfig,
    share_levels: Sequence[Level] | None = None,
    read_levels: Sequence[Level] | None = None,
) -> dict[str, float]:
    scores, values, diversity, origin, peak_ids, contributions = score_positions(
        positions, landscape, config
    )
    distances = normalized_l1_matrix(positions)
    num_agents = positions.shape[0]
    upper_triangle = distances[np.triu_indices(num_agents, k=1)]

    peak_thresholds = landscape.heights * config.discovery_value_fraction
    discovered = contributions[np.arange(num_agents), peak_ids] >= peak_thresholds[peak_ids]
    discovered_peak_ids = peak_ids[discovered]

    if len(discovered_peak_ids) > 0:
        _, counts = np.unique(discovered_peak_ids, return_counts=True)
        probabilities = counts / counts.sum()
        peak_entropy = -float(np.sum(probabilities * np.log2(probabilities + 1e-12)))
        peak_coverage = float(len(counts))
        max_peak_occupancy = float(counts.max())
    else:
        peak_entropy = 0.0
        peak_coverage = 0.0
        max_peak_occupancy = 0.0

    centroid = positions.mean(axis=0)
    distance_to_centroid = np.abs(positions - centroid).mean(axis=1)

    metrics = {
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std()),
        "best_score": float(scores.max()),
        "worst_score": float(scores.min()),
        "mean_value": float(values.mean()),
        "best_value": float(values.max()),
        "mean_diversity": float(diversity.mean()),
        "mean_origin": float(origin.mean()),
        "mean_pairwise_distance": float(upper_triangle.mean()),
        "min_pairwise_distance": float(upper_triangle.min()),
        "mean_distance_to_centroid": float(distance_to_centroid.mean()),
        "peak_coverage": peak_coverage,
        "max_peak_occupancy": max_peak_occupancy,
        "peak_entropy": peak_entropy,
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
    share_levels: Sequence[Level],
    rng: np.random.Generator,
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
    scores: np.ndarray,
    values: np.ndarray,
    share_levels: Sequence[Level],
    read_levels: Sequence[Level],
    rng: np.random.Generator,
    config: PeakGameConfig,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    num_agents = positions.shape[0]
    visible_sources = _visible_sources_by_reader(share_levels, rng)
    observations: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

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

        observations.append(
            (
                observed_ids,
                observed_positions,
                scores[observed_ids].copy(),
                values[observed_ids].copy(),
            )
        )

    return observations


def run_game(
    strategies: Sequence[Any],
    config: PeakGameConfig,
    seed: int = 0,
    landscape: PeakLandscape | None = None,
) -> PeakRunResult:
    if len(strategies) != config.num_agents:
        raise ValueError(
            f"Expected {config.num_agents} strategies, got {len(strategies)}"
        )

    rng = np.random.default_rng(seed)
    if landscape is None:
        landscape = generate_landscape(config, rng)

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
    previous_state: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    for round_index in range(config.rounds):
        scores, values, diversity, origin, _, _ = score_positions(positions, landscape, config)
        actions: list[CollaborationAction] = [
            strategy.choose_collaboration(
                round_index=round_index,
                position=positions[agent_id].copy(),
                score=float(scores[agent_id]),
                value=float(values[agent_id]),
                diversity=float(diversity[agent_id]),
                origin=float(origin[agent_id]),
                config=config,
                rng=agent_rngs[agent_id],
                previous_metrics=previous_metrics,
            )
            for agent_id, strategy in enumerate(strategies)
        ]
        share_levels = [action.share for action in actions]
        read_levels = [action.read for action in actions]

        if config.delayed_observation and previous_state is not None:
            observable_positions, observable_scores, observable_values = previous_state
        else:
            observable_positions = positions
            observable_scores = scores
            observable_values = values

        allocated = _allocate_observations(
            observable_positions,
            observable_scores,
            observable_values,
            share_levels,
            read_levels,
            rng,
            config,
        )

        next_positions = []
        for agent_id, strategy in enumerate(strategies):
            observed_ids, observed_positions, observed_scores, observed_values = allocated[agent_id]
            observation = PeakObservation(
                round_index=round_index,
                agent_id=agent_id,
                own_position=positions[agent_id].copy(),
                own_score=float(scores[agent_id]),
                own_value=float(values[agent_id]),
                own_diversity=float(diversity[agent_id]),
                own_origin=float(origin[agent_id]),
                observed_ids=observed_ids,
                observed_positions=observed_positions,
                observed_scores=observed_scores,
                observed_values=observed_values,
                previous_metrics=previous_metrics,
            )
            next_position = strategy.update_position(
                observation=observation,
                rng=agent_rngs[agent_id],
                config=config,
            )
            next_positions.append(next_position)

        previous_state = (positions.copy(), scores.copy(), values.copy())
        positions = np.clip(np.vstack(next_positions).astype(float), config.lower, config.upper)
        metrics = summarize_positions(
            positions,
            landscape,
            config,
            share_levels=share_levels,
            read_levels=read_levels,
        )
        metrics["round"] = round_index
        round_metrics.append(metrics)
        previous_metrics = metrics

    final_scores, final_values, final_diversity, final_origin, final_peak_ids, _ = score_positions(
        positions, landscape, config
    )
    agent_records = [
        {
            "agent_id": agent_id,
            "strategy": getattr(strategies[agent_id], "name", type(strategies[agent_id]).__name__),
            "score": float(final_scores[agent_id]),
            "value": float(final_values[agent_id]),
            "diversity": float(final_diversity[agent_id]),
            "origin": float(final_origin[agent_id]),
            "peak_id": int(final_peak_ids[agent_id]),
            **{
                f"z{k}": float(positions[agent_id, k])
                for k in range(config.dimensions)
            },
        }
        for agent_id in range(config.num_agents)
    ]

    return PeakRunResult(
        config=config,
        landscape=landscape,
        seed=seed,
        positions=positions,
        final_scores=final_scores,
        final_values=final_values,
        final_diversity=final_diversity,
        final_origin=final_origin,
        final_peak_ids=final_peak_ids,
        round_metrics=round_metrics,
        agent_records=agent_records,
    )

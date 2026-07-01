from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import numpy as np

from .core import (
    CollaborationAction,
    Level,
    NegotiationExchange,
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
    scores = values
    return scores, values, diversity, origin, peak_ids, contributions


def score_upper_bound(landscape: PeakLandscape, config: PeakGameConfig) -> float:
    """Individual-score upper bound for the value-only landscape."""

    return float(np.max(landscape.heights))


def system_optimization_index(
    scores: np.ndarray,
    landscape: PeakLandscape,
    config: PeakGameConfig,
) -> float:
    """Return mean population score as a fraction of the global optimum."""

    upper_bound = score_upper_bound(landscape, config)
    if upper_bound <= 0.0:
        return 0.0
    return float(np.mean(scores) / upper_bound)


def best_value_ratio(values: np.ndarray, landscape: PeakLandscape) -> float:
    global_optimum = float(np.max(landscape.heights))
    if global_optimum <= 0.0:
        return 0.0
    return float(np.max(values) / global_optimum)


def optimality_gap(values: np.ndarray, landscape: PeakLandscape) -> float:
    return max(0.0, float(np.max(landscape.heights) - np.max(values)))


def value_ratio(value: float, landscape: PeakLandscape) -> float:
    global_optimum = float(np.max(landscape.heights))
    if global_optimum <= 0.0:
        return 0.0
    return float(value / global_optimum)


def top_peak_ids(landscape: PeakLandscape, config: PeakGameConfig) -> np.ndarray:
    """Return the ids of the highest-value peaks used for the success condition."""

    count = max(0, min(int(config.top_peak_count), len(landscape.heights)))
    if count == 0:
        return np.empty(0, dtype=int)
    return np.argsort(landscape.heights)[::-1][:count].astype(int)


def discovered_peak_ids(
    positions: np.ndarray,
    landscape: PeakLandscape,
    config: PeakGameConfig,
) -> np.ndarray:
    """Return peak ids discovered by any point under the top-peak threshold rule."""

    if positions.size == 0:
        return np.empty(0, dtype=int)
    _, _, contributions = value_positions(positions, landscape)
    thresholds = landscape.heights[None, :] * float(config.top_peak_discovery_fraction)
    return np.flatnonzero(np.any(contributions >= thresholds, axis=0)).astype(int)


def top_peak_discovery_metrics(
    discovered_ids: set[int] | np.ndarray,
    landscape: PeakLandscape,
    config: PeakGameConfig,
    *,
    prefix: str = "top_peak",
) -> dict[str, float | str]:
    """Summarize how many of the highest peaks have been discovered."""

    target_ids = top_peak_ids(landscape, config)
    if isinstance(discovered_ids, np.ndarray):
        discovered_set = {int(peak_id) for peak_id in discovered_ids.tolist()}
    else:
        discovered_set = {int(peak_id) for peak_id in discovered_ids}
    target_set = {int(peak_id) for peak_id in target_ids.tolist()}
    discovered_top_ids = sorted(target_set & discovered_set)
    target_count = len(target_ids)
    coverage_count = len(discovered_top_ids)
    coverage_ratio = coverage_count / target_count if target_count else 0.0
    return {
        f"{prefix}_count": float(target_count),
        f"{prefix}_ids": ",".join(str(int(peak_id)) for peak_id in target_ids),
        f"{prefix}_discovered_ids": ",".join(str(peak_id) for peak_id in discovered_top_ids),
        f"{prefix}_coverage_count": float(coverage_count),
        f"{prefix}_coverage_ratio": float(coverage_ratio),
        f"{prefix}_success": float(target_count > 0 and coverage_count == target_count),
        f"{prefix}_discovery_fraction": float(config.top_peak_discovery_fraction),
    }


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def summarize_positions(
    positions: np.ndarray,
    landscape: PeakLandscape,
    config: PeakGameConfig,
    visibility_levels: Sequence[Level] | None = None,
    request_levels: Sequence[Level] | None = None,
) -> dict[str, float]:
    scores, values, diversity, origin, peak_ids, contributions = score_positions(
        positions, landscape, config
    )
    distances = normalized_l1_matrix(positions)
    num_agents = positions.shape[0]
    upper_triangle = distances[np.triu_indices(num_agents, k=1)]

    peak_thresholds = landscape.heights * config.discovery_value_fraction
    discovered = contributions[np.arange(num_agents), peak_ids] >= peak_thresholds[peak_ids]
    locally_discovered_peak_ids = peak_ids[discovered]

    if len(locally_discovered_peak_ids) > 0:
        _, counts = np.unique(locally_discovered_peak_ids, return_counts=True)
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
        "total_score": float(scores.sum()),
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std()),
        "best_score": float(scores.max()),
        "worst_score": float(scores.min()),
        "score_upper_bound": score_upper_bound(landscape, config),
        "system_optimization": system_optimization_index(scores, landscape, config),
        "global_optimum": score_upper_bound(landscape, config),
        "best_value_ratio": best_value_ratio(values, landscape),
        "optimality_gap": optimality_gap(values, landscape),
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
        **top_peak_discovery_metrics(
            discovered_peak_ids(positions, landscape, config),
            landscape,
            config,
            prefix="current_top_peak",
        ),
    }

    if visibility_levels is not None:
        metrics["mean_visibility_capacity"] = float(
            np.mean([level_to_capacity(level, num_agents) for level in visibility_levels])
        )
    if request_levels is not None:
        metrics["mean_request_capacity"] = float(
            np.mean([level_to_capacity(level, num_agents) for level in request_levels])
        )

    return metrics


def _initial_visible_sources_by_reader(
    visibility_levels: Sequence[Level],
    rng: np.random.Generator,
) -> list[list[int]]:
    num_agents = len(visibility_levels)
    visible_sources: list[list[int]] = [[] for _ in range(num_agents)]
    agent_ids = np.arange(num_agents)

    for source_id, level in enumerate(visibility_levels):
        capacity = level_to_capacity(level, num_agents)
        if capacity == 0:
            continue
        recipients = agent_ids[agent_ids != source_id]
        if capacity < len(recipients):
            recipients = rng.choice(recipients, size=capacity, replace=False)
        for recipient_id in recipients:
            visible_sources[int(recipient_id)].append(source_id)

    return visible_sources


def _run_negotiation(
    positions: np.ndarray,
    scores: np.ndarray,
    actions: Sequence[CollaborationAction],
    rng: np.random.Generator,
    config: PeakGameConfig,
    round_index: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], list[NegotiationExchange], list[dict[str, Any]]]:
    num_agents = positions.shape[0]
    initial_visible_sources = _initial_visible_sources_by_reader(
        [action.visibility for action in actions], rng
    )
    initially_visible_to_counts = [0 for _ in range(num_agents)]
    for visible_sources in initial_visible_sources:
        for source_id in visible_sources:
            initially_visible_to_counts[source_id] += 1
    observed_source_sets = [set(sources) for sources in initial_visible_sources]
    exchanges: list[NegotiationExchange] = []
    stats: list[dict[str, Any]] = [
        {
            "initial_visible_count": len(initial_visible_sources[agent_id]),
            "initially_visible_to_count": initially_visible_to_counts[agent_id],
            "requests_sent": 0,
            "requests_received": 0,
            "accepted_requests_sent": 0,
            "accepted_requests_received": 0,
            "rejected_requests_sent": 0,
            "rejected_requests_received": 0,
            "reciprocal_offers_sent": 0,
            "reciprocal_exchanges": 0,
            "initial_visible_source_ids": sorted(
                int(source_id) for source_id in initial_visible_sources[agent_id]
            ),
            "requested_target_ids": [],
            "accepted_target_ids": [],
            "rejected_target_ids": [],
            "incoming_requester_ids": [],
            "accepted_incoming_requester_ids": [],
            "rejected_incoming_requester_ids": [],
            "reciprocal_observed_agent_ids": [],
        }
        for agent_id in range(num_agents)
    ]

    agent_ids = np.arange(num_agents)
    for requester_id, action in enumerate(actions):
        request_capacity = level_to_capacity(action.request_count, num_agents)
        if request_capacity == 0:
            continue

        candidates = np.array(
            [
                agent_id
                for agent_id in agent_ids
                if agent_id != requester_id and agent_id not in observed_source_sets[requester_id]
            ],
            dtype=int,
        )
        if len(candidates) == 0:
            continue

        count = min(request_capacity, len(candidates))
        targets = rng.choice(candidates, size=count, replace=False)
        for target_id_raw in targets:
            target_id = int(target_id_raw)
            reciprocal_offer = bool(action.offer_reciprocal)
            accept_probability = float(actions[target_id].accept_probability)
            accept_probability = min(1.0, max(0.0, accept_probability))
            accepted = bool(rng.random() < accept_probability)

            stats[requester_id]["requests_sent"] += 1
            stats[target_id]["requests_received"] += 1
            stats[requester_id]["requested_target_ids"].append(target_id)
            stats[target_id]["incoming_requester_ids"].append(requester_id)
            if reciprocal_offer:
                stats[requester_id]["reciprocal_offers_sent"] += 1

            if accepted:
                observed_source_sets[requester_id].add(target_id)
                stats[requester_id]["accepted_requests_sent"] += 1
                stats[target_id]["accepted_requests_received"] += 1
                stats[requester_id]["accepted_target_ids"].append(target_id)
                stats[target_id]["accepted_incoming_requester_ids"].append(requester_id)
                if reciprocal_offer:
                    observed_source_sets[target_id].add(requester_id)
                    stats[requester_id]["reciprocal_exchanges"] += 1
                    stats[target_id]["reciprocal_exchanges"] += 1
                    stats[requester_id]["reciprocal_observed_agent_ids"].append(target_id)
                    stats[target_id]["reciprocal_observed_agent_ids"].append(requester_id)
            else:
                stats[requester_id]["rejected_requests_sent"] += 1
                stats[target_id]["rejected_requests_received"] += 1
                stats[requester_id]["rejected_target_ids"].append(target_id)
                stats[target_id]["rejected_incoming_requester_ids"].append(requester_id)

            exchanges.append(
                NegotiationExchange(
                    round_index=round_index,
                    requester_id=requester_id,
                    target_id=target_id,
                    reciprocal_offer=reciprocal_offer,
                    accepted=accepted,
                )
            )

    observations: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for reader_id, visible in enumerate(observed_source_sets):
        observed_ids = np.array(sorted(visible), dtype=int)
        stats[reader_id]["observed_count"] = len(observed_ids)
        stats[reader_id]["observed_agent_ids"] = [int(agent_id) for agent_id in observed_ids]

        observed_positions = positions[observed_ids].copy()
        if config.observation_noise > 0.0 and len(observed_ids) > 0:
            observed_positions += rng.normal(
                loc=0.0,
                scale=config.observation_noise,
                size=observed_positions.shape,
            )
            observed_positions = np.clip(observed_positions, config.lower, config.upper)

        observations.append((observed_ids, observed_positions, scores[observed_ids].copy()))

    return observations, exchanges, stats


def _update_agent_position(
    strategy: Any,
    observation: PeakObservation,
    rng: np.random.Generator,
    config: PeakGameConfig,
) -> np.ndarray:
    return strategy.update_position(
        observation=observation,
        rng=rng,
        config=config,
    )


def _update_all_positions(
    strategies: Sequence[Any],
    observations: Sequence[PeakObservation],
    agent_rngs: Sequence[np.random.Generator],
    config: PeakGameConfig,
) -> list[np.ndarray]:
    if not config.parallel_agent_updates or len(strategies) <= 1:
        return [
            _update_agent_position(strategy, observation, agent_rng, config)
            for strategy, observation, agent_rng in zip(strategies, observations, agent_rngs)
        ]

    max_workers = config.max_parallel_agent_updates or len(strategies)
    max_workers = max(1, min(max_workers, len(strategies)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(
            executor.map(
                _update_agent_position,
                strategies,
                observations,
                agent_rngs,
                [config] * len(strategies),
            )
        )


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

    position_history = [positions.copy()]
    action_history: list[list[CollaborationAction]] = []
    negotiation_history: list[list[NegotiationExchange]] = []
    communication_history: list[list[dict[str, Any]]] = []
    observed_count_history: list[np.ndarray] = []
    round_metrics: list[dict[str, Any]] = []
    previous_state: tuple[np.ndarray, np.ndarray] | None = None
    initial_scores, initial_values, _, _, _, _ = score_positions(positions, landscape, config)
    best_index = int(np.argmax(initial_values))
    best_score_found = float(initial_scores[best_index])
    best_value_found = float(initial_values[best_index])
    best_value_found_round = 0
    best_value_found_agent_id = best_index
    cumulative_discovered_peak_ids = {
        int(peak_id) for peak_id in discovered_peak_ids(positions, landscape, config).tolist()
    }
    target_top_peak_ids = {int(peak_id) for peak_id in top_peak_ids(landscape, config).tolist()}
    top_peak_first_success_round = (
        0 if target_top_peak_ids and target_top_peak_ids <= cumulative_discovered_peak_ids else -1
    )

    for round_index in range(config.rounds):
        scores, values, diversity, origin, _, _ = score_positions(positions, landscape, config)
        actions: list[CollaborationAction] = [
            strategy.choose_collaboration(
                round_index=round_index,
                position=positions[agent_id].copy(),
                score=float(scores[agent_id]),
                config=config,
                rng=agent_rngs[agent_id],
            )
            for agent_id, strategy in enumerate(strategies)
        ]
        visibility_levels = [action.visibility for action in actions]
        request_levels = [action.request_count for action in actions]
        action_history.append(actions)

        if config.delayed_observation and previous_state is not None:
            observable_positions, observable_scores = previous_state
        else:
            observable_positions = positions
            observable_scores = scores

        allocated, exchanges, communication_stats = _run_negotiation(
            observable_positions,
            observable_scores,
            actions,
            rng,
            config,
            round_index,
        )
        negotiation_history.append(exchanges)
        communication_history.append(communication_stats)
        observed_count_history.append(
            np.array([len(observed_ids) for observed_ids, _, _ in allocated], dtype=int)
        )

        observations: list[PeakObservation] = []
        for agent_id in range(config.num_agents):
            observed_ids, observed_positions, observed_scores = allocated[agent_id]
            observations.append(
                PeakObservation(
                    round_index=round_index,
                    agent_id=agent_id,
                    own_position=positions[agent_id].copy(),
                    own_score=float(scores[agent_id]),
                    observed_ids=observed_ids,
                    observed_positions=observed_positions,
                    observed_scores=observed_scores,
                    communication_feedback=communication_stats[agent_id].copy(),
                )
            )

        next_positions = _update_all_positions(strategies, observations, agent_rngs, config)

        previous_state = (positions.copy(), scores.copy())
        positions = np.clip(np.vstack(next_positions).astype(float), config.lower, config.upper)
        position_history.append(positions.copy())
        current_scores, current_values, _, _, _, _ = score_positions(positions, landscape, config)
        cumulative_discovered_peak_ids.update(
            int(peak_id) for peak_id in discovered_peak_ids(positions, landscape, config).tolist()
        )
        if (
            top_peak_first_success_round < 0
            and target_top_peak_ids
            and target_top_peak_ids <= cumulative_discovered_peak_ids
        ):
            top_peak_first_success_round = round_index + 1
        current_best_index = int(np.argmax(current_values))
        current_best_value = float(current_values[current_best_index])
        if current_best_value > best_value_found:
            best_score_found = float(current_scores[current_best_index])
            best_value_found = current_best_value
            best_value_found_round = round_index + 1
            best_value_found_agent_id = current_best_index
        metrics = summarize_positions(
            positions,
            landscape,
            config,
            visibility_levels=visibility_levels,
            request_levels=request_levels,
        )
        metrics["round"] = round_index
        metrics["best_score_found"] = best_score_found
        metrics["best_value_found"] = best_value_found
        metrics["best_value_found_ratio"] = value_ratio(best_value_found, landscape)
        metrics["best_value_found_gap"] = max(
            0.0,
            float(np.max(landscape.heights) - best_value_found),
        )
        metrics["best_value_found_round"] = float(best_value_found_round)
        metrics["best_value_found_agent_id"] = float(best_value_found_agent_id)
        metrics.update(
            top_peak_discovery_metrics(
                cumulative_discovered_peak_ids,
                landscape,
                config,
                prefix="top_peak",
            )
        )
        metrics["top_peak_first_success_round"] = float(top_peak_first_success_round)
        if communication_stats:
            metrics["mean_observed_count"] = float(
                np.mean([row["observed_count"] for row in communication_stats])
            )
            metrics["request_count"] = float(sum(row["requests_sent"] for row in communication_stats))
            metrics["acceptance_rate"] = _safe_rate(
                sum(row["accepted_requests_sent"] for row in communication_stats),
                sum(row["requests_sent"] for row in communication_stats),
            )
            metrics["rejection_rate"] = _safe_rate(
                sum(row["rejected_requests_sent"] for row in communication_stats),
                sum(row["requests_sent"] for row in communication_stats),
            )
            metrics["reciprocal_offer_rate"] = _safe_rate(
                sum(row["reciprocal_offers_sent"] for row in communication_stats),
                sum(row["requests_sent"] for row in communication_stats),
            )
            metrics["reciprocal_exchange_rate"] = _safe_rate(
                sum(row["reciprocal_exchanges"] for row in communication_stats),
                max(1, sum(row["requests_sent"] for row in communication_stats)),
            )
        round_metrics.append(metrics)

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
        best_score_found=best_score_found,
        best_value_found=best_value_found,
        best_value_found_round=best_value_found_round,
        best_value_found_agent_id=best_value_found_agent_id,
        position_history=position_history,
        action_history=action_history,
        negotiation_history=negotiation_history,
        communication_history=communication_history,
        observed_count_history=observed_count_history,
        round_metrics=round_metrics,
        agent_records=agent_records,
    )

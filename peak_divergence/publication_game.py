from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import numpy as np

from .core import (
    PeakGameConfig,
    PeakLandscape,
    PeakObservation,
    PublicationRunResult,
    PublishedRecord,
)
from .game import (
    discovered_peak_ids,
    generate_landscape,
    score_positions,
    summarize_positions,
    top_peak_discovery_metrics,
    top_peak_ids,
    value_ratio,
)


def is_published_location(
    position: np.ndarray,
    published_positions: np.ndarray,
    *,
    atol: float = 1e-9,
) -> bool:
    """Return True when a candidate exactly matches a public registry location."""

    if published_positions.size == 0:
        return False
    return bool(
        np.any(np.all(np.isclose(published_positions, position, rtol=0.0, atol=atol), axis=1))
    )


def registry_arrays(
    records: Sequence[PublishedRecord],
    config: PeakGameConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        return (
            np.empty(0, dtype=int),
            np.empty((0, config.dimensions), dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=int),
        )
    return (
        np.array([record.agent_id for record in records], dtype=int),
        np.vstack([record.position for record in records]).astype(float),
        np.array([record.score for record in records], dtype=float),
        np.array([record.round_index for record in records], dtype=int),
    )


def run_publication_game(
    strategies: Sequence[Any],
    config: PeakGameConfig,
    seed: int = 0,
    landscape: PeakLandscape | None = None,
    *,
    exact_match_atol: float = 1e-9,
    max_blocked_resubmissions: int = 6,
) -> PublicationRunResult:
    """Run the first research scenario: optional public publication only.

    Agents receive only their own score, private history maintained by their
    strategy, and the public registry of published (location, true score) pairs.
    There is no private exchange. Published locations cannot be submitted again
    exactly.
    """

    if len(strategies) != config.num_agents:
        raise ValueError(f"Expected {config.num_agents} strategies, got {len(strategies)}")

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

    published_records: list[PublishedRecord] = []
    publication_history: list[list[dict[str, Any]]] = []
    position_history = [positions.copy()]
    round_metrics: list[dict[str, Any]] = []

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
        scores, values, _, _, _, _ = score_positions(positions, landscape, config)
        current_best_index = int(np.argmax(values))
        current_best_value = float(values[current_best_index])
        if current_best_value > best_value_found:
            best_score_found = float(scores[current_best_index])
            best_value_found = current_best_value
            best_value_found_round = round_index
            best_value_found_agent_id = current_best_index

        public_ids, public_positions, public_scores, public_rounds = registry_arrays(
            published_records,
            config,
        )
        observations = [
            PeakObservation(
                round_index=round_index,
                agent_id=agent_id,
                own_position=positions[agent_id].copy(),
                own_score=float(scores[agent_id]),
                observed_ids=public_ids.copy(),
                observed_positions=public_positions.copy(),
                observed_scores=public_scores.copy(),
                communication_feedback={
                    "communication_mode": "public_publication",
                    "public_registry_size": len(published_records),
                    "public_record_rounds": public_rounds.tolist(),
                    "private_exchange_allowed": False,
                    "published_locations_blocked": True,
                },
            )
            for agent_id in range(config.num_agents)
        ]

        decisions = _update_publication_decisions(strategies, observations, agent_rngs, config)
        proposed_next_positions = [decision[0] for decision in decisions]
        publish_decisions = [bool(decision[1]) for decision in decisions]

        round_records: list[dict[str, Any]] = []
        registry_before_count = len(published_records)
        for agent_id, should_publish in enumerate(publish_decisions):
            current_already_published = is_published_location(
                positions[agent_id],
                public_positions,
                atol=exact_match_atol,
            )
            published = bool(should_publish and not current_already_published)
            if published:
                published_records.append(
                    PublishedRecord(
                        round_index=round_index,
                        agent_id=agent_id,
                        position=positions[agent_id].copy(),
                        score=float(scores[agent_id]),
                    )
                )
            round_records.append(
                {
                    "round": round_index,
                    "agent_id": agent_id,
                    "score": float(scores[agent_id]),
                    "publish_decision": bool(should_publish),
                    "published": published,
                    "current_location_already_published": bool(current_already_published),
                    "public_registry_size_before": registry_before_count,
                    "blocked_resubmissions": 0,
                }
            )

        _, updated_public_positions, _, _ = registry_arrays(published_records, config)
        valid_next_positions: list[np.ndarray] = []
        for agent_id, proposed_position in enumerate(proposed_next_positions):
            next_position, blocked_count = _repair_until_unpublished(
                strategy=strategies[agent_id],
                proposed_position=np.clip(proposed_position, config.lower, config.upper),
                observation=observations[agent_id],
                rng=agent_rngs[agent_id],
                config=config,
                published_positions=updated_public_positions,
                exact_match_atol=exact_match_atol,
                max_blocked_resubmissions=max_blocked_resubmissions,
            )
            round_records[agent_id]["blocked_resubmissions"] = blocked_count
            round_records[agent_id]["next_position_blocked_initially"] = blocked_count > 0
            valid_next_positions.append(next_position)

        publication_history.append(round_records)
        positions = np.clip(np.vstack(valid_next_positions).astype(float), config.lower, config.upper)
        position_history.append(positions.copy())

        next_scores, next_values, _, _, _, _ = score_positions(positions, landscape, config)
        cumulative_discovered_peak_ids.update(
            int(peak_id) for peak_id in discovered_peak_ids(positions, landscape, config).tolist()
        )
        if (
            top_peak_first_success_round < 0
            and target_top_peak_ids
            and target_top_peak_ids <= cumulative_discovered_peak_ids
        ):
            top_peak_first_success_round = round_index + 1
        next_best_index = int(np.argmax(next_values))
        next_best_value = float(next_values[next_best_index])
        if next_best_value > best_value_found:
            best_score_found = float(next_scores[next_best_index])
            best_value_found = next_best_value
            best_value_found_round = round_index + 1
            best_value_found_agent_id = next_best_index

        metrics = summarize_positions(positions, landscape, config)
        metrics["round"] = round_index
        metrics["communication_mode"] = "public_publication"
        metrics["public_registry_size"] = float(len(published_records))
        metrics["published_count_this_round"] = float(
            sum(1 for record in round_records if record["published"])
        )
        metrics["publish_decision_count"] = float(sum(publish_decisions))
        metrics["publish_rate"] = float(np.mean(publish_decisions))
        metrics["blocked_resubmission_count"] = float(
            sum(record["blocked_resubmissions"] for record in round_records)
        )
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
        round_metrics.append(metrics)

    final_scores, final_values, final_diversity, final_origin, final_peak_ids, _ = score_positions(
        positions,
        landscape,
        config,
    )
    return PublicationRunResult(
        config=config,
        landscape=landscape,
        seed=seed,
        positions=positions,
        final_scores=final_scores,
        final_values=final_values,
        final_diversity=final_diversity,
        final_origin=final_origin,
        final_peak_ids=final_peak_ids,
        published_records=published_records,
        publication_history=publication_history,
        position_history=position_history,
        round_metrics=round_metrics,
        best_score_found=best_score_found,
        best_value_found=best_value_found,
        best_value_found_round=best_value_found_round,
        best_value_found_agent_id=best_value_found_agent_id,
    )


def _update_publication_decisions(
    strategies: Sequence[Any],
    observations: Sequence[PeakObservation],
    agent_rngs: Sequence[np.random.Generator],
    config: PeakGameConfig,
) -> list[tuple[np.ndarray, bool]]:
    if not config.parallel_agent_updates or len(strategies) <= 1:
        return [
            strategy.update_publication_decision(observation, agent_rng, config)
            for strategy, observation, agent_rng in zip(strategies, observations, agent_rngs)
        ]

    max_workers = config.max_parallel_agent_updates or len(strategies)
    max_workers = max(1, min(max_workers, len(strategies)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(
            executor.map(
                _update_one_publication_decision,
                strategies,
                observations,
                agent_rngs,
                [config] * len(strategies),
            )
        )


def _update_one_publication_decision(
    strategy: Any,
    observation: PeakObservation,
    rng: np.random.Generator,
    config: PeakGameConfig,
) -> tuple[np.ndarray, bool]:
    return strategy.update_publication_decision(observation, rng, config)


def _repair_until_unpublished(
    *,
    strategy: Any,
    proposed_position: np.ndarray,
    observation: PeakObservation,
    rng: np.random.Generator,
    config: PeakGameConfig,
    published_positions: np.ndarray,
    exact_match_atol: float,
    max_blocked_resubmissions: int,
) -> tuple[np.ndarray, int]:
    position = np.clip(proposed_position.astype(float), config.lower, config.upper)
    blocked_count = 0
    while is_published_location(position, published_positions, atol=exact_match_atol):
        blocked_count += 1
        if blocked_count > max_blocked_resubmissions:
            position = _tiny_unpublished_jitter(
                position,
                published_positions,
                rng,
                config,
                exact_match_atol,
            )
            break
        feedback = dict(observation.communication_feedback)
        feedback["blocked_resubmissions"] = blocked_count
        feedback["blocked_position"] = [round(float(value), 8) for value in position]
        retry_observation = PeakObservation(
            round_index=observation.round_index,
            agent_id=observation.agent_id,
            own_position=observation.own_position,
            own_score=observation.own_score,
            observed_ids=observation.observed_ids,
            observed_positions=observation.observed_positions.copy(),
            observed_scores=observation.observed_scores,
            communication_feedback=feedback,
        )
        position = np.clip(
            strategy.repair_blocked_position(position, retry_observation, rng, config),
            config.lower,
            config.upper,
        )
    return position, blocked_count


def _tiny_unpublished_jitter(
    position: np.ndarray,
    published_positions: np.ndarray,
    rng: np.random.Generator,
    config: PeakGameConfig,
    exact_match_atol: float,
) -> np.ndarray:
    scale = max(1e-5, (config.upper - config.lower) * 1e-6)
    candidate = position.copy()
    for _ in range(100):
        candidate = np.clip(
            position + rng.normal(0.0, scale, size=config.dimensions),
            config.lower,
            config.upper,
        )
        if not is_published_location(candidate, published_positions, atol=exact_match_atol):
            return candidate
        scale *= 2.0
    raise RuntimeError("Could not produce an unpublished continuous location")

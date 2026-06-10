from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .core import CollaborationAction, Level, PeakGameConfig, PeakObservation


def _level_for_target(target: int, config: PeakGameConfig) -> Level:
    return "all" if config.num_agents - 1 <= target else target


def _sample_level(rng: np.random.Generator, config: PeakGameConfig) -> Level:
    level = rng.choice(np.array(config.share_read_levels, dtype=object))
    return level.item() if hasattr(level, "item") else level


def _random_corner(rng: np.random.Generator, config: PeakGameConfig, count: int = 1) -> np.ndarray:
    bits = rng.integers(0, 2, size=(count, config.dimensions))
    corners = np.where(bits == 1, config.upper, config.lower).astype(float)
    return corners[0] if count == 1 else corners


def _clip(position: np.ndarray, config: PeakGameConfig) -> np.ndarray:
    return np.clip(position.astype(float), config.lower, config.upper)


def _coordinate_best_response(
    observed_positions: np.ndarray,
    config: PeakGameConfig,
) -> np.ndarray:
    if observed_positions.size == 0:
        return _random_corner(np.random.default_rng(), config)
    means = observed_positions.mean(axis=0)
    return np.where(means <= config.upper / 2.0, config.upper, config.lower).astype(float)


def _surrogate_value(
    candidates: np.ndarray,
    anchor_positions: np.ndarray,
    anchor_values: np.ndarray,
    length_scale: float,
) -> np.ndarray:
    if anchor_positions.size == 0 or anchor_values.size == 0:
        return np.zeros(candidates.shape[0], dtype=float)
    deltas = candidates[:, None, :] - anchor_positions[None, :, :]
    squared_l2 = np.sum(deltas * deltas, axis=2)
    contributions = anchor_values[None, :] * np.exp(-squared_l2 / (2.0 * length_scale**2))
    return contributions.max(axis=1)


def _distance_terms(
    candidates: np.ndarray,
    observed_positions: np.ndarray,
    config: PeakGameConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if observed_positions.size == 0:
        neutral = np.full(candidates.shape[0], config.upper / 2.0)
        return neutral, neutral
    distances = np.abs(candidates[:, None, :] - observed_positions[None, :, :]).mean(axis=2)
    return distances.mean(axis=1), distances.min(axis=1)


def _candidate_pool(
    rng: np.random.Generator,
    config: PeakGameConfig,
    current_position: np.ndarray,
    anchor_positions: np.ndarray,
    count: int,
    progress: float,
) -> np.ndarray:
    uniform_count = max(12, count // 10)
    corner_count = max(12, count // 12)
    mutation_count = max(20, count - uniform_count - corner_count)
    local_scale = max(3.0, 26.0 * (1.0 - progress))
    broad_scale = max(8.0, 42.0 * (1.0 - progress))

    candidates = [
        rng.uniform(config.lower, config.upper, size=(uniform_count, config.dimensions)),
        _random_corner(rng, config, count=corner_count),
        current_position[None, :],
    ]

    if anchor_positions.size > 0:
        anchor_indices = rng.choice(
            len(anchor_positions),
            size=mutation_count,
            replace=len(anchor_positions) < mutation_count,
        )
        anchors = anchor_positions[anchor_indices]
        scales = rng.choice(
            np.array([local_scale, broad_scale]),
            size=(mutation_count, 1),
            p=np.array([0.7, 0.3]),
        )
        candidates.append(anchors + rng.normal(0.0, scales, size=anchors.shape))
        candidates.append(anchor_positions[: min(len(anchor_positions), 24)])
    else:
        candidates.append(
            current_position
            + rng.normal(0.0, broad_scale, size=(mutation_count, config.dimensions))
        )

    return np.unique(
        np.round(_clip(np.vstack(candidates), config), decimals=8),
        axis=0,
    )


def _soft_top_choice(
    candidates: np.ndarray,
    scores: np.ndarray,
    rng: np.random.Generator,
    top_k: int = 10,
    temperature: float = 5.0,
) -> np.ndarray:
    top_k = max(1, min(top_k, len(scores)))
    top_indices = np.argsort(scores)[-top_k:]
    top_scores = scores[top_indices]
    scaled = (top_scores - top_scores.max()) / max(temperature, 1e-9)
    weights = np.exp(scaled)
    weights = weights / weights.sum()
    return candidates[int(rng.choice(top_indices, p=weights))].copy()


@dataclass
class MemoryMixin:
    memory_limit: int = 180
    memory_positions: list[np.ndarray] = field(default_factory=list)
    memory_values: list[float] = field(default_factory=list)
    memory_scores: list[float] = field(default_factory=list)

    def remember(
        self,
        position: np.ndarray,
        value: float,
        score: float,
        observed_positions: np.ndarray,
        observed_values: np.ndarray,
        observed_scores: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        self.memory_positions.append(position.copy())
        self.memory_values.append(float(value))
        self.memory_scores.append(float(score))
        for pos, val, scr in zip(observed_positions, observed_values, observed_scores):
            self.memory_positions.append(pos.copy())
            self.memory_values.append(float(val))
            self.memory_scores.append(float(scr))

        if len(self.memory_positions) > self.memory_limit:
            values = np.array(self.memory_values)
            elite_count = min(self.memory_limit // 2, len(values))
            elite_indices = np.argsort(values)[-elite_count:]
            remaining = np.setdiff1d(np.arange(len(values)), elite_indices, assume_unique=False)
            random_count = self.memory_limit - elite_count
            if len(remaining) > random_count:
                random_indices = rng.choice(remaining, size=random_count, replace=False)
            else:
                random_indices = remaining
            keep = np.concatenate([elite_indices, random_indices]).astype(int)
            self.memory_positions = [self.memory_positions[int(idx)] for idx in keep]
            self.memory_values = [self.memory_values[int(idx)] for idx in keep]
            self.memory_scores = [self.memory_scores[int(idx)] for idx in keep]

    def memory_arrays(self, config: PeakGameConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.memory_positions:
            return (
                np.empty((0, config.dimensions)),
                np.empty(0),
                np.empty(0),
            )
        return (
            np.vstack(self.memory_positions).astype(float),
            np.array(self.memory_values, dtype=float),
            np.array(self.memory_scores, dtype=float),
        )

    def top_value_anchors(self, config: PeakGameConfig, max_count: int = 60) -> np.ndarray:
        positions, values, _ = self.memory_arrays(config)
        if positions.size == 0:
            return positions
        count = min(max_count, len(values))
        return positions[np.argsort(values)[-count:]]


class Strategy:
    name = "strategy"

    def initial_position(
        self,
        agent_id: int,
        rng: np.random.Generator,
        config: PeakGameConfig,
    ) -> np.ndarray:
        return rng.uniform(config.lower, config.upper, size=config.dimensions)

    def choose_collaboration(
        self,
        round_index: int,
        position: np.ndarray,
        score: float,
        value: float,
        diversity: float,
        origin: float,
        config: PeakGameConfig,
        rng: np.random.Generator,
        previous_metrics: dict[str, float] | None,
    ) -> CollaborationAction:
        return CollaborationAction(share=0, read=0)

    def update_position(
        self,
        observation: PeakObservation,
        rng: np.random.Generator,
        config: PeakGameConfig,
    ) -> np.ndarray:
        return observation.own_position


class RandomStrategy(Strategy):
    name = "random"

    def update_position(
        self,
        observation: PeakObservation,
        rng: np.random.Generator,
        config: PeakGameConfig,
    ) -> np.ndarray:
        return rng.uniform(config.lower, config.upper, size=config.dimensions)


@dataclass
class RandomCornerStrategy(Strategy):
    name = "random_corner"
    corner: np.ndarray | None = None

    def initial_position(
        self,
        agent_id: int,
        rng: np.random.Generator,
        config: PeakGameConfig,
    ) -> np.ndarray:
        self.corner = _random_corner(rng, config)
        return self.corner.copy()

    def update_position(
        self,
        observation: PeakObservation,
        rng: np.random.Generator,
        config: PeakGameConfig,
    ) -> np.ndarray:
        if self.corner is None:
            self.corner = _random_corner(rng, config)
        return self.corner.copy()


class OriginMaximizerStrategy(Strategy):
    name = "origin_maximizer"

    def initial_position(
        self,
        agent_id: int,
        rng: np.random.Generator,
        config: PeakGameConfig,
    ) -> np.ndarray:
        return np.full(config.dimensions, config.upper, dtype=float)

    def update_position(
        self,
        observation: PeakObservation,
        rng: np.random.Generator,
        config: PeakGameConfig,
    ) -> np.ndarray:
        return np.full(config.dimensions, config.upper, dtype=float)


@dataclass
class IndependentSearchStrategy(MemoryMixin, Strategy):
    name = "independent_search"
    memory_limit: int = 60

    def update_position(
        self,
        observation: PeakObservation,
        rng: np.random.Generator,
        config: PeakGameConfig,
    ) -> np.ndarray:
        self.remember(
            observation.own_position,
            observation.own_value,
            observation.own_score,
            observation.observed_positions,
            observation.observed_values,
            observation.observed_scores,
            rng,
        )
        anchors = self.top_value_anchors(config, max_count=8)
        progress = observation.round_index / max(1, config.rounds - 1)
        if anchors.size == 0 or rng.random() < 0.18 * (1.0 - progress):
            return rng.uniform(config.lower, config.upper, size=config.dimensions)
        center = anchors[-1]
        scale = max(3.5, 24.0 * (1.0 - progress))
        return _clip(center + rng.normal(0.0, scale, size=config.dimensions), config)


@dataclass
class ValueOnlyStrategy(MemoryMixin, Strategy):
    name = "value_only"
    memory_limit: int = 240

    def choose_collaboration(self, *args, **kwargs) -> CollaborationAction:
        return CollaborationAction(share="all", read="all")

    def update_position(
        self,
        observation: PeakObservation,
        rng: np.random.Generator,
        config: PeakGameConfig,
    ) -> np.ndarray:
        self.remember(
            observation.own_position,
            observation.own_value,
            observation.own_score,
            observation.observed_positions,
            observation.observed_values,
            observation.observed_scores,
            rng,
        )
        positions, values, _ = self.memory_arrays(config)
        best_position = positions[int(np.argmax(values))]
        progress = observation.round_index / max(1, config.rounds - 1)
        scale = max(2.0, 18.0 * (1.0 - progress))
        return _clip(best_position + rng.normal(0.0, scale, size=config.dimensions), config)


class DiversityOnlyStrategy(Strategy):
    name = "diversity_only"

    def choose_collaboration(
        self,
        round_index: int,
        position: np.ndarray,
        score: float,
        value: float,
        diversity: float,
        origin: float,
        config: PeakGameConfig,
        rng: np.random.Generator,
        previous_metrics: dict[str, float] | None,
    ) -> CollaborationAction:
        return CollaborationAction(
            share=_level_for_target(20, config),
            read=_level_for_target(20, config),
        )

    def update_position(
        self,
        observation: PeakObservation,
        rng: np.random.Generator,
        config: PeakGameConfig,
    ) -> np.ndarray:
        if observation.observed_positions.size == 0:
            return _random_corner(rng, config)
        return _coordinate_best_response(observation.observed_positions, config)


@dataclass
class FullCollaborationStrategy(MemoryMixin, Strategy):
    name = "full_collaboration"
    memory_limit: int = 400

    def choose_collaboration(self, *args, **kwargs) -> CollaborationAction:
        return CollaborationAction(share="all", read="all")

    def update_position(
        self,
        observation: PeakObservation,
        rng: np.random.Generator,
        config: PeakGameConfig,
    ) -> np.ndarray:
        self.remember(
            observation.own_position,
            observation.own_value,
            observation.own_score,
            observation.observed_positions,
            observation.observed_values,
            observation.observed_scores,
            rng,
        )
        positions, _, scores = self.memory_arrays(config)
        best_position = positions[int(np.argmax(scores))]
        progress = observation.round_index / max(1, config.rounds - 1)
        scale = max(1.0, 10.0 * (1.0 - progress))
        return _clip(best_position + rng.normal(0.0, scale, size=config.dimensions), config)


@dataclass
class RandomCollaborationStrategy(MemoryMixin, Strategy):
    name = "random_collaboration"
    memory_limit: int = 180

    def choose_collaboration(
        self,
        round_index: int,
        position: np.ndarray,
        score: float,
        value: float,
        diversity: float,
        origin: float,
        config: PeakGameConfig,
        rng: np.random.Generator,
        previous_metrics: dict[str, float] | None,
    ) -> CollaborationAction:
        return CollaborationAction(
            share=_sample_level(rng, config),
            read=_sample_level(rng, config),
        )

    def update_position(
        self,
        observation: PeakObservation,
        rng: np.random.Generator,
        config: PeakGameConfig,
    ) -> np.ndarray:
        self.remember(
            observation.own_position,
            observation.own_value,
            observation.own_score,
            observation.observed_positions,
            observation.observed_values,
            observation.observed_scores,
            rng,
        )
        positions, values, _ = self.memory_arrays(config)
        progress = observation.round_index / max(1, config.rounds - 1)
        anchors = self.top_value_anchors(config, max_count=50)
        candidates = _candidate_pool(
            rng,
            config,
            observation.own_position,
            anchors,
            count=180,
            progress=progress,
        )
        v_hat = _surrogate_value(candidates, positions, values, length_scale=24.0)
        mean_distance, nearest_distance = _distance_terms(
            candidates,
            observation.observed_positions,
            config,
        )
        origin = candidates.mean(axis=1)
        scores = v_hat * (
            1.0
            + config.beta_diversity * (0.75 * mean_distance + 0.25 * nearest_distance)
            + config.gamma_origin * origin
        )
        scores += rng.normal(0.0, max(1.0, 8.0 * (1.0 - progress)), size=scores.shape)
        return _soft_top_choice(candidates, scores, rng, top_k=16, temperature=8.0)


@dataclass
class StrategicCollaborationStrategy(MemoryMixin, Strategy):
    name = "strategic_collaboration"
    memory_limit: int = 320

    def choose_collaboration(
        self,
        round_index: int,
        position: np.ndarray,
        score: float,
        value: float,
        diversity: float,
        origin: float,
        config: PeakGameConfig,
        rng: np.random.Generator,
        previous_metrics: dict[str, float] | None,
    ) -> CollaborationAction:
        progress = round_index / max(1, config.rounds - 1)
        expected_peak_value = 0.35 * config.peak_height_range[1]
        is_low_value = value < expected_peak_value

        if progress < 0.25 or is_low_value:
            return CollaborationAction(
                share=_level_for_target(20, config),
                read=_level_for_target(100, config),
            )
        if progress < 0.65:
            return CollaborationAction(
                share=_level_for_target(5, config),
                read=_level_for_target(20, config),
            )
        if progress < 0.90:
            return CollaborationAction(share=0, read=_level_for_target(5, config))
        return CollaborationAction(share=0, read=0)

    def update_position(
        self,
        observation: PeakObservation,
        rng: np.random.Generator,
        config: PeakGameConfig,
    ) -> np.ndarray:
        self.remember(
            observation.own_position,
            observation.own_value,
            observation.own_score,
            observation.observed_positions,
            observation.observed_values,
            observation.observed_scores,
            rng,
        )
        positions, values, scores_seen = self.memory_arrays(config)
        progress = observation.round_index / max(1, config.rounds - 1)
        anchors = self.top_value_anchors(config, max_count=80)
        candidates = _candidate_pool(
            rng,
            config,
            observation.own_position,
            anchors,
            count=320,
            progress=progress,
        )

        value_length_scale = 26.0 if progress < 0.5 else 18.0
        v_hat = _surrogate_value(candidates, positions, values, length_scale=value_length_scale)
        local_reference = observation.observed_positions
        if local_reference.size == 0 and positions.size:
            top_score_count = min(80, len(scores_seen))
            local_reference = positions[np.argsort(scores_seen)[-top_score_count:]]

        mean_distance, nearest_distance = _distance_terms(candidates, local_reference, config)
        origin = candidates.mean(axis=1)
        estimated_multiplier = (
            1.0
            + config.beta_diversity * (0.55 * mean_distance + 0.45 * nearest_distance)
            + config.gamma_origin * origin
        )
        crowd_penalty = np.maximum(0.0, 18.0 - nearest_distance)
        estimated_scores = v_hat * estimated_multiplier - 0.75 * crowd_penalty

        if observation.own_value < 0.25 * config.peak_height_range[1]:
            estimated_scores += rng.normal(0.0, 10.0 * (1.0 - progress), size=len(candidates))
        else:
            estimated_scores += rng.normal(0.0, 3.0 * (1.0 - progress), size=len(candidates))

        return _soft_top_choice(
            candidates,
            estimated_scores,
            rng,
            top_k=12,
            temperature=5.0,
        )


StrategyFactory = Callable[[], Strategy]


_STRATEGIES: dict[str, StrategyFactory] = {
    "random": RandomStrategy,
    "random_corner": RandomCornerStrategy,
    "origin_maximizer": OriginMaximizerStrategy,
    "independent_search": IndependentSearchStrategy,
    "value_only": ValueOnlyStrategy,
    "diversity_only": DiversityOnlyStrategy,
    "full_collaboration": FullCollaborationStrategy,
    "random_collaboration": RandomCollaborationStrategy,
    "strategic_collaboration": StrategicCollaborationStrategy,
}


def available_strategies() -> tuple[str, ...]:
    return tuple(_STRATEGIES.keys())


def make_strategy(name: str) -> Strategy:
    try:
        return _STRATEGIES[name]()
    except KeyError as exc:
        available = ", ".join(available_strategies())
        raise ValueError(f"Unknown strategy {name!r}. Available: {available}") from exc


def make_population(strategy_name: str, num_agents: int) -> list[Strategy]:
    return [make_strategy(strategy_name) for _ in range(num_agents)]

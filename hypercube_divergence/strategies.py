from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .core import CollaborationAction, GameConfig, Level, Observation


def _level_for_target(target: int, config: GameConfig) -> Level:
    return "all" if config.num_agents - 1 <= target else target


def _sample_level(rng: np.random.Generator, config: GameConfig) -> Level:
    level = rng.choice(np.array(config.share_read_levels, dtype=object))
    return level.item() if hasattr(level, "item") else level


def _random_corner(rng: np.random.Generator, config: GameConfig, count: int = 1) -> np.ndarray:
    bits = rng.integers(0, 2, size=(count, config.dimensions))
    corners = np.where(bits == 1, config.upper, config.lower).astype(float)
    return corners[0] if count == 1 else corners


def _coordinate_best_response(
    observed_positions: np.ndarray,
    config: GameConfig,
    lambda_origin: float,
) -> np.ndarray:
    if observed_positions.size == 0:
        return np.full(config.dimensions, config.upper, dtype=float)
    means = observed_positions.mean(axis=0)
    threshold = config.upper * (1.0 + lambda_origin) / 2.0
    return np.where(means <= threshold, config.upper, config.lower).astype(float)


def _candidate_pool(
    rng: np.random.Generator,
    config: GameConfig,
    observed_positions: np.ndarray,
    current_position: np.ndarray,
    count: int,
) -> np.ndarray:
    random_corners = _random_corner(rng, config, count=count)
    random_points = rng.uniform(
        config.lower,
        config.upper,
        size=(max(8, count // 8), config.dimensions),
    )

    anchors = [
        np.full(config.dimensions, config.upper, dtype=float),
        np.full(config.dimensions, config.lower, dtype=float),
        current_position.copy(),
        _coordinate_best_response(observed_positions, config, config.lambda_origin),
        _coordinate_best_response(observed_positions, config, 0.0),
    ]
    candidates = np.vstack([random_corners, random_points, np.vstack(anchors)])
    return np.unique(np.round(candidates, decimals=8), axis=0)


def _estimate_scores(
    candidates: np.ndarray,
    observed_positions: np.ndarray,
    config: GameConfig,
    lambda_origin: float,
    nearest_weight: float = 0.0,
) -> np.ndarray:
    if observed_positions.size == 0:
        diversity = np.full(candidates.shape[0], config.upper / 2.0)
        nearest = diversity
    else:
        distances = np.abs(candidates[:, None, :] - observed_positions[None, :, :]).mean(axis=2)
        diversity = distances.mean(axis=1)
        nearest = distances.min(axis=1)
    origin = candidates.mean(axis=1)
    return (1.0 - nearest_weight) * diversity + nearest_weight * nearest + lambda_origin * origin


def _mean_field_origin_target(config: GameConfig) -> float:
    """Best Bernoulli-corner origin rate under an independent population model."""

    target_high_probability = np.clip(0.5 + config.lambda_origin / 4.0, 0.0, 1.0)
    return float(config.upper * target_high_probability)


def _soft_top_choice(
    candidates: np.ndarray,
    scores: np.ndarray,
    rng: np.random.Generator,
    top_k: int = 8,
    temperature: float = 3.0,
) -> np.ndarray:
    top_k = max(1, min(top_k, len(scores)))
    top_indices = np.argsort(scores)[-top_k:]
    top_scores = scores[top_indices]
    scaled = (top_scores - top_scores.max()) / max(temperature, 1e-9)
    weights = np.exp(scaled)
    weights = weights / weights.sum()
    return candidates[int(rng.choice(top_indices, p=weights))].copy()


class Strategy:
    name = "strategy"

    def initial_position(
        self, agent_id: int, rng: np.random.Generator, config: GameConfig
    ) -> np.ndarray:
        return rng.uniform(config.lower, config.upper, size=config.dimensions)

    def choose_collaboration(
        self,
        round_index: int,
        position: np.ndarray,
        config: GameConfig,
        rng: np.random.Generator,
        previous_metrics: dict[str, float] | None,
    ) -> CollaborationAction:
        return CollaborationAction(share=0, read=0)

    def update_position(
        self,
        observation: Observation,
        rng: np.random.Generator,
        config: GameConfig,
    ) -> np.ndarray:
        return observation.own_position


class IndependentStrategy(Strategy):
    name = "independent"

    def choose_collaboration(self, *args, **kwargs) -> CollaborationAction:
        return CollaborationAction(share=0, read=0)


class OriginMaximizerStrategy(Strategy):
    name = "origin_maximizer"

    def initial_position(
        self, agent_id: int, rng: np.random.Generator, config: GameConfig
    ) -> np.ndarray:
        return np.full(config.dimensions, config.upper, dtype=float)

    def choose_collaboration(self, *args, **kwargs) -> CollaborationAction:
        return CollaborationAction(share=0, read=0)

    def update_position(
        self,
        observation: Observation,
        rng: np.random.Generator,
        config: GameConfig,
    ) -> np.ndarray:
        return np.full(config.dimensions, config.upper, dtype=float)


@dataclass
class CornerRandomStrategy(Strategy):
    name = "corner_random"
    corner: np.ndarray | None = None

    def initial_position(
        self, agent_id: int, rng: np.random.Generator, config: GameConfig
    ) -> np.ndarray:
        self.corner = _random_corner(rng, config)
        return self.corner.copy()

    def choose_collaboration(self, *args, **kwargs) -> CollaborationAction:
        return CollaborationAction(share=0, read=0)

    def update_position(
        self,
        observation: Observation,
        rng: np.random.Generator,
        config: GameConfig,
    ) -> np.ndarray:
        if self.corner is None:
            self.corner = _random_corner(rng, config)
        return self.corner.copy()


class FullCollaborationBestResponseStrategy(Strategy):
    name = "full_collaboration"

    def choose_collaboration(self, *args, **kwargs) -> CollaborationAction:
        return CollaborationAction(share="all", read="all")

    def update_position(
        self,
        observation: Observation,
        rng: np.random.Generator,
        config: GameConfig,
    ) -> np.ndarray:
        return _coordinate_best_response(
            observation.observed_positions,
            config,
            lambda_origin=config.lambda_origin,
        )


class DiversityOnlyStrategy(Strategy):
    name = "diversity_only"

    def choose_collaboration(
        self,
        round_index: int,
        position: np.ndarray,
        config: GameConfig,
        rng: np.random.Generator,
        previous_metrics: dict[str, float] | None,
    ) -> CollaborationAction:
        return CollaborationAction(share=_level_for_target(20, config), read=_level_for_target(20, config))

    def update_position(
        self,
        observation: Observation,
        rng: np.random.Generator,
        config: GameConfig,
    ) -> np.ndarray:
        if observation.observed_positions.size == 0:
            return _random_corner(rng, config)
        return _coordinate_best_response(observation.observed_positions, config, lambda_origin=0.0)


class RandomCollaborationStrategy(Strategy):
    name = "random_collaboration"

    def choose_collaboration(
        self,
        round_index: int,
        position: np.ndarray,
        config: GameConfig,
        rng: np.random.Generator,
        previous_metrics: dict[str, float] | None,
    ) -> CollaborationAction:
        return CollaborationAction(
            share=_sample_level(rng, config),
            read=_sample_level(rng, config),
        )

    def update_position(
        self,
        observation: Observation,
        rng: np.random.Generator,
        config: GameConfig,
    ) -> np.ndarray:
        candidates = _candidate_pool(
            rng,
            config,
            observation.observed_positions,
            observation.own_position,
            count=160,
        )
        scores = _estimate_scores(
            candidates,
            observation.observed_positions,
            config,
            lambda_origin=config.lambda_origin,
            nearest_weight=0.15,
        )
        return _soft_top_choice(candidates, scores, rng, top_k=16, temperature=8.0)


@dataclass
class StrategicCollaborationStrategy(Strategy):
    name = "strategic_collaboration"
    memory_limit: int = 240
    memory: np.ndarray = field(default_factory=lambda: np.empty((0, 10)))

    def choose_collaboration(
        self,
        round_index: int,
        position: np.ndarray,
        config: GameConfig,
        rng: np.random.Generator,
        previous_metrics: dict[str, float] | None,
    ) -> CollaborationAction:
        progress = round_index / max(1, config.rounds - 1)
        if progress < 0.25:
            return CollaborationAction(
                share=_level_for_target(20, config),
                read=_level_for_target(100, config),
            )
        if progress < 0.70:
            return CollaborationAction(
                share=_level_for_target(5, config),
                read=_level_for_target(20, config),
            )
        if progress < 0.90:
            return CollaborationAction(share=0, read=_level_for_target(5, config))
        return CollaborationAction(share=0, read=0)

    def _remember(self, observed_positions: np.ndarray, rng: np.random.Generator) -> None:
        if observed_positions.size == 0:
            return
        if self.memory.shape[1] != observed_positions.shape[1]:
            self.memory = np.empty((0, observed_positions.shape[1]))
        self.memory = np.vstack([self.memory, observed_positions])
        if len(self.memory) > self.memory_limit:
            keep = rng.choice(len(self.memory), size=self.memory_limit, replace=False)
            self.memory = self.memory[keep]

    def update_position(
        self,
        observation: Observation,
        rng: np.random.Generator,
        config: GameConfig,
    ) -> np.ndarray:
        self._remember(observation.observed_positions, rng)
        if observation.observed_positions.size and self.memory.size:
            reference = np.vstack([observation.observed_positions, self.memory])
        elif self.memory.size:
            reference = self.memory
        else:
            reference = observation.observed_positions

        candidates = _candidate_pool(
            rng,
            config,
            reference,
            observation.own_position,
            count=384,
        )
        scores = _estimate_scores(
            candidates,
            reference,
            config,
            lambda_origin=config.lambda_origin,
            nearest_weight=0.30,
        )
        origin_target = _mean_field_origin_target(config)
        origin_deviation = np.abs(candidates.mean(axis=1) - origin_target)
        scores = scores - 0.45 * origin_deviation

        progress = observation.round_index / max(1, config.rounds - 1)
        exploration = max(0.0, 1.0 - progress)
        scores = scores + rng.normal(0.0, 1.5 * exploration, size=scores.shape)

        return _soft_top_choice(candidates, scores, rng, top_k=12, temperature=4.0)


StrategyFactory = Callable[[], Strategy]


_STRATEGIES: dict[str, StrategyFactory] = {
    "independent": IndependentStrategy,
    "origin_maximizer": OriginMaximizerStrategy,
    "corner_random": CornerRandomStrategy,
    "full_collaboration": FullCollaborationBestResponseStrategy,
    "diversity_only": DiversityOnlyStrategy,
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

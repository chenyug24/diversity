from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .core import CollaborationAction, Level, PeakGameConfig, PeakObservation
from .llm_agent import (
    LLMDecision,
    build_llm_prompt,
    call_openai_llm,
    decision_to_action,
    fallback_decision,
    parse_llm_decision,
)


def _level_for_target(target: int, config: PeakGameConfig) -> Level:
    return "all" if config.num_agents - 1 <= target else target


def _sample_level(rng: np.random.Generator, config: PeakGameConfig) -> Level:
    level = rng.choice(np.array(config.communication_levels, dtype=object))
    return level.item() if hasattr(level, "item") else level


def _negotiation_action(
    *,
    visibility: Level,
    request_count: Level,
    offer_reciprocal: bool = False,
    accept_probability: float = 0.5,
) -> CollaborationAction:
    return CollaborationAction(
        visibility=visibility,
        request_count=request_count,
        offer_reciprocal=offer_reciprocal,
        accept_probability=float(np.clip(accept_probability, 0.0, 1.0)),
    )


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
        return np.full(config.dimensions, config.upper, dtype=float)
    means = observed_positions.mean(axis=0)
    return np.where(means <= config.upper / 2.0, config.upper, config.lower).astype(float)


def _surrogate_score(
    candidates: np.ndarray,
    anchor_positions: np.ndarray,
    anchor_scores: np.ndarray,
    length_scale: float,
) -> np.ndarray:
    """Black-box score model from observed position/score pairs.

    Agents do not know the hidden value function or scoring formula. This local
    radial surrogate only says: points close to previously high-scoring points
    are expected to score well.
    """

    if anchor_positions.size == 0 or anchor_scores.size == 0:
        return np.zeros(candidates.shape[0], dtype=float)
    scores = np.maximum(anchor_scores, 0.0)
    deltas = candidates[:, None, :] - anchor_positions[None, :, :]
    squared_l2 = np.sum(deltas * deltas, axis=2)
    contributions = scores[None, :] * np.exp(-squared_l2 / (2.0 * length_scale**2))
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
    memory_scores: list[float] = field(default_factory=list)

    def remember(
        self,
        position: np.ndarray,
        score: float,
        observed_positions: np.ndarray,
        observed_scores: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        self.memory_positions.append(position.copy())
        self.memory_scores.append(float(score))
        for pos, scr in zip(observed_positions, observed_scores):
            self.memory_positions.append(pos.copy())
            self.memory_scores.append(float(scr))

        if len(self.memory_positions) > self.memory_limit:
            scores = np.array(self.memory_scores)
            elite_count = min(self.memory_limit // 2, len(scores))
            elite_indices = np.argsort(scores)[-elite_count:]
            remaining = np.setdiff1d(np.arange(len(scores)), elite_indices, assume_unique=False)
            random_count = self.memory_limit - elite_count
            if len(remaining) > random_count:
                random_indices = rng.choice(remaining, size=random_count, replace=False)
            else:
                random_indices = remaining
            keep = np.concatenate([elite_indices, random_indices]).astype(int)
            self.memory_positions = [self.memory_positions[int(idx)] for idx in keep]
            self.memory_scores = [self.memory_scores[int(idx)] for idx in keep]

    def memory_arrays(self, config: PeakGameConfig) -> tuple[np.ndarray, np.ndarray]:
        if not self.memory_positions:
            return np.empty((0, config.dimensions)), np.empty(0)
        return (
            np.vstack(self.memory_positions).astype(float),
            np.array(self.memory_scores, dtype=float),
        )

    def top_score_anchors(self, config: PeakGameConfig, max_count: int = 60) -> np.ndarray:
        positions, scores = self.memory_arrays(config)
        if positions.size == 0:
            return positions
        count = min(max_count, len(scores))
        return positions[np.argsort(scores)[-count:]]

    def score_quantile(self, quantile: float, default: float = 0.0) -> float:
        if not self.memory_scores:
            return default
        return float(np.quantile(np.array(self.memory_scores, dtype=float), quantile))


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
        config: PeakGameConfig,
        rng: np.random.Generator,
    ) -> CollaborationAction:
        return _negotiation_action(visibility=0, request_count=0, accept_probability=0.0)

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
    memory_limit: int = 80

    def update_position(
        self,
        observation: PeakObservation,
        rng: np.random.Generator,
        config: PeakGameConfig,
    ) -> np.ndarray:
        self.remember(
            observation.own_position,
            observation.own_score,
            observation.observed_positions,
            observation.observed_scores,
            rng,
        )
        anchors = self.top_score_anchors(config, max_count=10)
        progress = observation.round_index / max(1, config.rounds - 1)
        if anchors.size == 0 or rng.random() < 0.20 * (1.0 - progress):
            return rng.uniform(config.lower, config.upper, size=config.dimensions)
        center = anchors[-1]
        scale = max(3.5, 24.0 * (1.0 - progress))
        return _clip(center + rng.normal(0.0, scale, size=config.dimensions), config)


@dataclass
class ScoreFollowingStrategy(MemoryMixin, Strategy):
    name = "score_following"
    memory_limit: int = 260

    def choose_collaboration(self, *args, **kwargs) -> CollaborationAction:
        return _negotiation_action(
            visibility="all",
            request_count="all",
            offer_reciprocal=True,
            accept_probability=1.0,
        )

    def update_position(
        self,
        observation: PeakObservation,
        rng: np.random.Generator,
        config: PeakGameConfig,
    ) -> np.ndarray:
        self.remember(
            observation.own_position,
            observation.own_score,
            observation.observed_positions,
            observation.observed_scores,
            rng,
        )
        positions, scores = self.memory_arrays(config)
        best_position = positions[int(np.argmax(scores))]
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
        config: PeakGameConfig,
        rng: np.random.Generator,
    ) -> CollaborationAction:
        return _negotiation_action(
            visibility=_level_for_target(20, config),
            request_count=_level_for_target(20, config),
            offer_reciprocal=True,
            accept_probability=0.8,
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
    memory_limit: int = 420

    def choose_collaboration(self, *args, **kwargs) -> CollaborationAction:
        return _negotiation_action(
            visibility="all",
            request_count="all",
            offer_reciprocal=True,
            accept_probability=1.0,
        )

    def update_position(
        self,
        observation: PeakObservation,
        rng: np.random.Generator,
        config: PeakGameConfig,
    ) -> np.ndarray:
        self.remember(
            observation.own_position,
            observation.own_score,
            observation.observed_positions,
            observation.observed_scores,
            rng,
        )
        positions, scores = self.memory_arrays(config)
        best_position = positions[int(np.argmax(scores))]
        progress = observation.round_index / max(1, config.rounds - 1)
        scale = max(1.0, 9.0 * (1.0 - progress))
        return _clip(best_position + rng.normal(0.0, scale, size=config.dimensions), config)


@dataclass
class RandomCollaborationStrategy(MemoryMixin, Strategy):
    name = "random_collaboration"
    memory_limit: int = 200

    def choose_collaboration(
        self,
        round_index: int,
        position: np.ndarray,
        score: float,
        config: PeakGameConfig,
        rng: np.random.Generator,
    ) -> CollaborationAction:
        return _negotiation_action(
            visibility=_sample_level(rng, config),
            request_count=_sample_level(rng, config),
            offer_reciprocal=bool(rng.random() < 0.5),
            accept_probability=float(rng.uniform(0.1, 0.9)),
        )

    def update_position(
        self,
        observation: PeakObservation,
        rng: np.random.Generator,
        config: PeakGameConfig,
    ) -> np.ndarray:
        self.remember(
            observation.own_position,
            observation.own_score,
            observation.observed_positions,
            observation.observed_scores,
            rng,
        )
        positions, scores = self.memory_arrays(config)
        progress = observation.round_index / max(1, config.rounds - 1)
        anchors = self.top_score_anchors(config, max_count=55)
        candidates = _candidate_pool(
            rng,
            config,
            observation.own_position,
            anchors,
            count=190,
            progress=progress,
        )
        score_hat = _surrogate_score(candidates, positions, scores, length_scale=24.0)
        _, nearest_distance = _distance_terms(
            candidates,
            observation.observed_positions,
            config,
        )
        estimated_scores = score_hat + 0.45 * nearest_distance
        estimated_scores += rng.normal(0.0, max(1.0, 8.0 * (1.0 - progress)), size=len(candidates))
        return _soft_top_choice(candidates, estimated_scores, rng, top_k=16, temperature=8.0)


@dataclass
class StrategicCollaborationStrategy(MemoryMixin, Strategy):
    name = "strategic_collaboration"
    memory_limit: int = 340

    def choose_collaboration(
        self,
        round_index: int,
        position: np.ndarray,
        score: float,
        config: PeakGameConfig,
        rng: np.random.Generator,
    ) -> CollaborationAction:
        progress = round_index / max(1, config.rounds - 1)
        low_score = score < self.score_quantile(0.45, default=float("inf"))
        high_score = score >= self.score_quantile(0.75, default=float("inf"))

        if progress < 0.25 or low_score:
            visibility: Level = _level_for_target(20, config)
            if high_score and progress > 0.20:
                visibility = _level_for_target(5, config)
            return _negotiation_action(
                visibility=visibility,
                request_count=_level_for_target(100, config),
                offer_reciprocal=True,
                accept_probability=0.75,
            )
        if progress < 0.65:
            return _negotiation_action(
                visibility=0 if high_score else _level_for_target(5, config),
                request_count=_level_for_target(20, config),
                offer_reciprocal=not high_score,
                accept_probability=0.45 if high_score else 0.70,
            )
        if progress < 0.90:
            return _negotiation_action(
                visibility=0,
                request_count=_level_for_target(5, config),
                offer_reciprocal=False,
                accept_probability=0.30,
            )
        return _negotiation_action(visibility=0, request_count=0, accept_probability=0.20)

    def update_position(
        self,
        observation: PeakObservation,
        rng: np.random.Generator,
        config: PeakGameConfig,
    ) -> np.ndarray:
        self.remember(
            observation.own_position,
            observation.own_score,
            observation.observed_positions,
            observation.observed_scores,
            rng,
        )
        positions, scores_seen = self.memory_arrays(config)
        progress = observation.round_index / max(1, config.rounds - 1)
        anchors = self.top_score_anchors(config, max_count=90)
        candidates = _candidate_pool(
            rng,
            config,
            observation.own_position,
            anchors,
            count=340,
            progress=progress,
        )

        length_scale = 28.0 if progress < 0.50 else 18.0
        score_hat = _surrogate_score(candidates, positions, scores_seen, length_scale=length_scale)
        local_reference = observation.observed_positions
        if local_reference.size == 0 and positions.size:
            top_score_count = min(80, len(scores_seen))
            local_reference = positions[np.argsort(scores_seen)[-top_score_count:]]

        mean_distance, nearest_distance = _distance_terms(candidates, local_reference, config)
        crowd_bonus = 0.25 * mean_distance + 0.75 * nearest_distance
        crowd_penalty = np.maximum(0.0, 16.0 - nearest_distance)
        estimated_scores = score_hat + 0.70 * crowd_bonus - 1.10 * crowd_penalty

        score_threshold = self.score_quantile(0.40, default=float("inf"))
        if observation.own_score < score_threshold:
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


@dataclass
class LLMBlackBoxStrategy(Strategy):
    """OpenAI-backed black-box agent.

    The LLM only sees total scores and positions. It does not receive hidden
    value, diversity, origin, peak, or scoring-formula information.
    """

    name = "llm_blackbox"
    model: str | None = None
    incentive: str = "competitive"
    history_limit: int = 12
    fail_open: bool = False
    pending_action: CollaborationAction = field(
        default_factory=lambda: _negotiation_action(
            visibility=1,
            request_count=5,
            offer_reciprocal=True,
            accept_probability=0.6,
        )
    )
    history: list[tuple[np.ndarray, float]] = field(default_factory=list)

    def choose_collaboration(
        self,
        round_index: int,
        position: np.ndarray,
        score: float,
        config: PeakGameConfig,
        rng: np.random.Generator,
    ) -> CollaborationAction:
        return self.pending_action

    def update_position(
        self,
        observation: PeakObservation,
        rng: np.random.Generator,
        config: PeakGameConfig,
    ) -> np.ndarray:
        self.history.append((observation.own_position.copy(), float(observation.own_score)))
        self.history = self.history[-self.history_limit :]
        prompt = build_llm_prompt(
            observation,
            config,
            self.history,
            incentive=self.incentive,
        )

        try:
            output = call_openai_llm(prompt=prompt, model=self.model)
            decision = parse_llm_decision(output, config)
        except Exception:
            if not self.fail_open:
                raise
            decision = fallback_decision(observation, rng, config)

        self.pending_action = decision_to_action(decision)
        return decision.position


@dataclass
class LLMCooperativeStrategy(LLMBlackBoxStrategy):
    name = "llm_cooperative"
    incentive: str = "cooperative"


@dataclass
class LLMCompetitiveStrategy(LLMBlackBoxStrategy):
    name = "llm_competitive"
    incentive: str = "competitive"


StrategyFactory = Callable[[], Strategy]


_STRATEGIES: dict[str, StrategyFactory] = {
    "random": RandomStrategy,
    "random_corner": RandomCornerStrategy,
    "origin_maximizer": OriginMaximizerStrategy,
    "independent_search": IndependentSearchStrategy,
    "score_following": ScoreFollowingStrategy,
    "diversity_only": DiversityOnlyStrategy,
    "full_collaboration": FullCollaborationStrategy,
    "random_collaboration": RandomCollaborationStrategy,
    "strategic_collaboration": StrategicCollaborationStrategy,
    "llm_blackbox": LLMBlackBoxStrategy,
    "llm_cooperative": LLMCooperativeStrategy,
    "llm_competitive": LLMCompetitiveStrategy,
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

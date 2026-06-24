from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Level = int | Literal["all"]
DEPRECATED_DIVERSITY_WEIGHT: float = 0.0


@dataclass(frozen=True)
class PeakGameConfig:
    """Static configuration for Peak-Divergence Game.

    The benchmark uses a value-only score, S_i = V_i. Diversity and origin
    distance are logged as diagnostics only.
    """

    num_agents: int = 200
    dimensions: int = 10
    rounds: int = 14
    num_peaks: int = 12
    # Deprecated compatibility field. The score is value-only and ignores beta.
    beta_diversity: float = DEPRECATED_DIVERSITY_WEIGHT
    gamma_origin: float = 0.0
    lower: float = 0.0
    upper: float = 100.0
    peak_height_range: tuple[float, float] = (70.0, 120.0)
    peak_width_range: tuple[float, float] = (28.0, 48.0)
    min_peak_l2_distance: float = 45.0
    discovery_value_fraction: float = 0.25
    observation_noise: float = 0.0
    delayed_observation: bool = False
    parallel_agent_updates: bool = True
    max_parallel_agent_updates: int | None = None
    communication_levels: tuple[Level, ...] = (0, 1, 5, 20, 100, "all")
    share_read_levels: tuple[Level, ...] = (0, 1, 5, 20, 100, "all")


@dataclass(frozen=True)
class PeakLandscape:
    centers: np.ndarray
    heights: np.ndarray
    widths: np.ndarray


@dataclass(frozen=True)
class CollaborationAction:
    """Negotiated communication action for one round.

    The old implementation used share/read. The current benchmark follows the
    proposal's negotiation process: initial visibility, peer requests,
    reciprocal offers, and accept/reject decisions.
    """

    visibility: Level
    request_count: Level
    offer_reciprocal: bool = False
    accept_probability: float = 0.5

    @property
    def share(self) -> Level:
        return self.visibility

    @property
    def read(self) -> Level:
        return self.request_count


@dataclass(frozen=True)
class NegotiationExchange:
    round_index: int
    requester_id: int
    target_id: int
    reciprocal_offer: bool
    accepted: bool


@dataclass(frozen=True)
class PeakObservation:
    round_index: int
    agent_id: int
    own_position: np.ndarray
    own_score: float
    observed_ids: np.ndarray
    observed_positions: np.ndarray
    observed_scores: np.ndarray


@dataclass
class PeakRunResult:
    config: PeakGameConfig
    landscape: PeakLandscape
    seed: int
    positions: np.ndarray
    final_scores: np.ndarray
    final_values: np.ndarray
    final_diversity: np.ndarray
    final_origin: np.ndarray
    final_peak_ids: np.ndarray
    position_history: list[np.ndarray] = field(default_factory=list)
    action_history: list[list[CollaborationAction]] = field(default_factory=list)
    negotiation_history: list[list[NegotiationExchange]] = field(default_factory=list)
    communication_history: list[list[dict[str, Any]]] = field(default_factory=list)
    observed_count_history: list[np.ndarray] = field(default_factory=list)
    round_metrics: list[dict[str, Any]] = field(default_factory=list)
    agent_records: list[dict[str, Any]] = field(default_factory=list)

    def final_summary(self) -> dict[str, float]:
        final = self.round_metrics[-1] if self.round_metrics else {}
        return {
            "mean_score": float(np.mean(self.final_scores)),
            "std_score": float(np.std(self.final_scores)),
            "best_score": float(np.max(self.final_scores)),
            "worst_score": float(np.min(self.final_scores)),
            "total_score": float(final.get("total_score", np.sum(self.final_scores))),
            "score_upper_bound": float(final.get("score_upper_bound", 0.0)),
            "system_optimization": float(final.get("system_optimization", 0.0)),
            "global_optimum": float(final.get("global_optimum", 0.0)),
            "best_value_ratio": float(final.get("best_value_ratio", 0.0)),
            "optimality_gap": float(final.get("optimality_gap", 0.0)),
            "mean_value": float(np.mean(self.final_values)),
            "best_value": float(np.max(self.final_values)),
            "mean_diversity": float(np.mean(self.final_diversity)),
            "mean_origin": float(np.mean(self.final_origin)),
            "mean_pairwise_distance": float(final.get("mean_pairwise_distance", 0.0)),
            "peak_coverage": float(final.get("peak_coverage", 0.0)),
            "max_peak_occupancy": float(final.get("max_peak_occupancy", 0.0)),
            "peak_entropy": float(final.get("peak_entropy", 0.0)),
        }

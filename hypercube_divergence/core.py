from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Level = int | Literal["all"]


@dataclass(frozen=True)
class GameConfig:
    """Static game configuration.

    The default game is the direct vector version proposed in the benchmark:
    10 dimensions, coordinates in [0, 100], normalized L1 distance, and an
    origin bonus weighted by lambda_origin.
    """

    num_agents: int = 200
    dimensions: int = 10
    rounds: int = 12
    lambda_origin: float = 0.35
    lower: float = 0.0
    upper: float = 100.0
    observation_noise: float = 0.0
    delayed_observation: bool = False
    share_read_levels: tuple[Level, ...] = (0, 5, 20, 100, "all")


@dataclass(frozen=True)
class CollaborationAction:
    share: Level
    read: Level


@dataclass(frozen=True)
class Observation:
    round_index: int
    agent_id: int
    own_position: np.ndarray
    observed_ids: np.ndarray
    observed_positions: np.ndarray
    previous_metrics: dict[str, float] | None = None


@dataclass
class RunResult:
    config: GameConfig
    seed: int
    positions: np.ndarray
    final_scores: np.ndarray
    final_diversity: np.ndarray
    final_origin: np.ndarray
    round_metrics: list[dict[str, Any]] = field(default_factory=list)
    agent_records: list[dict[str, Any]] = field(default_factory=list)

    def final_summary(self) -> dict[str, float]:
        final = self.round_metrics[-1] if self.round_metrics else {}
        return {
            "mean_score": float(np.mean(self.final_scores)),
            "std_score": float(np.std(self.final_scores)),
            "best_score": float(np.max(self.final_scores)),
            "worst_score": float(np.min(self.final_scores)),
            "mean_diversity": float(np.mean(self.final_diversity)),
            "mean_origin": float(np.mean(self.final_origin)),
            "corner_coverage": float(final.get("corner_coverage", 0.0)),
            "max_corner_occupancy": float(final.get("max_corner_occupancy", 0.0)),
            "corner_entropy": float(final.get("corner_entropy", 0.0)),
            "mean_pairwise_distance": float(final.get("mean_pairwise_distance", 0.0)),
        }

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Level = int | Literal["all"]


@dataclass(frozen=True)
class PeakGameConfig:
    """Static configuration for Peak-Divergence Game.

    beta_diversity and gamma_origin are intentionally small by default because
    diversity and origin distances live on a 0-100 scale while value is the
    multiplicative quality term.
    """

    num_agents: int = 200
    dimensions: int = 10
    rounds: int = 14
    num_peaks: int = 12
    beta_diversity: float = 0.015
    gamma_origin: float = 0.010
    lower: float = 0.0
    upper: float = 100.0
    peak_height_range: tuple[float, float] = (70.0, 120.0)
    peak_width_range: tuple[float, float] = (28.0, 48.0)
    min_peak_l2_distance: float = 45.0
    discovery_value_fraction: float = 0.25
    observation_noise: float = 0.0
    delayed_observation: bool = False
    share_read_levels: tuple[Level, ...] = (0, 5, 20, 100, "all")


@dataclass(frozen=True)
class PeakLandscape:
    centers: np.ndarray
    heights: np.ndarray
    widths: np.ndarray


@dataclass(frozen=True)
class CollaborationAction:
    share: Level
    read: Level


@dataclass(frozen=True)
class PeakObservation:
    round_index: int
    agent_id: int
    own_position: np.ndarray
    own_score: float
    own_value: float
    own_diversity: float
    own_origin: float
    observed_ids: np.ndarray
    observed_positions: np.ndarray
    observed_scores: np.ndarray
    observed_values: np.ndarray
    previous_metrics: dict[str, float] | None = None


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
    round_metrics: list[dict[str, Any]] = field(default_factory=list)
    agent_records: list[dict[str, Any]] = field(default_factory=list)

    def final_summary(self) -> dict[str, float]:
        final = self.round_metrics[-1] if self.round_metrics else {}
        return {
            "mean_score": float(np.mean(self.final_scores)),
            "std_score": float(np.std(self.final_scores)),
            "best_score": float(np.max(self.final_scores)),
            "worst_score": float(np.min(self.final_scores)),
            "mean_value": float(np.mean(self.final_values)),
            "best_value": float(np.max(self.final_values)),
            "mean_diversity": float(np.mean(self.final_diversity)),
            "mean_origin": float(np.mean(self.final_origin)),
            "mean_pairwise_distance": float(final.get("mean_pairwise_distance", 0.0)),
            "peak_coverage": float(final.get("peak_coverage", 0.0)),
            "max_peak_occupancy": float(final.get("max_peak_occupancy", 0.0)),
            "peak_entropy": float(final.get("peak_entropy", 0.0)),
        }

"""Peak-Divergence Game benchmark package."""

from .core import (
    CollaborationAction,
    PeakGameConfig,
    PeakLandscape,
    PeakObservation,
    PeakRunResult,
)
from .game import generate_landscape, run_game, score_positions, summarize_positions
from .strategies import available_strategies, make_population, make_strategy

__all__ = [
    "CollaborationAction",
    "PeakGameConfig",
    "PeakLandscape",
    "PeakObservation",
    "PeakRunResult",
    "available_strategies",
    "generate_landscape",
    "make_population",
    "make_strategy",
    "run_game",
    "score_positions",
    "summarize_positions",
]

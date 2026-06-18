from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .core import CollaborationAction, Level, PeakGameConfig, PeakObservation


ALLOWED_LEVELS: tuple[Level, ...] = (0, 5, 20, 100, "all")


@dataclass(frozen=True)
class LLMDecision:
    position: np.ndarray
    next_share: Level
    next_read: Level


def parse_llm_decision(text: str, config: PeakGameConfig) -> LLMDecision:
    """Parse a model JSON response into a clipped game action."""

    payload = json.loads(_extract_json_object(text))
    raw_position = payload.get("position")
    if not isinstance(raw_position, list) or len(raw_position) != config.dimensions:
        raise ValueError(
            f"Expected position to be a list of {config.dimensions} numbers, got {raw_position!r}"
        )
    position = np.array([float(value) for value in raw_position], dtype=float)
    position = np.clip(position, config.lower, config.upper)
    return LLMDecision(
        position=position,
        next_share=_parse_level(payload.get("next_share", 0)),
        next_read=_parse_level(payload.get("next_read", 0)),
    )


def call_openai_llm(
    *,
    prompt: str,
    model: str | None = None,
) -> str:
    """Call OpenAI Responses API and return output text.

    Requires `pip install openai` and `OPENAI_API_KEY` in the environment.
    The OpenAI SDK reads `OPENAI_API_KEY` automatically.
    """

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY='your_api_key'"
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI SDK is not installed. Run: pip install openai") from exc

    client = OpenAI()
    response = client.responses.create(
        model=model or os.getenv("OPENAI_AGENT_MODEL", "gpt-5.5"),
        input=prompt,
    )
    return response.output_text


def build_llm_prompt(
    observation: PeakObservation,
    config: PeakGameConfig,
    history: list[tuple[np.ndarray, float]],
    max_history: int = 8,
    max_peers: int = 12,
) -> str:
    """Build the black-box game prompt shown to the LLM agent."""

    history_rows = [
        {
            "position": _round_vector(position),
            "score": round(float(score), 4),
        }
        for position, score in history[-max_history:]
    ]
    peer_order = np.argsort(observation.observed_scores)[::-1]
    peer_rows: list[dict[str, Any]] = []
    for idx in peer_order[:max_peers]:
        peer_rows.append(
            {
                "peer_id": int(observation.observed_ids[idx]),
                "position": _round_vector(observation.observed_positions[idx]),
                "score": round(float(observation.observed_scores[idx]), 4),
            }
        )

    example_position = [0 for _ in range(config.dimensions)]
    return f"""
You are one agent in a black-box multi-agent search game.

Known facts:
- The action space is {config.dimensions}-dimensional.
- Every coordinate must be between {config.lower} and {config.upper}.
- Your objective is to maximize your own future total score.
- You do not know the scoring formula.
- You do not know hidden peak locations or reward components.
- Collaboration can reveal useful high-score examples, but copying visible successes may cause crowding.

Current round: {observation.round_index}
Your current position: {_round_vector(observation.own_position)}
Your current total score: {round(float(observation.own_score), 4)}

Your recent history:
{json.dumps(history_rows, separators=(",", ":"))}

Observed peers this round, sorted by peer score:
{json.dumps(peer_rows, separators=(",", ":"))}

Choose:
1. your next {config.dimensions}D position
2. how many peers can observe you next round: next_share in [0, 5, 20, 100, "all"]
3. how many visible peers you want to read next round: next_read in [0, 5, 20, 100, "all"]

Return only valid JSON with this exact shape:
{{"position":{json.dumps(example_position)},"next_share":0,"next_read":20}}
""".strip()


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text!r}")
    return match.group(0)


def _parse_level(value: Any) -> Level:
    if value == "all":
        return "all"
    if isinstance(value, str) and value.strip().lower() == "all":
        return "all"
    numeric = int(value)
    if numeric not in (0, 5, 20, 100):
        raise ValueError(f"Invalid collaboration level {value!r}")
    return numeric


def _round_vector(position: np.ndarray) -> list[float]:
    return [round(float(value), 2) for value in position.tolist()]


def fallback_decision(
    observation: PeakObservation,
    rng: np.random.Generator,
    config: PeakGameConfig,
) -> LLMDecision:
    """Cheap fallback used when model output is malformed."""

    if observation.observed_positions.size and observation.observed_scores.size:
        best = int(np.argmax(observation.observed_scores))
        center = observation.observed_positions[best]
        scale = max(4.0, 18.0 * (1.0 - observation.round_index / max(1, config.rounds - 1)))
        position = center + rng.normal(0.0, scale, size=config.dimensions)
    else:
        position = rng.uniform(config.lower, config.upper, size=config.dimensions)
    return LLMDecision(
        position=np.clip(position, config.lower, config.upper),
        next_share=5,
        next_read=20,
    )


def decision_to_action(decision: LLMDecision) -> CollaborationAction:
    return CollaborationAction(share=decision.next_share, read=decision.next_read)

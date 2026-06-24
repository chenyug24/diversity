from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .core import CollaborationAction, Level, PeakGameConfig, PeakObservation


ALLOWED_LEVELS: tuple[Level, ...] = (0, 1, 5, 20, 100, "all")


@dataclass(frozen=True)
class LLMDecision:
    position: np.ndarray
    next_visibility: Level
    next_request_count: Level
    offer_reciprocal: bool
    accept_probability: float


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
        next_visibility=_parse_level(payload.get("next_visibility", payload.get("next_share", 0))),
        next_request_count=_parse_level(
            payload.get("next_request_count", payload.get("next_read", 0))
        ),
        offer_reciprocal=bool(payload.get("offer_reciprocal", False)),
        accept_probability=float(
            np.clip(float(payload.get("accept_probability", 0.5)), 0.0, 1.0)
        ),
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

    load_local_env()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it or create a local .env file from .env.example."
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


def load_local_env(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file if env vars are absent."""

    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_llm_prompt(
    observation: PeakObservation,
    config: PeakGameConfig,
    history: list[tuple[np.ndarray, float]],
    incentive: str = "competitive",
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
    objective_text = _objective_text(incentive)
    return f"""
You are one agent in a black-box multi-agent search game.

Known facts:
- The action space is {config.dimensions}-dimensional.
- Every coordinate must be between {config.lower} and {config.upper}.
- {objective_text}
- You do not know the scoring formula.
- You do not know hidden peak locations or reward components.
- Communication has a negotiation phase. Agents choose initial visibility, request peer information, offer reciprocal exchange, and accept or reject incoming requests.
- Available peer examples came from initial visibility or accepted information exchanges.

Current round: {observation.round_index}
Your current position: {_round_vector(observation.own_position)}
Your current total score: {round(float(observation.own_score), 4)}

Your recent history:
{json.dumps(history_rows, separators=(",", ":"))}

Observed peers this round, sorted by peer score:
{json.dumps(peer_rows, separators=(",", ":"))}

Choose:
1. your next {config.dimensions}D position
2. how many peers can initially see your information next round: next_visibility in [0, 1, 5, 20, 100, "all"]
3. how many peers you will request information from next round: next_request_count in [0, 1, 5, 20, 100, "all"]
4. whether your requests offer reciprocal exchange: offer_reciprocal as true or false
5. probability that you accept incoming requests next round: accept_probability between 0 and 1

Return only valid JSON with this exact shape:
{{"position":{json.dumps(example_position)},"next_visibility":1,"next_request_count":5,"offer_reciprocal":true,"accept_probability":0.6}}
""".strip()


def _objective_text(incentive: str) -> str:
    if incentive == "cooperative":
        return (
            "Cooperative objective: help the whole group find the highest possible "
            "future total score; useful information sharing may improve the system outcome."
        )
    if incentive == "competitive":
        return (
            "Competitive objective: maximize your own future total score and try to "
            "outperform the other agents."
        )
    return "Your objective is to maximize future total score."


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
    if numeric not in (0, 1, 5, 20, 100):
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
        next_visibility=1,
        next_request_count=5,
        offer_reciprocal=True,
        accept_probability=0.6,
    )


def decision_to_action(decision: LLMDecision) -> CollaborationAction:
    return CollaborationAction(
        visibility=decision.next_visibility,
        request_count=decision.next_request_count,
        offer_reciprocal=decision.offer_reciprocal,
        accept_probability=decision.accept_probability,
    )

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .core import CollaborationAction, Level, PeakGameConfig, PeakObservation


@dataclass(frozen=True)
class LLMDecision:
    position: np.ndarray
    next_visibility: Level
    next_request_count: Level
    offer_reciprocal: bool
    accept_probability: float


@dataclass(frozen=True)
class PublicationDecision:
    position: np.ndarray
    publish: bool


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


def parse_publication_decision(text: str, config: PeakGameConfig) -> PublicationDecision:
    payload = json.loads(_extract_json_object(text))
    raw_position = payload.get("position")
    if not isinstance(raw_position, list) or len(raw_position) != config.dimensions:
        raise ValueError(
            f"Expected position to be a list of {config.dimensions} numbers, got {raw_position!r}"
        )
    position = np.array([float(value) for value in raw_position], dtype=float)
    return PublicationDecision(
        position=np.clip(position, config.lower, config.upper),
        publish=bool(payload.get("publish", False)),
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
    communication_feedback = _sanitize_feedback(
        getattr(observation, "communication_feedback", {})
    )
    return f"""
You are one agent in a data-feedback multi-agent search game.

Known facts:
- The action space is {config.dimensions}-dimensional.
- Every coordinate must be between {config.lower} and {config.upper}.
- {objective_text}
- You do not know the scoring formula.
- You do not know hidden peak locations or reward components.
- The environment only gives numerical feedback: your own score, your own history, peer position/score examples revealed through communication, and communication outcomes.
- Communication has a negotiation phase. Agents choose initial visibility, request peer information, offer reciprocal exchange, and accept or reject incoming requests.
- Available peer examples came only from initial visibility or accepted information exchanges.

Current round: {observation.round_index}
Your current position: {_round_vector(observation.own_position)}
Your current total score: {round(float(observation.own_score), 4)}

Communication feedback from this round:
{json.dumps(communication_feedback, separators=(",", ":"))}

Your recent history:
{json.dumps(history_rows, separators=(",", ":"))}

Observed peers this round, sorted by peer score:
{json.dumps(peer_rows, separators=(",", ":"))}

Choose:
1. your next {config.dimensions}D position
2. how many peers can initially see your information next round: next_visibility as any non-negative integer or "all"
3. how many peers you will request information from next round: next_request_count as any non-negative integer or "all"
4. whether your requests offer reciprocal exchange: offer_reciprocal as true or false
5. probability that you accept incoming requests next round: accept_probability between 0 and 1

Return only valid JSON with this exact shape:
{{"position":{json.dumps(example_position)},"next_visibility":1,"next_request_count":5,"offer_reciprocal":true,"accept_probability":0.6}}
""".strip()


def build_publication_prompt(
    observation: PeakObservation,
    config: PeakGameConfig,
    history: list[tuple[np.ndarray, float]],
    incentive: str = "research",
    max_history: int = 8,
    max_public_records: int = 18,
) -> str:
    history_rows = [
        {
            "position": _round_vector(position),
            "score": round(float(score), 4),
        }
        for position, score in history[-max_history:]
    ]
    public_order = np.argsort(observation.observed_scores)[::-1]
    public_rows: list[dict[str, Any]] = []
    for idx in public_order[:max_public_records]:
        public_rows.append(
            {
                "published_by_agent": int(observation.observed_ids[idx]),
                "location": _round_vector_precise(observation.observed_positions[idx]),
                "value": round(float(observation.observed_scores[idx]), 4),
            }
        )
    feedback = _sanitize_feedback(getattr(observation, "communication_feedback", {}))
    example_position = [0 for _ in range(config.dimensions)]
    objective = _publication_objective_text(incentive)
    return f"""
You are one research agent in a public research publication game.

Known facts:
- The research space is {config.dimensions}-dimensional and continuous.
- Every coordinate must be between {config.lower} and {config.upper}.
- {objective}
- You do not know the scoring formula or hidden landscape.
- You receive your own true value after trying a location.
- Publishing is optional. If you publish, your current location and true value become visible to all agents.
- There is no private exchange, no private messaging, and no selective sharing.
- A location already in the public registry cannot be submitted again exactly.
- Because the space is continuous, nearby locations are allowed; only exact reuse of a published location is blocked.

Current round: {observation.round_index}
Your current location: {_round_vector_precise(observation.own_position)}
Your current true value: {round(float(observation.own_score), 4)}

Your recent private history:
{json.dumps(history_rows, separators=(",", ":"))}

Public registry records visible before this decision, sorted by value:
{json.dumps(public_rows, separators=(",", ":"))}

Feedback:
{json.dumps(feedback, separators=(",", ":"))}

Choose:
1. publish: whether to publish your current location and value to everyone
2. position: your next {config.dimensions}D location; it must not exactly match any public registry location

Return only valid JSON with this exact shape:
{{"publish":true,"position":{json.dumps(example_position)}}}
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


def _publication_objective_text(incentive: str) -> str:
    if incentive == "cooperative":
        return (
            "Cooperative objective: help the research group discover high-value "
            "research directions while avoiding exact duplication of published work."
        )
    if incentive == "competitive":
        return (
            "Competitive objective: maximize your own future research value and "
            "avoid giving competitors unnecessary help."
        )
    return (
        "Research objective: discover high-value research directions and decide "
        "when public publication is useful."
    )


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
    if numeric < 0:
        raise ValueError(f"Collaboration level must be non-negative, got {value!r}")
    return numeric


def _round_vector(position: np.ndarray) -> list[float]:
    return [round(float(value), 2) for value in position.tolist()]


def _sanitize_feedback(feedback: dict[str, Any]) -> dict[str, Any]:
    """Convert communication feedback to compact JSON-safe values for prompts."""

    compact_keys = [
        "initial_visible_count",
        "initially_visible_to_count",
        "requests_sent",
        "requests_received",
        "accepted_requests_sent",
        "accepted_requests_received",
        "rejected_requests_sent",
        "rejected_requests_received",
        "reciprocal_offers_sent",
        "reciprocal_exchanges",
        "observed_count",
        "initial_visible_source_ids",
        "requested_target_ids",
        "accepted_target_ids",
        "rejected_target_ids",
        "incoming_requester_ids",
        "accepted_incoming_requester_ids",
        "rejected_incoming_requester_ids",
        "observed_agent_ids",
        "communication_mode",
        "public_registry_size",
        "public_record_rounds",
        "private_exchange_allowed",
        "published_locations_blocked",
        "blocked_resubmissions",
        "blocked_position",
        "blocked_reason",
    ]
    sanitized: dict[str, Any] = {}
    for key in compact_keys:
        if key not in feedback:
            continue
        value = feedback[key]
        if isinstance(value, np.ndarray):
            sanitized[key] = value.astype(int).tolist()
        elif isinstance(value, list):
            sanitized[key] = [int(item) if isinstance(item, (np.integer, int)) else item for item in value]
        elif isinstance(value, (np.integer, int)):
            sanitized[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            sanitized[key] = round(float(value), 4)
        else:
            sanitized[key] = value
    return sanitized


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


def fallback_publication_decision(
    observation: PeakObservation,
    rng: np.random.Generator,
    config: PeakGameConfig,
) -> PublicationDecision:
    if observation.observed_positions.size and observation.observed_scores.size:
        best = int(np.argmax(observation.observed_scores))
        center = observation.observed_positions[best]
        scale = max(3.0, 20.0 * (1.0 - observation.round_index / max(1, config.rounds - 1)))
        position = center + rng.normal(0.0, scale, size=config.dimensions)
        public_median = float(np.median(observation.observed_scores))
        publish = float(observation.own_score) >= public_median
    else:
        position = rng.uniform(config.lower, config.upper, size=config.dimensions)
        publish = True
    return PublicationDecision(
        position=np.clip(position, config.lower, config.upper),
        publish=publish,
    )


def _round_vector_precise(position: np.ndarray) -> list[float]:
    return [round(float(value), 6) for value in position.tolist()]


def decision_to_action(decision: LLMDecision) -> CollaborationAction:
    return CollaborationAction(
        visibility=decision.next_visibility,
        request_count=decision.next_request_count,
        offer_reciprocal=decision.offer_reciprocal,
        accept_probability=decision.accept_probability,
    )

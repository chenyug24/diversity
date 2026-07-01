# Black-Box Peak-Divergence Game

A multi-agent benchmark for testing whether communication and incentive structure
help agents reach the global optimum in a hidden black-box landscape.

Agents search in a bounded 10-dimensional space with hidden Gaussian-shaped
peaks. The system computes a private value score, but agents only see
their own total score. Through optional negotiated information exchange, agents
may observe other agents' positions and total scores. They never observe hidden peak
locations or the scoring formula.

## Game

Each agent chooses a point:

```text
z_i in [0, 100]^10
```

The system secretly generates `K` hidden Gaussian-shaped peaks:

```text
V(z) = max_m h_m * exp(-||z - mu_m||_2^2 / (2 * sigma_m^2))
```

The system evaluates each final point with:

```text
S_i = V_i
```

where `V_i` is hidden value. Diversity and origin distance are logged only as
diagnostic metrics; they are not rewarded.

The primary optimization metric is the highest value found by any agent at any
round:

```text
best_value_found = max_{i,t} V(z_i,t)
best_value_found_ratio = best_value_found / max_peak_height
```

The reports also include a system-level average metric:

```text
system_optimization = mean_score / max_peak_height
```

This average metric is diagnostic; it says how well the whole population is
doing, not whether the system discovered the best solution.

## Black-Box Information Setting

Agents know:

- The action space is 10-dimensional and bounded in `[0, 100]`.
- Depending on the strategy, they may be prompted to optimize cooperatively or competitively.
- They can choose initial visibility, request peer information, offer reciprocal exchange,
  and accept or reject incoming requests.
- They receive their own total score after each round.

Agents do not know:

- The hidden peak locations, heights, or widths.
- The scoring formula.
- The hidden value component of the score.
- The population-level metrics.

When collaboration reveals a peer, the observing agent sees only:

```text
(peer_position, peer_total_score)
```

This makes collaboration a source of uncertain information. Reading high-scoring
peers can reveal useful regions, but it may also cause imitation and crowding.
Sharing a good location can help others learn, but may attract competitors.

## Task 1: Public Research Publication

The simplest research scenario removes private exchange entirely. Agents act like
researchers exploring a continuous hidden opportunity landscape.

```text
1. Each agent submits a research location.
2. The environment returns that agent's true score.
3. The agent may optionally publish its current location and true score.
4. Published records enter a public registry visible to all agents in later decisions.
5. A location already in the public registry cannot be submitted again exactly.
6. No private messages, private exchanges, requests, or accept/reject negotiation exist.
```

This first task asks whether public publication helps agents learn useful
directions, or whether it mainly creates imitation around already published
successes. The space is continuous, so nearby locations are allowed; only exact
reuse of a public location is blocked.

Run it with:

```bash
python3 scripts/run_publication_case.py \
  --strategy score_following \
  --agents 6 \
  --rounds 20 \
  --peaks 2 \
  --dimensions 2 \
  --out results/publication_case
```

For API-backed agents, use `--strategy llm_public_research`,
`llm_public_cooperative`, or `llm_public_competitive`.

## Negotiated Communication

At every round, the benchmark follows the proposal's communication process:

```text
1. Each agent submits a point.
2. The environment returns each agent's own score.
3. Each agent chooses an initial visibility level.
4. Agents may request information from selected peers and offer reciprocal exchange.
5. Target agents accept or reject incoming requests.
6. The environment reveals initially visible and accepted exchanged information.
7. Each agent chooses the next point.
```

The implemented communication levels are:

```text
visibility_i    in nonnegative integers, or all
request_count_i in nonnegative integers, or all
```

Values larger than the number of available peers are capped at all available
peers.

The evaluator records visibility, requests, reciprocal offers, accept/reject
outcomes, and the final number of peer records observed by each agent. The
agent also receives a compact communication-feedback record for the current
round, including which requests were accepted or rejected and which peer records
were actually observed. This keeps the first case simple: agents only adapt from
numeric feedback and communication outcomes.

Agent updates are synchronous within a round. The environment builds all
observations from the same previous population state, asks all agents for their
next positions, and only then updates the population. For LLM-backed agents, the
API requests for one round are dispatched in parallel by default.

## Implemented Strategies

- `random`: samples a fresh random point each round.
- `random_corner`: chooses one random hypercube corner.
- `origin_maximizer`: always chooses `(100, ..., 100)`.
- `independent_search`: never communicates; locally searches from its own score history.
- `score_following`: makes information visible, requests peers, and follows the highest observed total score.
- `diversity_only`: spreads away from observed agents while ignoring score.
- `full_collaboration`: all agents make information visible, accept exchanges, and converge toward the highest observed total score.
- `random_collaboration`: randomly chooses negotiation actions and builds a local score surrogate.
- `strategic_collaboration`: requests more when its own score is weak, reveals less when its score is strong, and balances high-score imitation against crowd avoidance.
- `llm_blackbox` / `llm_competitive`: calls the OpenAI API with a competitive objective.
- `llm_cooperative`: calls the OpenAI API with a cooperative group-optimization objective.

## Quick Start

```bash
python3 -m unittest discover

python3 scripts/run_experiment.py \
  --agents 200 \
  --rounds 14 \
  --peaks 12 \
  --seeds 10 \
  --out results/default
```

Outputs:

- `results/default/final_metrics.csv`: one row per strategy and seed.
- `results/default/round_metrics.csv`: round-by-round diagnostic metrics.
- `results/default/peaks.csv`: hidden peak metadata for analysis.
- `results/default/summary.md`: ranked strategy summary.

Run the simplest feedback-only communication case:

```bash
python3 scripts/run_feedback_communication_case.py \
  --strategy strategic_collaboration \
  --agents 6 \
  --rounds 20 \
  --peaks 2 \
  --dimensions 2 \
  --out results/feedback_case
```

This exports `agent_round_feedback.csv`, `observed_peer_examples.csv`,
`negotiation_edges.csv`, `metrics_by_round.csv`, and two communication plots.
Use `--strategy llm_cooperative` or `--strategy llm_competitive` when you want
the same first case with API-backed agents.

Generate figures from an experiment directory:

```bash
python3 scripts/plot_results.py \
  --results results/default \
  --out results/default/figures
```

## Running an OpenAI Agent

Install dependencies and set your API key:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .

cp .env.example .env
```

Then edit `.env` and replace `sk-your-real-key-here` with your real OpenAI API
key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys).
Do not commit `.env`; it is ignored by git.

Run a small LLM-backed experiment first:

```bash
python3 scripts/run_experiment.py \
  --agents 3 \
  --rounds 2 \
  --peaks 3 \
  --seeds 1 \
  --strategies llm_blackbox \
  --out results/llm_demo
```

Each `llm_blackbox` agent calls the OpenAI API once per round. Calls within the
same round run in parallel by default, but large runs still produce many total
API calls, so start small before scaling up. Use `--max-parallel-agent-updates`
to limit concurrent requests if needed.

## Recommended Experiments

First test whether fixed collaboration creates peak convergence:

```bash
python3 scripts/run_experiment.py \
  --agents 200 \
  --rounds 14 \
  --peaks 12 \
  --seeds 30 \
  --write-agent-scores \
  --out results/main
```

Then sweep population size and number of peaks with the value-only score:

```bash
python3 scripts/run_sweep.py \
  --agents 100 200 400 \
  --peaks 6 12 24 \
  --seeds 20 \
  --out results/sweep
```

The central hypothesis is:

```text
Collaboration helps agents discover high-scoring regions,
but excessive collaboration causes imitation and overcrowding.
Strategic visibility and inspection should improve global optimization while
avoiding redundant search.
```

See [PROPOSAL.md](PROPOSAL.md) for the full project proposal.

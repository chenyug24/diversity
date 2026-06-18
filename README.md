# Black-Box Peak-Divergence Game

A multi-agent benchmark for testing whether collaboration helps agents discover
high-scoring but undercrowded solutions when the reward rule is not revealed.

Agents search in a bounded 10-dimensional space with hidden Gaussian-shaped
peaks. The system computes a private quality-diversity score, but agents only see
their own total score. Through optional read/share actions, agents may observe
other agents' positions and total scores. They never observe hidden peak
locations, the value component, the diversity component, or the scoring formula.

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
S_i = V_i * (1 + beta_diversity * D_i + gamma_origin * O_i)
```

where `V_i` is hidden value, `D_i` is average distance from other agents, and
`O_i` is distance from the origin. These components are used for evaluation
metrics, but they are not shown to agents.

## Black-Box Information Setting

Agents know:

- The action space is 10-dimensional and bounded in `[0, 100]`.
- They want to maximize their own total score.
- They can choose how much to read and how much to share.
- They receive their own total score after each round.

Agents do not know:

- The hidden peak locations, heights, or widths.
- The scoring formula.
- The value, diversity, or origin components of the score.
- The population-level metrics.

When collaboration reveals a peer, the observing agent sees only:

```text
(peer_position, peer_total_score)
```

This makes collaboration a source of uncertain information. Reading high-scoring
peers can reveal useful regions, but it may also cause imitation and crowding.
Sharing a good location can help others learn, but may attract competitors.

## Collaboration

At every round, each agent chooses:

```text
share_i in {0, 5, 20, 100, all}
read_i  in {0, 5, 20, 100, all}
```

Sharing controls how many other agents may observe the agent. Reading controls
how many visible peer states the agent receives.

## Implemented Strategies

- `random`: samples a fresh random point each round.
- `random_corner`: chooses one random hypercube corner.
- `origin_maximizer`: always chooses `(100, ..., 100)`.
- `independent_search`: never reads or shares; locally searches from its own score history.
- `score_following`: reads/shares all and follows the highest observed total score.
- `diversity_only`: spreads away from observed agents while ignoring score.
- `full_collaboration`: all agents read/share all and converge toward the highest observed total score.
- `random_collaboration`: randomly chooses read/share levels and builds a local score surrogate.
- `strategic_collaboration`: reads more when its own score is weak, shares less when its score is strong, and balances high-score imitation against crowd avoidance.
- `llm_blackbox`: calls the OpenAI API each round to choose a position from black-box observations.

## Quick Start

```bash
python3 -m unittest discover

python3 scripts/run_experiment.py \
  --agents 200 \
  --rounds 14 \
  --peaks 12 \
  --seeds 10 \
  --beta-diversity 0.015 \
  --gamma-origin 0.010 \
  --out results/default
```

Outputs:

- `results/default/final_metrics.csv`: one row per strategy and seed.
- `results/default/round_metrics.csv`: round-by-round diagnostic metrics.
- `results/default/peaks.csv`: hidden peak metadata for analysis.
- `results/default/summary.md`: ranked strategy summary.

Generate figures from an experiment directory:

```bash
python3 scripts/plot_results.py \
  --results results/default \
  --out results/default/figures
```

## Running an OpenAI Agent

Install dependencies and set your API key:

```bash
pip install -e .
export OPENAI_API_KEY="your_api_key"
export OPENAI_AGENT_MODEL="gpt-5.5"
```

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

Each `llm_blackbox` agent calls the OpenAI API once per round. Large runs can
produce many API calls, so start small before scaling up.

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

Then sweep population size, number of peaks, and diversity pressure:

```bash
python3 scripts/run_sweep.py \
  --agents 100 200 400 \
  --peaks 6 12 24 \
  --betas 0.0 0.015 0.05 0.15 \
  --seeds 20 \
  --out results/sweep
```

The central hypothesis is:

```text
Collaboration helps agents discover high-scoring regions,
but excessive collaboration causes imitation and overcrowding.
Selective collaboration should perform best when diversity pressure is high.
```

See [PROPOSAL.md](PROPOSAL.md) for the full project proposal.

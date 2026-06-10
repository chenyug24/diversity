# Peak-Divergence Game

A multi-agent benchmark for strategic collaboration in quality-diversity search.

Hundreds of agents search in a bounded 10-dimensional space with hidden Gaussian
peaks. A good agent must find high-value regions while avoiding overcrowded
regions discovered by other agents. Collaboration is not fixed: each agent chooses
how much to reveal and how much peer information to read at every round.

## Game

Each agent chooses a point:

```text
z_i in [0, 100]^10
```

The hidden landscape contains `K` Gaussian peaks:

```text
V(z) = max_m h_m * exp(-||z - mu_m||_2^2 / (2 * sigma_m^2))
```

The final score is:

```text
S_i = V_i * (1 + beta_diversity * D_i + gamma_origin * O_i)
```

where:

- `V_i` is hidden-landscape value.
- `D_i` is average normalized L1 distance from other agents.
- `O_i` is normalized distance from the origin.

The multiplicative score gates novelty by quality: a point that is far from
everyone but has low value still scores poorly.

## Collaboration

At every round, each agent chooses:

```text
share_i in {0, 5, 20, 100, all}
read_i  in {0, 5, 20, 100, all}
```

Sharing controls how many other agents may observe the agent. Reading controls
how many visible peer states the agent receives. Observations include peer
coordinates, value, and score, but hidden peak locations are never revealed.

## Implemented Strategies

- `random`: samples a fresh random point each round.
- `random_corner`: chooses one random hypercube corner.
- `origin_maximizer`: always chooses `(100, ..., 100)`.
- `independent_search`: uses only its own value feedback and performs local search.
- `value_only`: reads/shares all and follows the highest observed value.
- `diversity_only`: spreads away from observed agents while ignoring value.
- `full_collaboration`: reads/shares all and follows the highest observed total score.
- `random_collaboration`: randomly chooses read/share levels and uses a value surrogate.
- `strategic_collaboration`: reads more when value is low or early, shares less after finding value, and balances surrogate value against crowding.

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
- `results/default/round_metrics.csv`: round-by-round value, diversity, and peak coverage.
- `results/default/peaks.csv`: hidden peak metadata for each seed.
- `results/default/summary.md`: ranked strategy summary.

## Recommended Experiments

First test whether fixed collaboration creates peak convergence:

```bash
python3 scripts/run_experiment.py \
  --agents 200 \
  --rounds 14 \
  --peaks 12 \
  --seeds 30 \
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

The key hypothesis is that low `beta_diversity` settings mostly reward peak
finding, so value-only or full-collaboration agents may perform well by crowding
the highest peak. As diversity pressure increases, excessive collaboration should
become less attractive, and selective collaboration should achieve a better
quality-diversity tradeoff.

## Notes

The current agents are heuristic baselines, not trained neural policies. They are
designed to make the benchmark runnable and to expose the main strategic tension:
finding valuable peaks is useful, but crowded peaks become less attractive as
diversity pressure rises.

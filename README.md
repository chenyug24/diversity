# Hypercube Divergence Game

A small simulation benchmark for strategic differentiation under limited collaboration.

Agents choose points in a bounded 10-dimensional hypercube. Each final score is:

```text
score_i = average_normalized_L1_distance_to_other_agents
          + lambda_origin * normalized_L1_distance_from_origin
```

The benchmark includes read/share collaboration budgets. A source agent's `share`
level limits how many other agents can see it. A reader's `read` level limits how
many visible positions it receives.

## Quick Start

```bash
python3 -m unittest discover
python3 scripts/run_experiment.py --agents 200 --rounds 12 --seeds 10 --lambda-origin 0.35 --out results/default
```

Outputs:

- `results/default/final_metrics.csv`: one row per strategy and seed.
- `results/default/round_metrics.csv`: round-by-round diagnostics.
- `results/default/summary.json`: aggregated means and standard deviations.
- `results/default/summary.md`: a compact ranked table.

## Implemented Strategies

- `independent`: samples an initial point, reads nothing, shares nothing, and stays put.
- `origin_maximizer`: always chooses `(100, ..., 100)`.
- `corner_random`: chooses one random hypercube corner and stays there.
- `full_collaboration`: shares with all, reads all, then makes a synchronized coordinate-wise best response.
- `diversity_only`: reads/shares a medium amount and maximizes distance from observed agents without origin reward.
- `random_collaboration`: randomly chooses read/share budgets and uses stochastic candidate search.
- `strategic_collaboration`: reads a lot early, reduces reading over time, hides late, and uses memory plus candidate search to avoid observed crowding.

## Recommended First Experiment

Start with a homogeneous-population sweep:

```bash
python3 scripts/run_experiment.py \
  --agents 200 \
  --rounds 12 \
  --seeds 30 \
  --lambda-origin 0.35 \
  --out results/main
```

The key hypothesis to test first is an information-regime hypothesis:

```text
independent and origin-maximizing agents underperform,
full collaboration converges or oscillates into crowded regions,
and strategic partial collaboration achieves the best mean final score.
```

After that, sweep `lambda_origin` across `0.0, 0.25, 0.5, 0.75, 1.0` and repeat with
`--observation-noise 5` and `--delayed-observation`. Those two ablations tell you
whether the result is a real collaboration effect or just an artifact of exact,
instantaneous coordinates.

```bash
python3 scripts/run_lambda_sweep.py \
  --agents 200 \
  --rounds 12 \
  --seeds 30 \
  --lambdas 0.0 0.25 0.35 0.5 0.75 1.0 \
  --out results/lambda_sweep
```

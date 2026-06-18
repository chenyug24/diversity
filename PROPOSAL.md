# Black-Box Peak-Divergence Game: A Multi-Agent Benchmark for Strategic Collaboration Under Partial Feedback

## 1. Motivation

Many multi-agent systems assume that collaboration improves performance. However,
collaboration can also cause agents to imitate the same successful examples and
collapse into the same region. This is especially harmful in tasks where agents
must not only find good solutions, but also find solutions that remain different
from those chosen by others.

We propose **Black-Box Peak-Divergence Game**, a benchmark for studying strategic
collaboration under uncertainty. Hundreds of agents search in a hidden
10-dimensional multi-peak landscape. Agents do not know the scoring rule, the
hidden peak locations, or the value/diversity decomposition of the reward. Each
agent only observes its own total score. Through optional collaboration, agents
may observe other agents' positions and total scores.

The central research question is:

```text
Can limited collaboration help agents discover high-scoring regions without
causing the population to converge onto the same overcrowded solution?
```

This benchmark tests collaboration as information sharing under uncertainty. A
high-scoring peer may reveal a useful region, but if many agents copy the same
peer, the region becomes crowded. Agents must decide when to read, when to share,
and when to avoid visible success.

## 2. Game Setup

The game has `N` agents and proceeds for `T` rounds. Each agent submits a point
in a bounded 10-dimensional space:

```text
z_i = (z_i1, z_i2, ..., z_i10), where z_ik in [0, 100].
```

The system secretly generates `K` hidden peaks:

```text
mu_1, mu_2, ..., mu_K in [0, 100]^10.
```

Each peak has a height `h_m` and width `sigma_m`. The hidden value of a point is:

```text
V(z_i) = max_m h_m * exp(-||z_i - mu_m||_2^2 / (2 * sigma_m^2)).
```

This creates a search problem. Some peaks may be broad and easy to find; others
may be narrow but higher value.

## 3. Hidden Scoring Rule

The system computes three internal quantities:

```text
V_i = hidden landscape value
D_i = average normalized L1 distance from other agents
O_i = normalized distance from the origin
```

where:

```text
D_i = (1 / (N - 1)) * sum_{j != i} (1 / 10) * sum_k |z_ik - z_jk|
O_i = (1 / 10) * sum_k z_ik
```

The system's total score is:

```text
S_i = V_i * (1 + beta * D_i + gamma * O_i).
```

This multiplicative form prevents agents from winning through meaningless
novelty. A point that is far from everyone but has low hidden value still scores
poorly.

In the benchmark's main setting, this formula is **not revealed to agents**. The
formula is used by the environment and for evaluation, but agents only observe
their own total score and, when collaboration allows it, peer total scores.

## 4. Black-Box Information Setting

Agents know:

```text
- The action space is [0, 100]^10.
- The objective is to maximize their own total score.
- They can choose read/share collaboration levels.
- They receive their own total score after each round.
```

Agents do not know:

```text
- The hidden peak locations.
- The scoring formula.
- Whether score comes from value, diversity, origin distance, or crowding.
- Population-level diagnostics such as average diversity or peak coverage.
```

When an agent observes a peer, it receives only:

```text
(peer_position, peer_total_score).
```

This turns the task from direct mathematical optimization into strategic
collaboration under partial feedback. Agents must infer which regions are useful
from observed successes while deciding whether copying those successes will cause
overcrowding.

## 5. Collaboration Mechanism

At each round, each agent chooses two collaboration actions:

```text
share_i in {0, 5, 20, 100, all}
read_i  in {0, 5, 20, 100, all}
```

`share_i` controls how many other agents may observe the agent's current
position and total score. `read_i` controls how many visible peer states the
agent receives.

Reading more can help discover high-scoring regions, but it can also induce
imitation. Sharing can spread useful information, but it may attract competitors
to the sharer's region. Collaboration is therefore an endogenous strategic
action rather than a fixed experimental condition.

## 6. Iterative Procedure

1. The environment initializes `N` agents.
2. The hidden peaks are generated but not revealed.
3. Each agent submits an initial 10-dimensional point.
4. The environment computes hidden value, diversity, origin distance, and total score.
5. Each agent observes only its own total score.
6. Each agent chooses `share` and `read` levels.
7. The environment reveals peer positions and peer total scores according to the read/share choices.
8. Each agent updates its point using only black-box feedback and observed peer states.
9. Steps 4-8 repeat for `T` rounds.
10. Final scores and diagnostic metrics are computed by the environment.

## 7. Baselines

The benchmark compares:

```text
Random:
  Samples a fresh random point each round.

Random Corner:
  Chooses a random hypercube corner.

Origin Maximizer:
  Always chooses (100, ..., 100).

Independent Search:
  Never reads or shares; locally searches from its own score history.

Score Following:
  Reads and shares all; follows the highest observed total score.

Diversity Only:
  Uses observed positions to move away from visible agents, ignoring scores.

Full Collaboration:
  All agents read and share all information each round.

Random Collaboration:
  Randomly chooses read/share levels and builds a black-box score surrogate.

Strategic Collaboration:
  Reads more when its own score is weak, shares less when its score is strong,
  and trades off high-score imitation against crowd avoidance.
```

## 8. Evaluation Metrics

The primary metric is average final score:

```text
(1 / N) * sum_i S_i.
```

Diagnostic metrics include:

```text
Average hidden value:
  Whether agents discover high-value regions.

Average diversity:
  Whether agents remain spread out.

Average origin distance:
  Whether agents move away from the default origin.

Peak coverage:
  How many distinct hidden peaks are discovered.

Peak occupancy:
  Whether agents overcrowd the same peak.

Convergence rate:
  Whether pairwise distance decreases over rounds.

Collaboration efficiency:
  Whether read/share behavior improves final score.
```

These metrics are computed by the environment after the run. They are not shown
to agents during play.

## 9. Experiments

The first experiment compares collaboration regimes:

```text
no collaboration vs full collaboration vs random collaboration vs strategic collaboration.
```

The hypothesis is that full collaboration discovers high-scoring regions quickly
but can cause crowding, while no collaboration preserves diversity but may miss
good regions.

The second experiment varies `beta`, the diversity pressure. Low `beta` settings
mainly reward peak discovery. High `beta` settings make overcrowding more costly
and should increase the value of selective collaboration.

The third experiment varies `N`, the number of agents. Larger populations should
make crowding more severe.

The fourth experiment varies `K`, the number of hidden peaks. More peaks should
increase the opportunity for strategic agents to spread across multiple
high-scoring regions.

The fifth experiment varies the observation mechanism, including noisy positions,
delayed observations, and limited read/share budgets.

## 10. Expected Results

We expect score-following and full-collaboration agents to locate high-scoring
regions quickly, but also to concentrate many agents near the same visible
successes. This should produce high hidden value but low diversity and low peak
coverage.

We expect independent agents to maintain more diversity and cover more peaks, but
to discover high-value regions more slowly because they cannot use peer
information.

We expect strategic collaboration to perform best when diversity pressure is
moderate or high: agents use collaboration to find promising regions, then reduce
sharing and avoid crowded areas once a region becomes too visible.

## 11. Contribution

Black-Box Peak-Divergence Game contributes:

1. A hidden quality-diversity search landscape with multi-agent crowding.
2. A black-box feedback setting where agents do not know the scoring rule.
3. A collaboration mechanism where read/share decisions are strategic actions.
4. Metrics that separate score, hidden value, diversity, peak coverage, and convergence.

The benchmark directly tests whether collaboration helps agents learn useful
information or instead causes harmful imitation.

## 12. Conclusion

Black-Box Peak-Divergence Game studies collaboration under partial feedback.
Agents search for high-scoring regions in a hidden 10-dimensional landscape, but
they only observe total scores. Collaboration can reveal useful examples, but it
can also cause agents to crowd the same region. The central challenge is not just
to find a good point, but to decide how much information to use and reveal while
remaining differentiated from the population.

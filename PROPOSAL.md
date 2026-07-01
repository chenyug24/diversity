# Communication and Incentives in Multi-Agent Black-Box Optimization

## 1. Motivation

Many multi-agent systems assume that collaboration improves performance. However,
collaboration can also cause agents to imitate the same successful examples and
collapse into the same region. In black-box optimization tasks, this can waste
search capacity and prevent the system from finding multiple high-value
opportunities.

This project studies how communication and incentive structure affect the ability
of multiple agents to solve a hidden black-box optimization problem. Agents
search in a bounded continuous space containing hidden Gaussian peaks. The score
of an agent is based only on the hidden value of the point it selects, not on
explicit diversity. Diversity is measured only as a diagnostic variable that
helps explain exploration, convergence, and redundant search.

The central research question is:

```text
Under what communication and incentive conditions do multi-agent systems
discover the highest-value opportunities in a hidden black-box landscape?
```

The project compares cooperative and competitive settings. In the cooperative
setting, agents are instructed to help the whole system find the best possible
solution. In the competitive setting, agents try to maximize their own score and
outperform the others. Both settings use the same hidden landscape, but they
create different incentives for communication.

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

The system computes one internal reward quantity:

```text
V_i = hidden landscape value
```

The system's total score is:

```text
S_i = V_i.
```

Diversity is not directly rewarded. It is measured as a diagnostic variable that
helps explain whether agents explore broadly, converge prematurely, or duplicate
one another's search effort.

Origin distance can still be logged as a diagnostic variable, but it is not part
of the reward. This keeps the benchmark focused on practical black-box
optimization rather than rewarding movement away from an arbitrary default.

In the benchmark's main setting, this formula is **not revealed to agents**. The
formula is used by the environment and for evaluation, but agents only observe
their own total score and, when collaboration allows it, peer total scores.

## 4. Black-Box Information Setting

Agents know:

```text
- The action space is [0, 100]^10.
- Depending on the condition, they are optimizing cooperatively or competitively.
- They can choose visibility, request peer information, offer reciprocal exchange,
  and accept or reject incoming requests.
- They receive their own total score after each round.
```

Agents do not know:

```text
- The hidden peak locations.
- The scoring formula.
- Whether score comes from hidden value, diversity, or crowding.
- Population-level diagnostics such as average diversity or peak coverage.
```

When an agent observes a peer, it receives only:

```text
(peer_position, peer_total_score).
```

This turns the task from direct mathematical optimization into strategic search
under partial feedback. Agents must infer which regions are useful from observed
successes while deciding how much information to reveal or request.

## 5. Negotiated Communication Mechanism

Communication is modeled as a negotiation process rather than a fixed passive
observation setting. At each round, each agent chooses:

```text
visibility_i    in nonnegative integers, or all
request_count_i in nonnegative integers, or all
offer_reciprocal_i in {true, false}
accept_probability_i in [0, 1]
```

`visibility_i` controls how many peers can initially see the agent's current
position and total score. `request_count_i` controls how many peers the agent
asks for information. If `offer_reciprocal_i` is true, the agent offers to reveal
its own information in exchange. The target agent accepts or rejects incoming
requests according to its communication policy.

Integer communication values larger than the number of available peers are
capped at all available peers.

This mechanism separates information availability from negotiated information
exchange. Communication is therefore an endogenous strategic action rather than a
fixed experimental condition.

## 6. Iterative Procedure

1. The environment initializes `N` agents.
2. The hidden peaks are generated but not revealed.
3. Each agent submits an initial 10-dimensional point.
4. The environment computes hidden value, diversity, and total score.
5. Each agent observes only its own total score.
6. Each agent chooses an initial visibility level.
7. Each agent may request information from selected peers and offer reciprocal exchange.
8. Target agents accept or reject incoming requests.
9. The environment reveals initially visible information and accepted exchanges.
10. All agents update their points simultaneously using only black-box feedback and observed peer states.
11. Steps 4-10 repeat for `T` rounds.
12. Final scores and diagnostic metrics are computed by the environment.

The simultaneous-update condition is important. Within a round, no agent sees
another agent's newly generated point until the next round. In API-based
implementations, all model calls for a round can be dispatched concurrently, but
their outputs are applied only after the full set of agent decisions returns.

## 7. Initial Public Publication Case

The first experimental case should be deliberately simple. Before adding richer
communication mechanisms such as random observation or negotiated exchange,
agents should be tested in a public research-publication setting.

The setting is:

```text
- Each agent chooses a research location in the continuous search space.
- The environment returns that agent's true value at the location.
- The agent may optionally publish its current location and true value.
- Published records enter a public registry visible to all agents.
- There is no private exchange, no selective messaging, and no negotiation.
- A location already in the public registry cannot be submitted again exactly.
```

The public registry represents prior work. Once a research direction has been
published, another agent cannot claim the exact same direction again. Because the
space is continuous, nearby locations are still allowed in the first version;
only exact reuse of a published location is blocked.

In this case, the environment gives each agent only:

```text
- its own current location,
- its own current true score,
- its own recent location/score history,
- the public registry of published location/score pairs.
```

The hidden peaks, global optimum, score formula, and population-level diagnostics
remain hidden from the agents. This case isolates the basic mechanism: given only
publicly published research feedback, do agents learn useful directions, avoid
exact duplication, and decide when publication helps or hurts future search?

The main success condition for this research scenario is top-peak coverage. If
the landscape contains `K` peaks, the evaluator ranks all peaks by hidden height
and selects the top three:

```text
TopPeaks = highest 3 peaks by h_m, or all peaks if K < 3.
```

The group succeeds when it discovers these top peaks. A top peak is counted as
discovered if at least one agent reaches a point whose contribution from that
peak is at least a fixed fraction of the peak height, for example `0.9 * h_m`.
This means the benchmark rewards discovering the best research directions, not
only finding one single best point.

## 8. Baselines

The benchmark compares:

```text
Random:
  Samples a fresh random point each round.

Random Corner:
  Chooses a random hypercube corner.

Origin Maximizer:
  Always chooses (100, ..., 100).

Independent Search:
  Never communicates; locally searches from its own score history.

Score Following:
  Reveals broadly, requests broadly, and follows the highest observed total score.

Diversity Only:
  Uses observed positions to move away from visible agents, ignoring scores.

Full Collaboration:
  All agents reveal broadly, request broadly, and accept exchanges each round.

Random Collaboration:
  Randomly chooses negotiation actions and builds a black-box score surrogate.

Strategic Collaboration:
  Requests more when its own score is weak, reveals less when its score is strong,
  and trades off high-score imitation against crowd avoidance.

LLM Black-Box Agent:
  Uses a language model API to choose positions from the same black-box
  observations: own position, own total score, and observed peer
  (position, total score) pairs.
```

## 9. Evaluation Metrics

The primary metric is top-peak coverage:

```text
top_peak_coverage = discovered_top_peaks / min(3, K).
```

By default, a top peak is considered discovered if:

```text
V_m(z_i,t) >= 0.9 * h_m
```

for at least one agent `i` and round `t`. Full success means:

```text
top_peak_coverage = 1.
```

The benchmark also reports the highest value discovered by any agent across all
rounds:

```text
best_value_found = max_{i,t} V(z_{i,t}).
```

Because the evaluator knows the hidden peak heights, this can be normalized by
the true global optimum:

```text
best_value_found_ratio = best_value_found / max_peak_height.
best_value_found_gap = max_peak_height - best_value_found.
```

We also report a population-average system optimization index:

```text
system_optimization = mean_score / max_peak_height.
```

This gives a 0-to-1 indicator of how close the population's average score is to
the known global optimum. It is useful diagnostically, but it is not the main
success metric when the research question is whether the multi-agent system
covers the best opportunities.

Diagnostic metrics include:

```text
Average hidden value:
  Whether agents discover high-value regions.

Average diversity:
  Whether agents remain spread out.

Average origin distance:
  Diagnostic only; whether agents drift toward high-coordinate regions.

Peak coverage:
  How many distinct hidden peaks are discovered.

Peak occupancy:
  Whether agents overcrowd the same peak.

Convergence rate:
  Whether pairwise distance decreases over rounds.

Collaboration efficiency:
  Whether negotiated communication improves final score.

Negotiation metrics:
  Initial visibility, request count, acceptance rate, rejection rate,
  reciprocal offer rate, reciprocal exchange rate, and observed peer count.
```

These metrics are computed by the environment after the run. They are not shown
to agents during play.

## 10. Experiments

The first experiment compares collaboration regimes:

```text
no communication vs full communication vs random negotiation vs strategic negotiation.
```

The hypothesis is that full collaboration discovers high-scoring regions quickly
but can cause crowding, while no collaboration preserves diversity but may miss
good regions.

The second experiment varies `N`, the number of agents. Larger populations should
make crowding more severe.

The third experiment varies `K`, the number of hidden peaks. More peaks should
increase the opportunity for strategic agents to spread across multiple
high-scoring regions.

The fourth experiment varies the observation mechanism, including noisy positions,
delayed observations, and limited communication budgets.

## 11. Expected Results

We expect score-following and full-collaboration agents to locate high-scoring
regions quickly, but also to concentrate many agents near the same visible
successes. This should produce high hidden value but low diversity and low peak
coverage.

We expect independent agents to maintain more diversity and cover more peaks, but
to discover high-value regions more slowly because they cannot use peer
information.

We expect strategic communication to perform best when agents use visibility and
inspection decisions to learn from promising regions without prematurely
collapsing into redundant search.

## 12. Contribution

Black-Box Peak-Divergence Game contributes:

1. A hidden value landscape for multi-agent black-box optimization.
2. A black-box feedback setting where agents do not know the scoring rule.
3. A negotiation mechanism where visibility, requests, reciprocal offers, and accept/reject decisions are strategic actions.
4. Metrics that separate global value, communication behavior, diagnostic diversity, and convergence.

The benchmark directly tests whether collaboration helps agents learn useful
information or instead causes harmful imitation.

## 13. Conclusion

This project studies communication and incentives under partial feedback. Agents
search for high-scoring regions in a hidden 10-dimensional landscape, but they
only observe total scores. Negotiated communication can reveal useful examples,
but it can also create imitation or strategic withholding. The central challenge
is to understand when cooperative or competitive agents, under different
communication structures, approach the global optimum.
